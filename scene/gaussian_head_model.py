from scene.gaussian_model import *
from scene.flame import blend_shapes,batch_rodrigues,batch_rigid_transform
import os,pytorch3d.ops
import torch.nn.functional as F

from net_modules.flame_params_net_smirk import FlameParamsNetSmirk
from utils.sh_utils import RGB2SH
from net_modules.NVDIFFREC.envmap import EnvironmentMap
from utils.general_utils import get_minimum_axis

from roma import rotmat_to_unitquat, quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
import time

class GaussianHeadModel(GaussianModel):
    def __init__(self, sh_degree : int,args):
        super().__init__(sh_degree,args)
        self.device=args.device
        # self.preload=args.preload
        self.flame_vertexes=torch.empty(0)
        self.shape_dirs=torch.empty(0)
        self.expression_dirs=torch.empty(0)
        self.pose_dirs=torch.empty(0)
        self.lbs_weights=torch.empty(0)
        self.J_regressor=torch.empty(0)
        self.rot_parents=torch.empty(0)
        self.triangles=torch.empty(0)
        self.flame_scale=torch.empty(0)
        
        self.cache_xyz_canonical=None
        self.eval=False
        
        self.scaling_offset=None
        self.cached_shaped_vertexes=False
        self.lbs_return_transform_quad=False
        self.shaped_vertexes=None
        self.d_deform_scaling=None
        
        self.color_precomp=args.color_precomp
        self.with_param_net_smirk=args.with_param_net_smirk

        if self.with_param_net_smirk:
            self.flame_params_net=FlameParamsNetSmirk(**args.flame_params_net_params).to(self.device)

        # Optional causal One-Euro smoother for SMIRK encoder outputs.
        # Off by default. set_smirk_smoother() (called from render.py when
        # --jitter_filter --jitter_filter_smirk is given) installs a buffer
        # that smooths expression / jaw / eyelid in temporal order.
        self._smirk_smoother = None
            
        self.with_depth_supervise=args.with_depth_supervise
        self.warm_up_iter=args.warm_up_iter
        
        self.Envmap=EnvironmentMap(args.diffuse_resolution,args.specular_resolution,args.mip_level,device=self.device)
        self.inverse_albedo_activation,self.inverse_roughness_activation = inverse_sigmoid,inverse_sigmoid
        self.albedo_activation,self.roughness_activation,self.reflectance_activation =torch.sigmoid, torch.sigmoid, torch.sigmoid
        self.max_reflectance,self.max_roughness=args.max_reflectance,args.max_roughness
        self.min_reflectance,self.min_roughness=args.min_reflectance,args.min_roughness

    @property
    def get_deform_scaling(self):
        _scaling=self._scaling
        self.deform_scaling=self.scaling_activation(_scaling)*self.d_deform_scaling *self.flame_scale
        return self.deform_scaling
    
    @property
    def get_canonical_scaling(self):
        return self.get_scaling
        
    @property
    def get_deform_rotation(self):
        
        rot=self.rotation_activation(self._rotation)
        self.deform_rotation=self.rotation_activation(quat_xyzw_to_wxyz(quat_product(self.d_deform_rotation_xyzw,quat_wxyz_to_xyzw(rot))))
        return self.deform_rotation
    
    @property
    def get_deform_xyz(self):
        return self.deform_xyz*self.flame_scale
    

    @property
    def get_deform_opacity(self):
        return self.opacity_activation(self._opacity)+self.d_deform_opacity
    
    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.deform_scaling, self.deform_rotation)
    

    @property
    def get_albedo(self):
        return self.albedo_activation(self._albedo)
    
    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)*(self.max_roughness-self.min_roughness)+self.min_roughness
    
    @property
    def get_reflectance(self):
        return self.reflectance_activation(self._reflectance)* \
            (self.max_reflectance-self.min_reflectance)+self.min_reflectance
    
    def get_deform_normal(self,cam_o):

        self.deform_normal=self.get_min_axis(cam_o,self.get_deform_rotation)
        return self.deform_normal
        
    def get_canonical_normal(self,cam_o):

        self._normal=self.get_min_axis(cam_o,self.get_rotation)
        return self._normal
 
    
    def create_from_pcd(self, head_scene_info, spatial_lr_scale,shape_params):
        super().create_from_pcd(head_scene_info.point_cloud, spatial_lr_scale)
        self.shape_param=shape_params.detach().clone()
        flame_mesh=head_scene_info.flame_mesh
        

        self.flame_scale=nn.Parameter(torch.tensor(head_scene_info.flame_scale,device=self.device,requires_grad=True))
        self.flame_vertexes=nn.Parameter(flame_mesh["v_template"].to(self.device).requires_grad_(False))
        self.shape_dirs=nn.Parameter(flame_mesh["shape_dirs"].to(self.device).requires_grad_(False))
        self.expression_dirs=nn.Parameter(flame_mesh["expression_dirs"].to(self.device).requires_grad_(False))
        self.pose_dirs=nn.Parameter(flame_mesh["pose_dirs"].to(self.device).requires_grad_(False))
        self.lbs_weights=nn.Parameter(flame_mesh["lbs_weights"].to(self.device).requires_grad_(False))
        self.J_regressor=nn.Parameter(flame_mesh["J_regressor"].T.to(self.device).requires_grad_(False))
        self.rot_parents=flame_mesh["parents"].to(self.device).long().requires_grad_(False)
        
        num_flame_vertexes=flame_mesh["v_template"].shape[0]
        self.triangles,self.flame_triangles=flame_mesh["triangles"],flame_mesh["flame_triangles"]
        
        r_eyelid_dirs,l_eyelid_dirs=torch.zeros_like(self.lbs_weights[:,:3]),torch.zeros_like(self.lbs_weights[:,:3])
        r_eyelid_dirs[:flame_mesh["r_eyelid"].shape[1],:]=flame_mesh["r_eyelid"]
        l_eyelid_dirs[:flame_mesh["l_eyelid"].shape[1],:]=flame_mesh["l_eyelid"]
        self.r_eyelid_dirs=nn.Parameter(r_eyelid_dirs.to(self.device).requires_grad_(False))
        self.l_eyelid_dirs=nn.Parameter(l_eyelid_dirs.to(self.device).requires_grad_(False))
        
        
        betas = shape_params
        self.flame_vertexes_shaped=(blend_shapes(betas,self.shape_dirs)+self.flame_vertexes).detach()
        
        self.flame_shape_dirs=self.shape_dirs.detach().clone()
        self.flame_expression_dirs=self.expression_dirs.detach().clone()
        self.flame_pose_dirs=self.pose_dirs.detach().clone()
        self.flame_lbs_weights=self.lbs_weights.detach().clone()
        self.flame_J_regressor=self.J_regressor.detach().clone()
        self.flame_r_eye_dirs=self.r_eyelid_dirs.detach().clone()
        self.flame_l_eye_dirs=self.l_eyelid_dirs.detach().clone()
        self.flame_vertex_normal=nn.Parameter(flame_mesh["v_normal"].to(self.device).requires_grad_(False))
        self.flame_vertex_idx_mask=flame_mesh["vertex_idx_mask"]
        shape_dirs=head_scene_info.point_cloud.shape_dirs.detach().clone().to(self.device)
        expression_dirs=head_scene_info.point_cloud.expression_dirs.detach().clone().to(self.device)
        pose_dirs=head_scene_info.point_cloud.pose_dirs.detach().clone().to(self.device)
        r_eyelid_dirs=head_scene_info.point_cloud.r_eyelid_dirs.detach().clone().to(self.device)
        l_eyelid_dirs=head_scene_info.point_cloud.l_eyelid_dirs.detach().clone().to(self.device)
        lbs_weights=head_scene_info.point_cloud.lbs_weights.detach().clone().to(self.device)
        self.flame_joint_center=torch.einsum('bik,ij->bjk', [self.flame_vertexes_shaped, self.flame_J_regressor])
        
        self.shape_dirs=nn.Parameter(shape_dirs,requires_grad=True)
        self.expression_dirs=nn.Parameter(expression_dirs,requires_grad=True)
        self.pose_dirs=nn.Parameter(pose_dirs,requires_grad=True)
        
        self.lbs_weights=nn.Parameter(lbs_weights,requires_grad=True)
        self.r_eyelid_dirs=nn.Parameter(r_eyelid_dirs,requires_grad=True)
        self.l_eyelid_dirs=nn.Parameter(l_eyelid_dirs,requires_grad=True)
        torch.cuda.empty_cache()
            

        self._albedo = nn.Parameter((torch.ones((self._xyz.shape[0], 3), device="cuda")*0.0).requires_grad_(True))
        _roughness=inverse_sigmoid(torch.ones((self._xyz.shape[0], 1), device="cuda")*0.9)
        _reflectance=inverse_sigmoid(torch.ones((self._xyz.shape[0], 1), device="cuda")*0.04)
        self._roughness = nn.Parameter(_roughness.requires_grad_(True))
        self._reflectance = nn.Parameter(_reflectance.requires_grad_( True))
            
    def training_setup(self, training_args):
        super().training_setup(training_args)
        self.optimizer.add_param_group({"params":[self.flame_scale],"lr":training_args.flame_scale_lr,"name":"flame_scale"})
       
        shape_dirs_params={"params":[self.shape_dirs],"lr":training_args.shape_dirs_lr,"name":"shape_dirs"}
        r_eyelid_dirs_params={"params":[self.r_eyelid_dirs],"lr":training_args.expression_dirs_lr,"name":"r_eyelid_dirs"}
        l_eyelid_dirs_params={"params":[self.l_eyelid_dirs],"lr":training_args.expression_dirs_lr,"name":"l_eyelid_dirs"}
        
        self.optimizer.add_param_group(shape_dirs_params)
        self.optimizer.add_param_group(r_eyelid_dirs_params)
        self.optimizer.add_param_group(l_eyelid_dirs_params)
        self.prune_params_names.append("shape_dirs")
        self.prune_params_names.append("l_eyelid_dirs")
        self.prune_params_names.append("r_eyelid_dirs")
        
        expression_dirs_params={"params":[self.expression_dirs],"lr":training_args.expression_dirs_lr,"name":"expression_dirs"}
        pose_dirs_params={"params":[self.pose_dirs],"lr":training_args.pose_dirs_lr,"name":"pose_dirs"}
        lbs_weights_params={"params":[self.lbs_weights],"lr":training_args.lbs_weights_lr,"name":"lbs_weights"}

        self.optimizer.add_param_group(expression_dirs_params)
        self.optimizer.add_param_group(pose_dirs_params)
        self.optimizer.add_param_group(lbs_weights_params)

        self.prune_params_names.append("expression_dirs")
        self.prune_params_names.append("pose_dirs")
        self.prune_params_names.append("lbs_weights")

        
        if self.with_param_net_smirk:
            flame_params_net_smirk_enocder_params={"params":self.flame_params_net.expression_encoder.encoder.parameters(),"lr":training_args.flame_params_net_smirk_encoder_lr,"name":"flame_params_net_smirk_encoder"}
            self.optimizer.add_param_group(flame_params_net_smirk_enocder_params)
            flame_params_net_smirk_decoder_params={"params":self.flame_params_net.expression_encoder.expression_layers.parameters(),"lr":training_args.flame_params_net_smirk_decoder_lr,"name":"flame_params_net_smirk_decoder"}
            self.optimizer.add_param_group(flame_params_net_smirk_decoder_params)
        
        self.shape_param=nn.Parameter(self.shape_param,requires_grad=False)
        
       
        
        self.optimizer.add_param_group({'params': list(self.Envmap.parameters()), 'lr': training_args.envmap_lr, "name": "Envmap"})
        
        self.optimizer.add_param_group({'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"})
        self.optimizer.add_param_group({'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"})
        self.optimizer.add_param_group({'params': [self._reflectance], 'lr': training_args.reflectance_lr, "name": "reflectance"})
        self.prune_params_names.extend(["roughness","albedo","reflectance"])
    
        self.albedo_schedule = get_expon_lr_func(lr_init=training_args.albedo_lr,
                                                lr_final=training_args.albedo_lr*training_args.appear_lr_decay,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)
        self.roughness_schedule = get_expon_lr_func(lr_init=training_args.roughness_lr,
                                                lr_final=training_args.roughness_lr*training_args.appear_lr_decay,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)
        self.reflectance_schedule = get_expon_lr_func(lr_init=training_args.reflectance_lr,
                                                lr_final=training_args.reflectance_lr*training_args.appear_lr_decay,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)
        self.envmap_schedule = get_expon_lr_func(lr_init=training_args.envmap_lr,
                                                lr_final=training_args.envmap_lr*training_args.env_map_lr_decay,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)
            
        
    
    def lbs_v2(self,xyz,shape_params=None, expression_params=None,full_pose_params=None,
            eyelid_params=None,translation_param=None,lbs_weight_t=None,):
        #Performs Linear Blend Skinning with the given shape and pose parameters        
        batch_size = shape_params.shape[0]
        device=self.device
        transform_dict={}
        #Full_ Pose [(global)3, neck (0)3, (jaw)3, eyepose (0)6]
        full_pose=full_pose_params
        expression_dirs,pose_dirs=self.expression_dirs,self.pose_dirs
        
        if self.eval:
            if self.cache_xyz_canonical is not None:
                xyz_canonical=self.cache_xyz_canonical
            else:
                xyz_canonical=xyz.unsqueeze(0).expand(batch_size, -1, -1)
                xyz_canonical=xyz_canonical+blend_shapes(shape_params,self.shape_dirs)
                self.cache_xyz_canonical=xyz_canonical.detach().clone()
            
        else :
            xyz_canonical=xyz.unsqueeze(0).expand(batch_size, -1, -1)
            xyz_canonical=xyz_canonical+blend_shapes(shape_params,self.shape_dirs)
        
        shape_dirs_t=expression_dirs#torch.cat([self.shape_dirs,expression_dirs],dim=2)
        betas = expression_params
        
        if lbs_weight_t is None:
            lbs_weight_t=self.lbs_weights
        pose_dirs_t=pose_dirs
        r_eyelid_dirs_t,l_eyeild_dirs_t=self.r_eyelid_dirs,self.l_eyelid_dirs
            
        shape_offsets=blend_shapes(betas,shape_dirs_t) # n 3 300+100
        xyz_shaped = xyz_canonical + shape_offsets

        # Get the joints 5 n
        J=self.flame_joint_center
        ident = torch.eye(3, dtype=torch.float32, device=device)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3), dtype=torch.float32).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])# 1 36
        # (b x l) x (m, 3 ,l) -> b x m x 3
        
        pose_offsets =blend_shapes(pose_feature,pose_dirs_t)
        xyz_posed = pose_offsets + xyz_shaped
        if eyelid_params is not None:
            xyz_posed = xyz_posed + r_eyelid_dirs_t.unsqueeze(0).expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            xyz_posed = xyz_posed + l_eyeild_dirs_t.unsqueeze(0).expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]
        
        J_transformed, A = batch_rigid_transform(rot_mats, J, self.rot_parents, dtype=torch.float32)
        #add identity matrix
        A=torch.cat([A,torch.eye(4,device=device)[None,None,...].expand(batch_size,-1,-1,-1)],dim=1)
        W = lbs_weight_t.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        #add identity rotation joint
        num_joints = self.J_regressor.shape[1]+1
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
        homogen_coord = torch.ones([batch_size, xyz_posed.shape[1], 1],
                                dtype=torch.float32, device=device)
        v_posed_homo = torch.cat([xyz_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        xyz_lbs = v_homo[:, :, :3, 0]
        if translation_param is not None:
            xyz_lbs=xyz_lbs+translation_param
        transform_dict.update({"transform_matrix":T,"shape_offsets":shape_offsets,"pose_offsets":pose_offsets,"xyz_posed":xyz_posed,"canonical":xyz_canonical})
        
        if self.lbs_return_transform_quad:
            A3=A[0,:,:3,:3]
            A3_quad=rotmat_to_unitquat(A3)
            T_quad=torch.matmul(W, A3_quad.view(batch_size, num_joints, 4)).view(batch_size, -1, 4)
            transform_dict["transform_matrix_quad"]=T_quad
        if not self.eval :#and self.with_normal_attribute:
            expr_offsets=blend_shapes(expression_params,self.flame_expression_dirs)
            pose_offsets=blend_shapes(pose_feature,self.flame_pose_dirs)
            eyelid_offset=0.0
            if eyelid_params is not None:
                eyelid_offset=self.flame_l_eye_dirs.unsqueeze(0).expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]\
                    +self.flame_r_eye_dirs.unsqueeze(0).expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            flame_vertex_posed=self.flame_vertexes_shaped.expand(batch_size, -1, -1)+expr_offsets+pose_offsets+eyelid_offset
            flame_W = self.flame_lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
            flame_T = torch.matmul(flame_W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
            flame_homogen_coord = torch.ones([batch_size, flame_vertex_posed.shape[1], 1],dtype=torch.float32, device=device)
            flame_posed_homo = torch.cat([flame_vertex_posed, flame_homogen_coord], dim=2)
            flame_v_homo = torch.matmul(flame_T, torch.unsqueeze(flame_posed_homo, dim=-1))
            self.flame_vertex_lbs = flame_v_homo[:, :, :3, 0].squeeze(0)
            
        return xyz_lbs,transform_dict

    
    
    def set_smirk_smoother(self, smoother):
        """Install / clear a CausalOneEuroBuffer for SMIRK output smoothing.

        Pass ``None`` to disable. Reset the buffer between independent render
        sequences (e.g. train split, test split, multi-view trajectory) by
        calling ``smoother.reset()`` before re-attaching.
        """
        self._smirk_smoother = smoother

    def _smooth_smirk_outputs(self, expression_param, jaw_params, eyelid_param):
        """Apply the installed One-Euro buffer to SMIRK encoder outputs.

        Operates on the first (B==1) sample. For B > 1 the smoother is
        applied per-sample in order; this matches the renderer's per-frame
        loop. Returns tensors on the original device/dtype.
        """
        smoother = self._smirk_smoother
        if smoother is None:
            return expression_param, jaw_params, eyelid_param

        def _apply(name, t):
            if t is None:
                return None
            arr = t.detach().cpu().numpy()
            out = smoother.filter(name, arr)
            return torch.as_tensor(out, dtype=t.dtype, device=t.device)

        return (
            _apply("expression", expression_param),
            _apply("jaw", jaw_params),
            _apply("eyelid", eyelid_param),
        )

    def forward(self,shape_param,expression_param,full_pose_param,camera_center,eyelid_param=None,translation_param=None,warped_image=None,iteration=torch.inf):
        
        if self.shape_param is not None:
            shape_param=self.shape_param
        if self.with_param_net_smirk and warped_image is not None:
            start_time=time.time()
            out_params=self.flame_params_net(warped_image)
            end_time=time.time()
            self.params_extract_time=(end_time-start_time)
            expression_param = out_params['expression_params']
            jaw_params = out_params.get('jaw_params', None)
            eyelid_param = out_params.get('eyelid_params', None)
            # Optional render-time smoothing of SMIRK outputs (per-sequence
            # causal One-Euro). Active only when render.py installed a
            # smoother. Training code path leaves _smirk_smoother=None so the
            # gradient flow through the encoder is unchanged.
            if self._smirk_smoother is not None:
                expression_param, jaw_params, eyelid_param = self._smooth_smirk_outputs(
                    expression_param, jaw_params, eyelid_param)
            #[(global)3, neck (0)3, (jaw)3, eyepose (0)6]
            #use_smirk_jaw_pose
            self.d_jaw_params=(full_pose_param[:,6:9]-jaw_params)
            full_pose_param[:,6:9]=jaw_params
            
        _xyz_t=self._xyz
           
        lbs_weights_exp=torch.relu(self.lbs_weights)
        self.lbs_weights_t=lbs_weights_exp / (torch.sum(lbs_weights_exp, dim=-1, keepdim=True)+1e-5)#F.softmax(self.lbs_weights,dim=1)

        _xyz_lbs,transform_dict=self.lbs_v2(_xyz_t,shape_param,expression_param,full_pose_param,
                                            eyelid_param,translation_param,lbs_weight_t=self.lbs_weights_t)
        self.deform_xyz=_xyz_lbs.squeeze(0)
        self.d_deform_rotation_xyzw=rotmat_to_unitquat(transform_dict["transform_matrix"][0,:,:3,:3])
        self.d_deform_scaling=1.0
        self.transform_dict=transform_dict

        self.d_deform_opacity=torch.tensor(0.,device=self.device)

            
    def save_model(self, path):
        
        super().save_ply(os.path.join(path,"point_cloud.ply"))
        atrributes_params_dict={}
        
        atrributes_params_dict["flame_scale"]=self.flame_scale
        atrributes_params_dict["flame_vertexes"]=self.flame_vertexes
        atrributes_params_dict["shape_dirs"]=self.shape_dirs
        atrributes_params_dict["expression_dirs"]=self.expression_dirs
        atrributes_params_dict["pose_dirs"]=self.pose_dirs
        atrributes_params_dict["lbs_weights"]=self.lbs_weights
        atrributes_params_dict["J_regressor"]=self.J_regressor
        atrributes_params_dict["rot_parents"]=self.rot_parents
        atrributes_params_dict["shape_param"]=self.shape_param
        atrributes_params_dict["flame_joint_center"]=self.flame_joint_center
        
        
        atrributes_params_dict["shape_dirs"]=self.shape_dirs
        atrributes_params_dict["expression_dirs"]=self.expression_dirs
        atrributes_params_dict["pose_dirs"]=self.pose_dirs
        atrributes_params_dict["lbs_weights"]=self.lbs_weights
        atrributes_params_dict["r_eyelid_dirs"]=self.r_eyelid_dirs
        atrributes_params_dict["l_eyelid_dirs"]=self.l_eyelid_dirs
        atrributes_params_dict["flame_J_regressor"]=self.flame_J_regressor
        atrributes_params_dict["flame_vertexes_shaped"]=self.flame_vertexes_shaped
            
        if self.with_param_net_smirk:
            torch.save(self.flame_params_net.state_dict(), os.path.join(path,"flame_params_net.pth"))
       

        atrributes_params_dict["albedo"]=self._albedo
        atrributes_params_dict["roughness"]=self._roughness
        atrributes_params_dict["reflectance"]=self._reflectance

        atrributes_params_dict["Envmap-diffuse_map"],atrributes_params_dict["Envmap-diffuse"]=self.Envmap.diffuse_map,self.Envmap.diffuse
        atrributes_params_dict["Envmap-specular_map"],atrributes_params_dict["Envmap-specular"]=self.Envmap.specular_map,self.Envmap.specular
        atrributes_params_dict["max_reflectance"],atrributes_params_dict["min_reflectance"]=self.max_reflectance,self.min_reflectance
        atrributes_params_dict["max_roughness"],atrributes_params_dict["min_roughness"]=self.max_roughness,self.min_roughness

        torch.save(atrributes_params_dict, os.path.join(path,"attributes_params.pth"))
            
    def load_model(self, path):
        super().load_ply(os.path.join(path,"point_cloud.ply"))
        atrributes_params_dict=torch.load(os.path.join(path,"attributes_params.pth"),map_location="cpu", weights_only=False)
        
        self.flame_scale=nn.Parameter(atrributes_params_dict["flame_scale"].to(self.device),requires_grad=True)
        self.flame_vertexes=nn.Parameter(atrributes_params_dict["flame_vertexes"].to(self.device),requires_grad=False)
        self.shape_dirs=nn.Parameter(atrributes_params_dict["shape_dirs"].to(self.device),requires_grad=False)
        self.expression_dirs=nn.Parameter(atrributes_params_dict["expression_dirs"].to(self.device),requires_grad=False)
        self.pose_dirs=nn.Parameter(atrributes_params_dict["pose_dirs"].to(self.device),requires_grad=False)
        self.lbs_weights=nn.Parameter(atrributes_params_dict["lbs_weights"].to(self.device),requires_grad=False)
        self.J_regressor=nn.Parameter(atrributes_params_dict["J_regressor"].to(self.device),requires_grad=False)
        self.rot_parents=atrributes_params_dict["rot_parents"].to(self.device).long()
        if "flame_joint_center" in atrributes_params_dict.keys():
            self.flame_joint_center=nn.Parameter(atrributes_params_dict["flame_joint_center"].to(self.device),requires_grad=False)
        if "shape_param" in atrributes_params_dict.keys():
            self.shape_param=nn.Parameter(atrributes_params_dict["shape_param"].to(self.device),requires_grad=False)
            
        
        self.shape_dirs=atrributes_params_dict["shape_dirs"].to(self.device)
        self.expression_dirs=atrributes_params_dict["expression_dirs"].to(self.device)
        self.pose_dirs=atrributes_params_dict["pose_dirs"].to(self.device)
        self.lbs_weights=atrributes_params_dict["lbs_weights"].to(self.device)
    
        self.r_eyelid_dirs=atrributes_params_dict["r_eyelid_dirs"].to(self.device)
        self.l_eyelid_dirs=atrributes_params_dict["l_eyelid_dirs"].to(self.device)
        self.flame_J_regressor=atrributes_params_dict["flame_J_regressor"]
        self.flame_vertexes_shaped=atrributes_params_dict["flame_vertexes_shaped"]

        
        if  self.with_param_net_smirk:
            statedict=torch.load( os.path.join(path,"flame_params_net.pth"),map_location="cpu", weights_only=False)
            self.flame_params_net.load_state_dict(statedict)
            self.flame_params_net.to(self.device)
        

        self._albedo=nn.Parameter(atrributes_params_dict["albedo"].to(self.device),requires_grad=True)
        self._roughness=nn.Parameter(atrributes_params_dict["roughness"].to(self.device),requires_grad=True)
        self._reflectance=nn.Parameter(atrributes_params_dict["reflectance"].to(self.device),requires_grad=True)

        self.Envmap.diffuse_map=nn.Parameter(atrributes_params_dict["Envmap-diffuse_map"].to(self.device),requires_grad=True)
        self.Envmap.diffuse=atrributes_params_dict["Envmap-diffuse"].clone().detach().to(self.device)
        self.Envmap.specular_map=nn.Parameter(atrributes_params_dict["Envmap-specular_map"].to(self.device),requires_grad=True)
        self.Envmap.specular=atrributes_params_dict["Envmap-specular"].clone().detach().to(self.device)
        self.max_reflectance,self.min_reflectance=atrributes_params_dict["max_reflectance"],atrributes_params_dict["min_reflectance"]
        self.max_roughness,self.min_roughness=atrributes_params_dict["max_roughness"],atrributes_params_dict["min_roughness"]
        
    def get_canonical_xyz(self):
        #save canonical xyz
        canonical_xyz=self._xyz
        return canonical_xyz
    
    def save_canonical_ply(self,path="./canonical_xyz.ply"):

        canonical_xyz=self.get_canonical_xyz()
        
        normals = np.zeros_like(canonical_xyz.detach().detach().cpu().numpy())
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc =  RGB2SH(self.albedo_activation(self._albedo)).detach().contiguous().cpu().numpy()
        
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        canonical_xyz=canonical_xyz.detach().cpu().numpy()
        elements = np.empty(canonical_xyz.shape[0], dtype=dtype_full)
        if f_rest.shape[0]!=canonical_xyz.shape[0]:
            f_rest=np.zeros((canonical_xyz.shape[0],f_rest.shape[1]))
        attributes = np.concatenate((canonical_xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        

    def cached_shaped_vertex(self,shape_param):
        vertexes=self.flame_vertexes
        shape_dirs_t=self.shape_dirs
        betas=shape_param
        self.shaped_vertexes=(blend_shapes(betas,shape_dirs_t).squeeze()+vertexes).detach()
        self.cached_shaped_vertexes=True
    
    def set_eval(self,state):
        self.eval=state
        if not state:
            self.cache_xyz_canonical=None

    def extra_loss(self):
        loss_dict={"d_opacity":0,"d_rotation":0,"d_scaling":0,"d_xyz":0}
        return loss_dict
    
    def get_min_axis(self, cam_o,rotation):
        pts = self.get_xyz
        p2o = cam_o[None] - pts
        scales = self.deform_scaling
        min_axis_id = torch.argmin(scales, dim = -1, keepdim=True)
        min_axis = torch.zeros_like(scales).scatter(1, min_axis_id, 1)
        rot_matrix = build_rotation(rotation)
        ndir = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)
        neg_msk = torch.sum(p2o*ndir, dim=-1) < 0
        ndir[neg_msk] = -ndir[neg_msk] # make sure normal orient to camera
        #ndir=get_minimum_axis(self.deform_scaling,rotation)

        return ndir
    