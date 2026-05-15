
import torch
from scene import Head_Scene
import os, imageio
from tqdm import tqdm
from os import makedirs
import gaussian_renderer
import torchvision
from utils.general_utils import safe_state,to8b,save_image_L

from utils.camera_utils import generate_multi_view_poses
from utils.graphics_utils import normal_from_depth_image
from scene.data_loader import TrackedData
from utils.onefilter import CausalOneEuroBuffer
from net_modules.NVDIFFREC.util import cubemap_to_latlong

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,add_more_argument
from gaussian_renderer import GaussianHeadModel
import numpy as np
from net_modules.NVDIFFREC.envmap import EnvironmentMap_relight
from copy import deepcopy
from natsort import natsorted

def _maybe_attach_smirk_smoother(gaussians, all_args):
    """Install a causal One-Euro buffer on `gaussians` for SMIRK smoothing.

    Returns the smoother (so the caller can reset it between sequences) or
    ``None`` if smoothing is disabled / not applicable.
    """
    if not getattr(all_args, "jitter_filter", False):
        return None
    if not getattr(all_args, "jitter_filter_smirk", False):
        return None
    if not getattr(gaussians, "with_param_net_smirk", False):
        return None
    smoother = CausalOneEuroBuffer(
        fps=getattr(all_args, "jitter_filter_fps", 30.0),
        min_cutoff=getattr(all_args, "jitter_filter_min_cutoff", 1.0),
        beta=getattr(all_args, "jitter_filter_beta", 0.0),
        d_cutoff=getattr(all_args, "jitter_filter_d_cutoff", 1.0),
    )
    gaussians.set_smirk_smoother(smoother)
    print(f"[render] SMIRK output smoothing enabled "
          f"(min_cutoff={smoother.min_cutoff} Hz, beta={smoother.beta}, "
          f"fps={smoother.fps})")
    return smoother


def _reset_smirk_smoother(gaussians, smoother):
    """Reset buffer state at sequence boundary so causal filter restarts."""
    if smoother is not None:
        smoother.reset()
    elif gaussians is not None and hasattr(gaussians, "set_smirk_smoother"):
        # No-op if smoother is None but keeps API symmetric.
        pass


def _detach_smirk_smoother(gaussians):
    if gaussians is not None and hasattr(gaussians, "set_smirk_smoother"):
        gaussians.set_smirk_smoother(None)


def render_set(model_path, name, epoch, cam_params, gaussians, pipeline, background,args):
    render=gaussian_renderer.render_with_deferred
    
    device=gaussians.device
    render_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(epoch), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if "render_albedo"in args.__dict__.keys() and args.render_albedo:
        render_albedo_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_albedo")
        makedirs(render_albedo_path, exist_ok=True)
    if "render_irradiance"in args.__dict__.keys() and args.render_irradiance:
        render_irradiance_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_irradiance")
        makedirs(render_irradiance_path, exist_ok=True)
    if  "render_specular"in args.__dict__.keys() and args.render_specular:
        render_I_specular_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_I_specular")
        makedirs(render_I_specular_path, exist_ok=True)
        #render_specular_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_specular")
        #render_specular_int_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_specular_intensity")
        # makedirs(render_specular_path, exist_ok=True)
        # makedirs(render_specular_int_path, exist_ok=True)
    if "render_normal"in args.__dict__.keys() and args.render_normal:
        render_normal_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_normal")
        makedirs(render_normal_path, exist_ok=True)
    if "render_KS"in args.__dict__.keys() and args.render_KS:
        render_KS_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_KS")
        makedirs(render_KS_path, exist_ok=True)
    if "render_roughness"in args.__dict__.keys() and args.render_roughness:
        render_roughness_path = os.path.join(model_path, name, "ours_{}".format(epoch), "render_roughness")
        makedirs(render_roughness_path, exist_ok=True)
    if "render_reflectance"in args.__dict__.keys() and args.render_reflectance:
        render_reflectance_path = os.path.join(model_path, name, "ours_{}".format(epoch), "render_reflectance")
        makedirs(render_reflectance_path, exist_ok=True)
    if "render_depth"in args.__dict__.keys() and args.render_depth:
        render_depth_path = os.path.join(model_path, name, "ours_{}".format(epoch), "render_depth")
        makedirs(render_depth_path, exist_ok=True)
    if "render_depth_normal"in args.__dict__.keys() and args.render_depth_normal:
        render_depth_normal_path = os.path.join(model_path, name, "ours_{}".format(epoch), "render_depth_normal")
        makedirs(render_depth_normal_path, exist_ok=True)

    scene_name=gaussians.scene_name
    vedio_path = os.path.dirname(render_path)
    rendering_imgs = []
    for idx, cam_param in enumerate(tqdm(cam_params, desc="Rendering progress")):
        gt = cam_param.original_image[:3, :, :]
        gt_alpha = cam_param.gt_alpha_mask.to(gt.device)
        # gt=torch.cat([gt,gt_alpha],dim=0)

        rendering_results = render(cam_param, gaussians, pipeline, background)
        rendering=rendering_results["render"]
  
        rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        alpha=rendering_results["alpha"]
        
        if "render_albedo"in args.__dict__.keys() and args.render_albedo:
            rendering_albedo=rendering_results["albedo"]
            rendering_albedo=torch.cat([rendering_albedo,alpha],dim=0)
            torchvision.utils.save_image(rendering_albedo, os.path.join(render_albedo_path, '{0:05d}'.format(idx) + ".png"))
        if "render_normal"in args.__dict__.keys() and args.render_normal:
            rendering_normal=(rendering_results["normal"]+1)/2
            rendering_normal=torch.cat([rendering_normal,alpha],dim=0)
            torchvision.utils.save_image(rendering_normal, os.path.join(render_normal_path, '{0:05d}'.format(idx) + ".png"))
        if "render_irradiance"in args.__dict__.keys() and args.render_irradiance:
            rendering_irradiance=rendering_results["irradiance"]
            rendering_irradiance=torch.cat([rendering_irradiance,alpha],dim=0)
            torchvision.utils.save_image(rendering_irradiance, os.path.join(render_irradiance_path, '{0:05d}'.format(idx) + ".png"))
        if  "render_specular"in args.__dict__.keys() and args.render_specular:
            rendering_i_specular=rendering_results["I_specular"]
            rendering_i_specular=torch.cat([rendering_i_specular,alpha],dim=0)
            torchvision.utils.save_image(rendering_i_specular, os.path.join(render_I_specular_path, '{0:05d}'.format(idx) + ".png"))
            # rendering_specular=rendering_results["specular"]
            # rendering_specular_int=rendering_results["specular_intensity"]
            # rendering_specular=torch.cat([rendering_specular,alpha],dim=0)
            # rendering_specular_int=torch.cat([rendering_specular_int,alpha],dim=0)
            # torchvision.utils.save_image(rendering_specular, os.path.join(render_specular_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(rendering_specular_int, os.path.join(render_specular_int_path, '{0:05d}'.format(idx) + ".png"))
            
            
        if "render_KS"in args.__dict__.keys() and args.render_KS:
            rendering_KS=rendering_results["KS"]
            rendering_KS=torch.cat([rendering_KS.repeat(3,1,1),alpha],dim=0)
            #save_image_L(rendering_KS, os.path.join(render_KS_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(rendering_KS, os.path.join(render_KS_path, '{0:05d}'.format(idx) + ".png"))
        if "render_roughness"in args.__dict__.keys() and args.render_roughness:
            rendering_roughness=rendering_results["roughness"]
            rendering_roughness=torch.cat([rendering_roughness.repeat(3,1,1),alpha],dim=0)
            torchvision.utils.save_image(rendering_roughness, os.path.join(render_roughness_path, '{0:05d}'.format(idx) + ".png"))
        if "render_reflectance"in args.__dict__.keys() and args.render_reflectance:
            render_reflectance=rendering_results["fresnel_reflectance"]
            render_reflectance=torch.cat([render_reflectance.repeat(3,1,1),alpha],dim=0)
            torchvision.utils.save_image(render_reflectance, os.path.join(render_reflectance_path, '{0:05d}'.format(idx) + ".png"))
        if "render_depth"in args.__dict__.keys() and args.render_depth:
            render_depth=rendering_results["depth"]
            render_depth=(render_depth-render_depth.min())/(render_depth.max()-render_depth.min())
            render_depth=torch.cat([render_depth.repeat(3,1,1),alpha],dim=0)
            torchvision.utils.save_image(render_depth, os.path.join(render_depth_path, '{0:05d}'.format(idx) + ".png"))
        if "render_depth_normal"in args.__dict__.keys() and args.render_depth_normal:
            render_depth=rendering_results["depth"]
            intrinsic_matrix, extrinsic_matrix = cam_param.intrinsic_matrix,cam_param.extrinsic_matrix
            normal_refer = normal_from_depth_image(render_depth[0], intrinsic_matrix.to(device), extrinsic_matrix.to(device)).permute(2,0,1)
            normal_refer=(normal_refer+1)/2
            render_depth_normal=(normal_refer*(rendering_results["alpha"].detach()))
            render_depth_normal=torch.cat([render_depth_normal,alpha],dim=0)
            torchvision.utils.save_image(render_depth_normal, os.path.join(render_depth_normal_path, '{0:05d}'.format(idx) + ".png"))

            
    rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(vedio_path, f'{name}_{scene_name}_video.mp4'), rendering_imgs, fps=30, quality=8)

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _fps_safe(seconds_per_image):
    return (1.0 / seconds_per_image) if seconds_per_image > 0 else float("inf")


def _format_speed_summary(label, total_s, params_s):
    pure_s = total_s - params_s
    return (
        f"[{label}] total: {total_s*1000:.2f} ms/img ({_fps_safe(total_s):.1f} FPS) | "
        f"params_extract: {params_s*1000:.2f} ms ({_fps_safe(params_s):.1f} FPS) | "
        f"pure_render: {pure_s*1000:.2f} ms ({_fps_safe(pure_s):.1f} FPS)"
    )


def test_rendering_speed(model_path, cam_params, gaussians, pipeline, background,args):
    import copy,time,json
    render=gaussian_renderer.render_with_deferred

    savefolder=model_path
    speed_dict={}
    print("test rendering speed.......")
    cam_params=copy.deepcopy(cam_params)
    test_length=500
    test_cam_params=cam_params[:test_length]

    # Warm-up render to amortise lazy CUDA init / first-launch overhead, then
    # synchronise so its cost is not folded into the timed loop.
    rendering_results = render(cam_params[-1], gaussians, pipeline, background)
    _cuda_sync()

    # Single synchronise straddling the loop: we measure the wall time of the
    # whole 500-iter render, not per-iter (per-iter sync inflates the cost).
    # NOTE: gaussians.params_extract_time is recorded inside forward() without
    # a CUDA sync, so it reflects CPU dispatch time of the SMIRK encoder, not
    # its true GPU runtime. The authoritative number here is total_speed.
    _cuda_sync()
    start_time=time.time()
    params_extract_time=0
    for idx in tqdm(range(test_length)):
        rendering_results = render(test_cam_params[idx], gaussians, pipeline, background)
        params_extract_time+=gaussians.params_extract_time
    _cuda_sync()
    end_time=time.time()
    avg_total_speed=(end_time-start_time)/test_length
    avg_params_extract_speed=(params_extract_time)/test_length
    pure_render_speed=avg_total_speed-avg_params_extract_speed
    speed_dict["total_speed"]={"times(ms/image)":avg_total_speed*1000,"FPS(images/s)":_fps_safe(avg_total_speed)}
    speed_dict["params_extract_speed"]={"times(ms/image)":avg_params_extract_speed*1000,"FPS(images/s)":_fps_safe(avg_params_extract_speed)}
    speed_dict["rendering_speed"]={"times(ms/image)":pure_render_speed*1000,"FPS(images/s)":_fps_safe(pure_render_speed)}

    print(_format_speed_summary("default", avg_total_speed, avg_params_extract_speed))


    if len(args.envmap_path)!=0:
        print("test rendering speed for relighting.......")
        envpath=args.envmap_path[0]
        new_Envmap=EnvironmentMap_relight(envpath)
        rendering_results = render(test_cam_params[-1], gaussians, pipeline, background,other_envmap=new_Envmap)
        _cuda_sync()
        params_extract_time=0
        _cuda_sync()
        start_time=time.time()
        for idx in tqdm(range(test_length)):
            cam_param=test_cam_params[idx]
            rendering_results = render(cam_param, gaussians, pipeline, background,other_envmap=new_Envmap)
            rendering=rendering_results["render"]
            params_extract_time+=gaussians.params_extract_time
        _cuda_sync()
        end_time=time.time()
        avg_total_speed=(end_time-start_time)/test_length
        avg_params_extract_speed=(params_extract_time)/test_length
        pure_render_speed=avg_total_speed-avg_params_extract_speed

        speed_dict["religting_total_speed"]={"times(ms/image)":avg_total_speed*1000,"FPS(images/s)":_fps_safe(avg_total_speed)}
        speed_dict["religting_params_extract_speed"]={"times(ms/image)":avg_params_extract_speed*1000,"FPS(images/s)":_fps_safe(avg_params_extract_speed)}
        speed_dict["religting_rendering_speed"]={"times(ms/image)":pure_render_speed*1000,"FPS(images/s)":_fps_safe(pure_render_speed)}

        print(_format_speed_summary("relight", avg_total_speed, avg_params_extract_speed))

    with open(os.path.join(savefolder, "rendering_speed" + '.json'), 'w') as fp:
        json.dump(speed_dict, fp)
    print(speed_dict)

def render_envmap(model_path, epoch, gaussians):
    envmap_path = os.path.join(model_path, "train", "ours_{}".format(epoch))
    makedirs(envmap_path, exist_ok=True)

 
    actfun=gaussians.Envmap.activation
    diifuse_map=cubemap_to_latlong(gaussians.Envmap.diffuse,[64,64*2])
    specular_map=cubemap_to_latlong(gaussians.Envmap.specular,[64,64*2])
    diifuse_map,specular_map=actfun(diifuse_map),actfun(specular_map)
    torchvision.utils.save_image(diifuse_map.permute(2,0,1), os.path.join(envmap_path,  "diifuse_envmap.png"))
    torchvision.utils.save_image(specular_map.permute(2,0,1), os.path.join(envmap_path,  "specular_envmap.png"))
    
    diifuse_map=cubemap_to_latlong(gaussians.Envmap.diffuse,[64,64*2],150)
    specular_map=cubemap_to_latlong(gaussians.Envmap.specular,[64,64*2],150)
    diifuse_map,specular_map=actfun(diifuse_map),actfun(specular_map)
    diifuse_map,specular_map=torch.flip(diifuse_map,dims=[1]),torch.flip(specular_map,dims=[1])
    torchvision.utils.save_image(diifuse_map.permute(2,0,1), os.path.join(envmap_path,  "diifuse_envmap_shift.png"))
    torchvision.utils.save_image(specular_map.permute(2,0,1), os.path.join(envmap_path,  "specular_envmap_shift.png"))

def render_relighting(model_path, name, epoch, cam_params, gaussians, pipeline, background,args):
    import time
    print("rendering relighting...")
    render=gaussian_renderer.render_with_deferred
        

    if len(args.envmap_path)!=0:
        for envpath in args.envmap_path:

            new_Envmap=EnvironmentMap_relight(envpath)
            max_steps=min(len(cam_params),500)
            envmap_name=os.path.basename(envpath)
            render_path = os.path.join(model_path, "dynamic_relight",f"{name}_{epoch}", f"renders_relight-{envmap_name}")
            makedirs(render_path, exist_ok=True)
            vedio_path = os.path.dirname(render_path)
            scene_name=gaussians.scene_name
            rendering_imgs = []
            for idx, cam_param in enumerate(tqdm(cam_params, desc="Rendering progress")):
                rotate_y_degree=idx/max_steps*360
                new_Envmap.set_rotate_y(rotate_y_degree)
                rendering_results = render(cam_param, gaussians, pipeline, background,other_envmap=new_Envmap)
                rendering=rendering_results["render"]
                if args.with_relight_background:
                    back_map=new_Envmap.get_back_map(rotate_y_degree)
                    #alpha=rendering_results["alpha"]
                    alpha=cam_param.gt_alpha_mask.cuda(rendering.device)
                    rendering=rendering*alpha+(1-alpha)*back_map
                # rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
                rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
            imageio.mimwrite(os.path.join(vedio_path, f'{name}_{scene_name}_relight-{envmap_name}_video.mp4'), rendering_imgs, fps=30, quality=8)

def render_static_relighting(model_path, name,epoch, cam_params, gaussians, pipeline, background,args,frame_idxs=[]):
    import time
    print(f"rendering static relighting. for {name} set..")
    render=gaussian_renderer.render_with_deferred

    if len(args.envmap_path)!=0:
        for envpath in args.envmap_path:
            new_Envmap=EnvironmentMap_relight(envpath)
            max_steps=360#180
            envmap_name=os.path.basename(envpath)
            for frame_idx in tqdm(frame_idxs):
                cam_param=cam_params[frame_idx]
                render_path = os.path.join(model_path, "static_relight",f"{name}_{epoch}",  f"renders_relight_{envmap_name}_{frame_idx}")
                makedirs(render_path, exist_ok=True)
                vedio_path = os.path.dirname(render_path)
                scene_name=gaussians.scene_name
                rendering_imgs = []
                for idx in (tqdm(range(max_steps), desc="Rendering progress")):
                    rotate_y_degree=idx/max_steps*360
                    new_Envmap.set_rotate_y(rotate_y_degree)
                    rendering_results = render(cam_param, gaussians, pipeline, background,other_envmap=new_Envmap)
                    rendering=rendering_results["render"]
                    alpha=rendering_results["alpha"]
                    if args.with_relight_background:
                        back_map=new_Envmap.get_back_map(rotate_y_degree)
                        #alpha=rendering_results["alpha"]
                        alpha=cam_param.gt_alpha_mask.cuda(rendering.device)
                        rendering=rendering*alpha+(1-alpha)*back_map
                    else:
                        rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
                    rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
                    torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
                imageio.mimwrite(os.path.join(vedio_path, f'{name}_{scene_name}_relight_{envmap_name}_{frame_idx}_video.mp4'), rendering_imgs, fps=30, quality=8)

def render_material_editing(model_path, name, epoch, cam_params, gaussians, pipeline, background,args):
    import time
    render=gaussian_renderer.render_with_deferred
    
    if len(args.envmap_path)>0:
        new_Envmap=EnvironmentMap_relight(args.envmap_path[0])
    
    max_steps=len(cam_params)
    # envmap_name=os.path.basename(args.envmap_path[0])
    render_path = os.path.join(model_path, name, "ours_{}".format(epoch), f"renders_material_editing")
    makedirs(render_path, exist_ok=True)
    vedio_path = os.path.dirname(render_path)
    scene_name=gaussians.scene_name
    rendering_imgs = []
    
    for idx, cam_param in enumerate(tqdm(cam_params, desc="Rendering progress")):
        roughness_scale=(1-idx/max_steps)
        if  args.envmap_path is not None:
            new_Envmap.set_reflectance_scale(idx/max_steps)
            new_Envmap.set_roughness_scale(roughness_scale)
            new_Envmap.set_rotate_y(idx/max_steps*360)
            rendering_results = render(cam_param, gaussians, pipeline, background,other_envmap=new_Envmap)
            
        else:
            gaussians.Envmap.set_reflectance(idx/max_steps)
            gaussians.Envmap.set_reflectance_scale(idx/max_steps)
            gaussians.Envmap.set_roughness_scale(roughness_scale)
            rendering_results = render(cam_param, gaussians, pipeline, background)
            
        rendering=rendering_results["render"]

        #alpha=rendering_results["alpha"]
        #rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
        rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(vedio_path, f'{name}_{scene_name}_material_editing_video.mp4'), rendering_imgs, fps=30, quality=8)
    gaussians.Envmap.set_reflectance_scale(0.0)
    gaussians.Envmap.set_roughness_scale(1.0)

def render_static_material_editing(model_path, name, epoch, cam_params, gaussians, pipeline, background,args,frame_idxs=[]):
    import time
    render=gaussian_renderer.render_with_deferred
    
    if len(args.envmap_path)>0:
        new_Envmap=EnvironmentMap_relight(args.envmap_path[0])
        envmap_name=os.path.basename(args.envmap_path[0])

    max_steps=240#100
    
    for frame_idx in frame_idxs:
        render_path = os.path.join(model_path, "static_material_editing", f"{name}_{epoch}", f"material_editing_{frame_idx}")
        makedirs(render_path, exist_ok=True)
        vedio_path = os.path.dirname(render_path)
        scene_name=gaussians.scene_name
        rendering_imgs = []
        cam_param=cam_params[frame_idx]
        for idx in (tqdm(range(max_steps), desc="Rendering progress")):
            roughness_scale=(1-idx/max_steps)
            if  args.envmap_path is not None:
                new_Envmap.set_reflectance_scale(idx/max_steps)
                new_Envmap.set_roughness_scale(roughness_scale)
                #new_Envmap.set_rotate_y(idx/max_steps*360)
                rendering_results = render(cam_param, gaussians, pipeline, background,other_envmap=new_Envmap)
                
            else:
                gaussians.Envmap.set_reflectance(idx/max_steps)
                gaussians.Envmap.set_reflectance_scale(idx/max_steps)
                gaussians.Envmap.set_roughness_scale(roughness_scale)
                rendering_results = render(cam_param, gaussians, pipeline, background)
                
            rendering=rendering_results["render"]
            alpha=rendering_results["alpha"]
            if args.with_relight_background and len(args.envmap_path)>0:
                back_map=new_Envmap.get_back_map(0.0)
                alpha=rendering_results["alpha"]
                #alpha=cam_param.gt_alpha_mask.cuda(rendering.device)
                rendering=rendering*alpha+(1-alpha)*back_map
            # rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
            rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
        imageio.mimwrite(os.path.join(vedio_path, f'{name}_{scene_name}_{frame_idx}_static_material_editing.mp4'), rendering_imgs, fps=30, quality=8)
        gaussians.Envmap.set_reflectance_scale(0.0)
        gaussians.Envmap.set_roughness_scale(1.0)

def render_cross_reenactment(model_path,ori_params, epoch, gaussians, pipeline, background,args):
    
    render=gaussian_renderer.render_with_deferred
    
    for cross_idx,corss_path in enumerate(args.corss_source_paths):
        args_copy=deepcopy(args)
        args_copy.source_path=corss_path
        args_copy.test_set_ratio=0.0
        cross_id_name=os.path.basename(corss_path)
        print("Loading cross id data...")
        Cross_head_data=TrackedData(args_copy.source_path,args_copy,split="train",load_image=True)
        
        scene_name=gaussians.scene_name
        render_path = os.path.join(model_path, "test_cross_reenactment", "ours_{}".format(epoch),f"{cross_id_name}_reenactment_{scene_name}")
        makedirs(render_path, exist_ok=True)
        # video_path = os.path.join(model_path, "test", "ours_{}".format(epoch))
        video_path = os.path.dirname(render_path)
        makedirs(render_path, exist_ok=True)
        rendering_imgs = []
        cam_params=Cross_head_data
        ori_param=deepcopy(ori_params[0])


        for idx, cam_param in enumerate(tqdm(cam_params, desc="Rendering progress")):
            rendering_results = render(cam_param, gaussians, pipeline, background)
            rendering=rendering_results["render"]
            if "alpha" in rendering_results.keys():
                alpha=rendering_results["alpha"]
            else:
                alpha=cam_param.gt_alpha_mask
            rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
            rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
        imageio.mimwrite(os.path.join(video_path, f'{cross_id_name}_reenactment_{scene_name}_video.mp4'), rendering_imgs, fps=30, quality=8)
        
def render_multi_views(model_path, name, epoch, cam_params, gaussians, pipeline, background,multi_cam_params=None):
    if multi_cam_params is None:
        multi_cam_params=generate_multi_view_poses(cam_params)
    render=gaussian_renderer.render_with_deferred

        
    scene_name=gaussians.scene_name
    render_path = os.path.join(model_path, f"{name}", "ours_{}".format(epoch),f'{name}_multi-views_{scene_name}')
    makedirs(render_path, exist_ok=True)
    #video_path = os.path.join(model_path, name, "ours_{}".format(epoch))
    video_path = os.path.dirname(render_path)
    makedirs(video_path, exist_ok=True)
    rendering_imgs = []
    for idx, cam_param in enumerate(tqdm(cam_params, desc="Rendering progress")):
        other_cam_parm=multi_cam_params[idx]
        rendering_results=render(cam_param, gaussians, pipeline, background,other_cam_param=other_cam_parm)
        rendering = rendering_results["render"]
        if "alpha" in rendering_results.keys():
            alpha=rendering_results["alpha"]
        else:
            alpha=cam_param.gt_alpha_mask.to(rendering.device)
        rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
        rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(video_path, f'{name}_multi-views_{scene_name}_video.mp4'), rendering_imgs, fps=30, quality=8)

def render_static_multi_views(model_path, name, epoch, cam_params, gaussians, pipeline, background,frame_idxs=[]):
    render=gaussian_renderer.render
    render=gaussian_renderer.render_with_deferred
    for frame_idx in frame_idxs:
        num_keyframes=360#180
        cam_param_frame=[deepcopy(cam_params[frame_idx]) for i in range(num_keyframes)]
        multi_cam_params=generate_multi_view_poses(cam_param_frame,num_keyframes=num_keyframes,y_offset=-0.0)
   
        scene_name=gaussians.scene_name
        render_path = os.path.join(model_path, f"{name}_static_multiviews", "ours_{}".format(epoch),f"{scene_name}_{frame_idx:05d}",)
        makedirs(render_path, exist_ok=True)
        #video_path = os.path.join(model_path, name, "ours_{}".format(epoch))
        video_path = os.path.dirname(render_path)
        makedirs(video_path, exist_ok=True)
        rendering_imgs = []
        for idx, cam_param in enumerate(tqdm(multi_cam_params, desc="Rendering progress")):
            # other_cam_parm=multi_cam_params[idx] ,other_cam_param=other_cam_parm
            rendering_results=render(cam_param, gaussians, pipeline, background)
            rendering = rendering_results["render"]
            if "alpha" in rendering_results.keys():
                alpha=rendering_results["alpha"]
            else:
                alpha=cam_param.gt_alpha_mask.to(rendering.device)
            rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
            rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
        imageio.mimwrite(os.path.join(video_path, f'{name}_static_multiviews_{scene_name}_{frame_idx:05d}_video.mp4'), rendering_imgs, fps=30, quality=8)

def extract_optimized_params( cam_params_train,cam_params_test, gaussians,data_path):
    import json
    
    all_cam_params=cam_params_train+cam_params_test
    tracked_params_path=os.path.join(data_path,"tracked_params.json")
    with open(tracked_params_path) as json_file:
        tracked_params_dict = json.load(json_file)
    image_keys=tracked_params_dict.keys()
    image_keys=natsorted(image_keys)
    print(f"Extracting {os.path.basename(data_path)} optimized params...")
    for idx,cam_param in enumerate(tqdm(all_cam_params)):
        image_key=image_keys[idx]
        assert idx==int(image_key.split(".")[0])
        if  gaussians.with_param_net_smirk and cam_param.warped_image is not None:
            warped_image=cam_param.warped_image.cuda(gaussians.device)
            out_params=gaussians.flame_params_net(warped_image)
            expression_param = out_params['expression_params']
            jaw_params = out_params.get('jaw_params', None)
            eyelid_param = out_params.get('eyelid_params', None)
            #[(global)3, neck (0)3, (jaw)3, eyepose (0)6]
            full_pose_param=cam_param.full_pose_code.cuda(gaussians.device)
            if gaussians.use_smirk_jaw_pose:
                full_pose_param[:,6:9]=jaw_params
            tracked_params_dict[image_key]["fullposecode"]=full_pose_param.detach().cpu().numpy().tolist()
            tracked_params_dict[image_key]["eyelids"]=eyelid_param.detach().cpu().numpy().tolist()
            tracked_params_dict[image_key]["expcode"]=expression_param.detach().cpu().numpy().tolist()
    with open(os.path.join(data_path, "tracked_params_optimized" + '.json'), 'w') as fjson:
            json.dump(tracked_params_dict,fjson)
    fjson.close()
    print("Done!")

def _maybe_lhg_override(args):
    """Build the in-memory ``tracked_params`` override from
    ``--lhg-features`` (LHG render demo) or return ``(None, None)``.

    The override completely replaces per-frame entries with values
    from ``lhg_features.npz`` while keeping ``shapecode`` from the
    base ``tracked_params.json`` at ``--source_path``.
    """
    npz_path = getattr(args, 'lhg_features', None)
    if not npz_path:
        return None, None
    import json as _json
    from lhg.output import LHGFeatures
    from lhg.render_adapter import features_to_tracked_params

    base_path = os.path.join(args.source_path, 'tracked_params.json')
    flame_scale = 4.0
    if not os.path.exists(base_path):
        alt = os.path.join(args.source_path, 'tracked_params_v2.json')
        if os.path.exists(alt):
            base_path = alt
            flame_scale = 1.0
        else:
            base_path = None
    base_dict = None
    if base_path is not None:
        with open(base_path) as fp:
            base_dict = _json.load(fp)
    features = LHGFeatures.read(npz_path)
    flame_scale = float(features.flame_scale)
    override = features_to_tracked_params(features, base_tracked_params=base_dict)
    print(f'[render] LHG override active: {npz_path} '
          f'({features.frame_basenames.shape[0]} frames, '
          f'flame_scale={flame_scale})')
    return override, flame_scale


def render_sets(dataset_args : ModelParams, epoch : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,all_args):
    with torch.no_grad():
        gaussians = GaussianHeadModel(dataset_args.sh_degree,all_args)
        lhg_override, lhg_flame_scale = _maybe_lhg_override(all_args)
        _train_dataset=TrackedData(
            args.source_path,args,split='train',pre_load=True,
            tracked_params_override=lhg_override,
            flame_scale_override=lhg_flame_scale,
        )
        _test_dataset=TrackedData(
            args.source_path,args,split='test',pre_load=True,
            tracked_params_override=lhg_override,
            flame_scale_override=lhg_flame_scale,
        )
        train_dataset=[_data for _data in _train_dataset]
        test_dataset=[_data for _data in _test_dataset]
        
        all_args.with_intrinsic_supervise=False
        
        scene = Head_Scene(all_args, gaussians, load_epoch=epoch,dataset=_train_dataset)
        gaussians.set_eval(True)
        bg_color= 1 if dataset_args.white_background else 0
        background = torch.tensor([bg_color]*3, dtype=torch.float32, device="cuda")


        test_rendering_speed(dataset_args.model_path, train_dataset, gaussians, pipeline, background,all_args)

        # Standalone speed-measurement mode: skip every other render path so
        # the user only pays for the timing run + JSON output.
        if getattr(all_args, "test_rendering_speed", False):
            gaussians.set_eval(False)
            return

        # Install SMIRK output smoother once; reset between independent sequences.
        smirk_smoother = _maybe_attach_smirk_smoother(gaussians, all_args)

        if not skip_train:
             _reset_smirk_smoother(gaussians, smirk_smoother)
             render_set(dataset_args.model_path, "train", scene.loaded_epoch, train_dataset, gaussians, pipeline, background,all_args)
        if not skip_test:
             _reset_smirk_smoother(gaussians, smirk_smoother)
             render_set(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args)

        if all_args.render_multi_views:
            _reset_smirk_smoother(gaussians, smirk_smoother)
            render_multi_views(dataset_args.model_path, "train", scene.loaded_epoch, train_dataset, gaussians, pipeline, background)
            _reset_smirk_smoother(gaussians, smirk_smoother)
            render_multi_views(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background)

        
        if all_args.render_envmap:
            render_envmap(dataset_args.model_path,scene.loaded_epoch,gaussians)
        if all_args.render_relighting:
            render_relighting(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args)
            
        if all_args.render_material_editing:
            render_material_editing(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args)

        if len(all_args.test_static_multiviews_idxs) >0:
            render_static_multi_views(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args.test_static_multiviews_idxs)
        if len(all_args.train_static_multiviews_idxs) >0:
            render_static_multi_views(dataset_args.model_path, "train", scene.loaded_epoch, train_dataset, gaussians, pipeline, background,all_args.train_static_multiviews_idxs)
        if len(all_args.corss_source_paths)!=0:
            _reset_smirk_smoother(gaussians, smirk_smoother)
            render_cross_reenactment(dataset_args.model_path,test_dataset, scene.loaded_epoch, gaussians, pipeline, background,all_args)
        if len(all_args.test_static_relight_idxs)>0:
            render_static_relighting(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args,all_args.test_static_relight_idxs)
        if len(all_args.train_static_relight_idxs)>0:
            render_static_relighting(dataset_args.model_path, "train", scene.loaded_epoch, train_dataset, gaussians, pipeline, background,all_args,all_args.train_static_relight_idxs)
        if len(all_args.test_static_material_edting_idxs)>0:
            render_static_material_editing(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args,all_args.test_static_material_edting_idxs)
        if len(all_args.train_static_material_edting_idxs)>0:
            render_static_material_editing(dataset_args.model_path, "train", scene.loaded_epoch, train_dataset, gaussians, pipeline, background,all_args,all_args.train_static_material_edting_idxs)

        if all_args.source_params_path!="":
            extract_optimized_params(train_dataset,test_dataset, gaussians,all_args.source_params_path)

        # Done with sequential renders — drop the smoother reference.
        _detach_smirk_smoother(gaussians)

    gaussians.set_eval(False)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_multi_views", action="store_true")
    parser.add_argument("--corss_source_paths", nargs="+",type=str,default=[],
                        help="Data Path of cross ID for reenactment")
    parser.add_argument("--param_net_ckpt_paths", nargs="+",type=str,default=[],
                        )
    parser.add_argument("--quiet", action="store_true") 
    
    parser.add_argument("--render_albedo", action="store_true",default=False)
    parser.add_argument("--render_irradiance", action="store_true",default=False)
    parser.add_argument("--render_specular", action="store_true",default=False)
    parser.add_argument("--render_normal", action="store_true",default=False)
    parser.add_argument("--render_KS", action="store_true",default=False)
    
    parser.add_argument("--render_roughness", action="store_true",default=False)
    parser.add_argument("--render_reflectance", action="store_true",default=False)
    parser.add_argument("--render_depth", action="store_true",default=False)
    parser.add_argument("--render_depth_normal", action="store_true",default=False)
    parser.add_argument("--render_envmap", action="store_true",default=False)
    parser.add_argument("--render_relighting", action="store_true",default=False)
    parser.add_argument("--render_material_editing", action="store_true",default=False)
    
    parser.add_argument("--envmap_path",nargs="+", type=str,default=[])
    parser.add_argument("--envmap_id", nargs="+",type=int,default=[0])

    parser.add_argument("--test_static_multiviews_idxs", nargs="+",type=int,default=[])
    parser.add_argument("--train_static_multiviews_idxs", nargs="+",type=int,default=[])
    parser.add_argument("--test_static_relight_idxs", nargs="+",type=int,default=[])
    parser.add_argument("--train_static_relight_idxs", nargs="+",type=int,default=[]) 
    parser.add_argument("--test_static_material_edting_idxs", nargs="+",type=int,default=[])
    parser.add_argument("--train_static_material_edting_idxs", nargs="+",type=int,default=[])
    parser.add_argument("--source_params_path",type=str,default="")
    parser.add_argument("--with_relight_background",action="store_true",default=False)
    parser.add_argument("--test_rendering_speed", action="store_true", default=False,
                        help="Run the rendering-speed measurement only, then exit. "
                             "Writes <model_path>/rendering_speed.json and prints a one-line summary.")
    parser.add_argument("--lhg-features", "--lhg_features", dest="lhg_features",
                        type=str, default=None,
                        help="Path to a lhg_features.npz produced by "
                             "lhg/extract.py (Stage 2). When set, the per-frame "
                             "tracker payload of TrackedData is replaced by "
                             "values from this file (see lhg/render_adapter.py). "
                             "shapecode is still taken from --source_path's "
                             "tracked_params.json so the avatar's baked shape "
                             "stays consistent. Used by demos/render_lhg_features.sh.")
    parser=add_more_argument(parser)
    args = get_combined_args(parser,model)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)
    
    render_sets(model.extract(args), args.epoch, pipeline.extract(args), args.skip_train, args.skip_test,args)