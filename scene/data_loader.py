

import os
import json
import torch
import numpy as np
import torchvision
from copy import deepcopy
from natsort import natsorted
from utils.graphics_utils import getWorld2View2, getProjectionMatrix,fov2focal,focal2fov
from utils.general_utils import run_mediapipe, crop_face
from utils.onefilter import smooth_sequence
from typing import NamedTuple
from glob import glob
from PIL import Image
from tqdm import tqdm
from skimage.transform import warp, SimilarityTransform


class Camera_params():
    original_image: torch.tensor
    gt_alpha_mask: torch.tensor
    albedo: torch.tensor
    specular: torch.tensor
    warped_image: torch.tensor
    
    R:torch.tensor
    T: torch.tensor
    world_view_transform: torch.tensor
    projection_matrix: torch.tensor
    full_proj_transform: torch.tensor
    camera_center: torch.tensor
    intrinsic_matrix: torch.tensor
    K: torch.tensor
    extrinsic_matrix: torch.tensor
    
    shape_code: torch.tensor
    translation_code: torch.tensor
    eyelid_code: torch.tensor
    full_pose_code: torch.tensor
    exp_code: torch.tensor
    
    FoVx : float
    FoVy: float
    image_width: int
    image_height: int
    image_name: str
    
    def __init__(self,original_image,gt_alpha_mask,albedo,specular,warped_image,world_view_transform,projection_matrix,full_proj_transform,
                 camera_center,intrinsic_matrix,K,R,T,extrinsic_matrix,shapecode,translation_code,eyelid_code,fullposecode,expcode,
                 FoVx,FoVy,image_width,image_height,image_name):
        self.original_image=original_image
        self.gt_alpha_mask=gt_alpha_mask
        self.albedo=albedo
        self.specular=specular
        self.warped_image=warped_image
        self.world_view_transform=world_view_transform
        self.projection_matrix=projection_matrix
        self.full_proj_transform=full_proj_transform
        self.camera_center=camera_center
        self.intrinsic_matrix=intrinsic_matrix
        self.K=K
        self.R=R
        self.T=T
        self.extrinsic_matrix=extrinsic_matrix
        self.shape_code=shapecode
        self.translation_code=translation_code
        self.eyelid_code=eyelid_code
        self.full_pose_code=fullposecode
        self.exp_code=expcode
        self.FoVx=FoVx
        self.FoVy=FoVy
        self.image_width=image_width
        self.image_height=image_height
        self.image_name=image_name
    
class TrackedData(torch.utils.data.Dataset):
    def __init__(self, path,args,split,pre_load=True,load_image=True,device='cpu'):
        self.args=args
        self.pre_load=pre_load
        self.load_image=load_image
        self.device=device
        
        
        if os.path.isdir(path):
            images_path = os.path.join(path, "image")
            mask_prepath = os.path.join(path, "mask")
            if not os.path.exists(images_path):
                images_path = os.path.join(path, "images")
            imagepath_list = glob(images_path + '/*.jpg') + glob(images_path + '/*.png') + glob(images_path + '/*.bmp')
            print('total {} images'.format(len(imagepath_list)))
            imagepath_list = natsorted(imagepath_list)

        tracked_params_path = os.path.join(path, "tracked_params.json")
        self.flame_scale = 4.0
        if not os.path.exists(tracked_params_path):
            tracked_params_path = os.path.join(path, "tracked_params_v2.json")
            self.flame_scale = 1.0
        with open(tracked_params_path) as json_file:
            tracked_params_dict = json.load(json_file)

        train_set_len = int(len(imagepath_list) * (1 - args.test_set_ratio))
        if args.test_set_num != -1:
            train_set_len = int(len(imagepath_list) - args.test_set_num)
        if split == 'train':
            imagepath_list = imagepath_list[:train_set_len]
        else:
            imagepath_list = imagepath_list[train_set_len:]
        self.imagepath_list=imagepath_list
        self.data_len=len(imagepath_list)
        
        # Per-frame SMIRK input crop. Two paths:
        #   (1) `stable_bbox.npz` exists at the dataset root → load the
        #       precomputed (basename → tform) mapping. Skip MediaPipe at
        #       training time entirely; same crop as DECA preprocessing saw.
        #   (2) Otherwise → fall back to legacy per-iter MediaPipe + crop_face.
        # Path (1) eliminates the mouth/blink leak into the SMIRK encoder
        # input that is the dominant source of expression-channel jitter.
        self._stable_bbox_tforms: dict[str, np.ndarray] | None = None
        if args.with_param_net_smirk:
            stable_bbox_path = os.path.join(path, 'stable_bbox.npz')
            if os.path.exists(stable_bbox_path):
                npz = np.load(stable_bbox_path, allow_pickle=False)
                names = [str(b) for b in npz['frame_basenames']]
                tforms = npz['tform']
                self._stable_bbox_tforms = {
                    name: tforms[i] for i, name in enumerate(names)
                }
                print(f'[data_loader] using precomputed stable bbox '
                      f'({len(self._stable_bbox_tforms)} frames) from '
                      f'{stable_bbox_path}')
            else:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                base_options = python.BaseOptions(model_asset_path='./assets/smirk/face_landmarker.task')
                options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                    output_face_blendshapes=True,
                                                    output_facial_transformation_matrixes=True,
                                                    num_faces=1,
                                                    min_face_detection_confidence=0.1,
                                                    min_face_presence_confidence=0.1)
                self.detector = vision.FaceLandmarker.create_from_options(options)
                print('[data_loader] stable_bbox.npz not found — falling back '
                      'to per-iteration MediaPipe + crop_face. Run '
                      'preprocess/stable_bbox.py to remove SMIRK input jitter.')

        # assume same resolution for all images
        image = Image.open(imagepath_list[0])
        self.imagew, self.imageh = image.size[0], image.size[1]

        self.bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device) if args.white_background else torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        shapecode = torch.tensor(tracked_params_dict["shapecode"], device=self.device)[:, :args.n_shape]

        # Optional One-Euro temporal smoothing of per-frame tracker params.
        # Inspired by MTamon/Gaussian-HS. Off by default; enabled when
        # `--jitter_filter` is set. The smoothed values replace the raw
        # per-frame entries below.
        self._smoothed: dict[str, dict[str, np.ndarray]] = {}
        if getattr(args, "jitter_filter", False):
            self._smoothed = self._build_smoothed_params(
                tracked_params_dict, imagepath_list, args)
        self.cam_params_list=[]
        for idx, imagepath in tqdm(enumerate(imagepath_list)):
            imagename = imagepath.split('/')[-1].split('.')[0]
            image_basename = os.path.basename(imagepath)
            imagekey = imagename if imagename in tracked_params_dict.keys() else image_basename
            
            if self.pre_load and self.load_image:
                image, mask_data, albedo, specular, warped_image = self._load_images(imagepath, args, self.bg)
            else:
                image, mask_data, albedo, specular, warped_image = None, None, None, None, None
            
            
            if "translation" in tracked_params_dict[imagekey].keys():
                translation_code = self._maybe_smoothed(
                    "translation", imagekey,
                    tracked_params_dict[imagekey]["translation"])
                translation_code = torch.tensor(translation_code, dtype=torch.float32, device=self.device)
            else:
                translation_code = None
            if "eyelids" in tracked_params_dict[imagekey].keys():
                eyelid_code = self._maybe_smoothed(
                    "eyelid", imagekey,
                    tracked_params_dict[imagekey]["eyelids"])
                eyelid_code = torch.tensor(eyelid_code, dtype=torch.float32, device=self.device)
            else:
                eyelid_code = None

            if  "intrinsics"in tracked_params_dict.keys():
                intrinsics=tracked_params_dict["intrinsics"]#[fx, fy, cx, cy]
                fovx=2*np.arctan2(intrinsics[2],intrinsics[0])


            fullposecode = self._maybe_smoothed(
                "fullpose", imagekey,
                tracked_params_dict[imagekey]["fullposecode"])
            fullposecode = torch.tensor(fullposecode, dtype=torch.float32, device=self.device)
            expcode = self._maybe_smoothed(
                "expression", imagekey,
                tracked_params_dict[imagekey]["expcode"])
            expcode = torch.tensor(expcode, dtype=torch.float32, device=self.device)[:, :args.n_expr]
            
            fovy = focal2fov(fov2focal(fovx, self.imagew), self.imageh)
            world_view_transform,projection_matrix,full_proj_transform,camera_center,extrinsic_matrix,intrinsic_matrix,R,T = \
                self._load_camera(tracked_params_dict,imagekey,fovx,fovy,zfar=100.0,znear =0.01)

            self.cam_params_list.append(Camera_params(original_image=image,gt_alpha_mask=mask_data,albedo=albedo,
                                                        specular=specular,warped_image=warped_image,world_view_transform=world_view_transform,
                                                        projection_matrix=projection_matrix,full_proj_transform=full_proj_transform,
                                                        camera_center=camera_center,intrinsic_matrix=intrinsic_matrix,K=intrinsic_matrix,
                                                        R=R,T=T,extrinsic_matrix=extrinsic_matrix, shapecode=shapecode,translation_code=translation_code,
                                                        eyelid_code=eyelid_code,fullposecode=fullposecode,expcode=expcode,
                                                        FoVx =fovx,FoVy=fovy,image_width=self.imagew,image_height=self.imageh,
                                                        image_name=imagename))
            
    def __getitem__(self, index):
        
        cam_param = self.cam_params_list[index]
        if not self.pre_load:
            cam_param.original_image,cam_param.gt_alpha_mask,cam_param.albedo,cam_param.specular,cam_param.warped_image = \
                self._load_images(self.imagepath_list[index], self.args, self.bg)
        
        return cam_param
    
    def __len__(self, ):
        return len(self.imagepath_list)
    
            
    def _load_images(self,imagepath,args,bg):
        bg=bg.numpy()
        image = Image.open(imagepath)
        pre_path = os.path.dirname(os.path.dirname(imagepath))
        image_basename=os.path.basename(imagepath)
        mask_prepath = os.path.join(pre_path, "mask")
        
        imagew,imageh=image.size[0],image.size[1]
        image = np.array(image,dtype=np.float32)/255.0
        if image.shape[2]==4:
            mask_data=image[:,:,3:4]
            image=image[:,:,:3]
        elif os.path.exists(mask_prepath):
            maskpath=os.path.join(mask_prepath,image_basename)
            mask=Image.open(maskpath)
            mask_data=(np.array(mask,dtype=np.float32)/255.0).mean(axis=2,keepdims=True)
        else:
            mask_data=np.ones((imageh,imagew,1))
            
        image = image*mask_data+(1-mask_data)*bg
        specular,albedo=None,None
        global landmark
        if args.with_intrinsic_supervise:
            albedo_path=os.path.join(pre_path,"albedo",image_basename)
            
            if os.path.exists(albedo_path):
                albedo=Image.open(albedo_path)
                albedo=np.array(albedo,dtype=np.float32)/255.0
                albedo=albedo*mask_data+(1-mask_data)*bg
                albedo=torch.tensor(albedo, device=self.device,dtype=torch.float32).permute(2,0,1)
                
            # specular_path=os.path.join(pre_path,"specular",image_basename)
            # if os.path.exists(specular_path):
            #     specular=Image.open(specular_path)
            #     specular=(np.array(specular,dtype=np.float32)/255.0).mean(axis=2,keepdims=True)
            #     specular=specular*mask_data+(1-mask_data)*bg
            #     specular=torch.tensor(specular, device=self.device,dtype=torch.float32)
            
        warped_image=None
        crop_size=[224,224]
        if args.with_param_net_smirk:
            if self._stable_bbox_tforms is not None:
                # Precomputed path: look up the stabilized similarity transform
                # by source-image basename and warp directly. No MediaPipe at
                # training time — same (center, size) DECA preprocessing saw.
                tform_params = self._stable_bbox_tforms.get(image_basename)
                if tform_params is None:
                    raise KeyError(
                        f'stable_bbox.npz has no entry for {image_basename}; '
                        f'regenerate it with preprocess/stable_bbox.py after '
                        f'changing image/.')
                tform = SimilarityTransform(matrix=tform_params)
            else:
                kpt_mediapipe = run_mediapipe((image*255.0).astype(np.uint8),self.detector)
                if (kpt_mediapipe is None):
                    print('Could not find landmarks for the image using last landmark.')
                    kpt_mediapipe=landmark
                else:
                    kpt_mediapipe = kpt_mediapipe[..., :2]
                    landmark=kpt_mediapipe
                tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=224)
            warped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True)
            warped_image=torch.tensor(warped_image, device=self.device,dtype=torch.float32)
            warped_image=warped_image.permute(2, 0, 1)[None]
            # warped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
            
        image=torch.tensor(image, device=self.device,dtype=torch.float32).permute(2, 0, 1)
        mask_data=torch.tensor(mask_data, device=self.device,dtype=torch.float32).permute(2, 0, 1)
        #adhere to Guassian splatting data reader
        image=(((image*255).to(torch.uint8))/255.0).to(torch.float32)
        mask_data=(((mask_data*255).to(torch.uint8))/255.0).to(torch.float32)
        albedo=(((albedo*255).to(torch.uint8))/255.0).to(torch.float32) if albedo is not None else None
        warped_image=(((warped_image*255).to(torch.uint8))/255.0).to(torch.float32) if warped_image is not None else None
    
        return image,mask_data,albedo,specular,warped_image
    
    def _load_camera(self,tracked_params_dict,imagekey,fovx,fovy,zfar=100.0,znear =0.01):

        world_mat_raw = self._maybe_smoothed(
            "world_mat", imagekey, tracked_params_dict[imagekey]["world_mat"])
        w2c=np.array([[1 ,0 ,0 ,0 ],
                    [0 ,-1,0 ,0 ],
                    [0 ,0 ,-1,0 ],
                    [0 ,0 ,0 ,1 ]],dtype=np.float32)@np.array(world_mat_raw,dtype=np.float32)
        R=np.transpose(w2c[:3,:3])
        T=w2c[:3, 3]
        fo=fov2focal(fovx, self.imagew)
        K= np.array([
            [fo, 0, self.imagew/2],
            [0, fo, self.imageh /2],
            [0, 0, 1]
        ],dtype=np.float32)
        
        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(self.device)
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).to(self.device)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        extrinsic_matrix=world_view_transform.transpose(0,1).cuda().contiguous()# cam2world
        intrinsic_matrix=torch.tensor(K,dtype=torch.float32,device=self.device)
        R=torch.tensor(R,dtype=torch.float32,device=self.device)
        T=torch.tensor(T,dtype=torch.float32,device=self.device)
        return world_view_transform,projection_matrix,full_proj_transform,camera_center,extrinsic_matrix,intrinsic_matrix,R,T

    # ---- One-Euro tracker smoothing (opt-in via --jitter_filter) -----------

    @staticmethod
    def _resolve_imagekey(imagepath, tracked_params_dict):
        imagename = imagepath.split('/')[-1].split('.')[0]
        image_basename = os.path.basename(imagepath)
        return imagename if imagename in tracked_params_dict else image_basename

    def _build_smoothed_params(self, tracked_params_dict, imagepath_list, args):
        """Pre-compute zero-phase One-Euro smoothed series per channel.

        Returns ``{channel: {imagekey: np.ndarray}}``. Channels not requested
        in ``args.jitter_filter_targets`` are not populated.
        """
        targets = set(getattr(args, "jitter_filter_targets",
                              ["translation", "fullpose", "expression", "eyelid"]))
        fps = float(getattr(args, "jitter_filter_fps", 30.0))
        min_cutoff = float(getattr(args, "jitter_filter_min_cutoff", 1.0))
        beta = float(getattr(args, "jitter_filter_beta", 0.0))
        d_cutoff = float(getattr(args, "jitter_filter_d_cutoff", 1.0))

        # (key in tracked_params_dict[imagekey], target_name)
        channel_specs = [
            ("translation", "translation"),
            ("fullposecode", "fullpose"),
            ("expcode", "expression"),
            ("eyelids", "eyelid"),
            ("world_mat", "world_mat"),
        ]

        smoothed: dict[str, dict[str, np.ndarray]] = {}
        keys_in_order: list[str] = [
            self._resolve_imagekey(p, tracked_params_dict) for p in imagepath_list
        ]

        n_smoothed = 0
        for src_key, target_name in channel_specs:
            if target_name not in targets:
                continue
            sample = tracked_params_dict.get(keys_in_order[0], {})
            if src_key not in sample:
                continue
            try:
                seq = np.asarray(
                    [tracked_params_dict[k][src_key] for k in keys_in_order],
                    dtype=np.float64,
                )
            except (KeyError, ValueError) as exc:
                print(f"[data_loader] jitter_filter: cannot stack '{src_key}' "
                      f"({exc}); skipping this channel.")
                continue
            if seq.shape[0] < 2:
                continue
            smoothed_seq = smooth_sequence(
                seq, fps=fps, min_cutoff=min_cutoff, beta=beta,
                d_cutoff=d_cutoff, bidirectional=True,
            )
            smoothed[target_name] = {
                k: smoothed_seq[i] for i, k in enumerate(keys_in_order)
            }
            n_smoothed += 1

        if n_smoothed:
            print(f"[data_loader] jitter_filter: smoothed {n_smoothed} channel(s) "
                  f"({sorted(smoothed.keys())}) "
                  f"with min_cutoff={min_cutoff} Hz, beta={beta}, fps={fps}")
        else:
            print("[data_loader] jitter_filter: enabled but no channels matched; "
                  "verify --jitter_filter_targets and tracked_params.json contents.")
        return smoothed

    def _maybe_smoothed(self, target_name, imagekey, raw_value):
        """Return smoothed array if available, else the raw value unchanged."""
        if not self._smoothed:
            return raw_value
        per_key = self._smoothed.get(target_name)
        if per_key is None:
            return raw_value
        out = per_key.get(imagekey)
        if out is None:
            return raw_value
        # Preserve original shape (e.g. (1, D) for fullpose / exp / translation,
        # (D,) for eyelids, (3, 4) for world_mat).
        raw_arr = np.asarray(raw_value)
        if out.shape != raw_arr.shape:
            try:
                out = out.reshape(raw_arr.shape)
            except ValueError:
                return raw_value
        return out

