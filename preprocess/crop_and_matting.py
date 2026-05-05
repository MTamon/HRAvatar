import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from abc import ABC
from glob import glob
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import face_alignment
import numpy as np
import torch, shutil
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.general_utils import natural_sort_key


def _decode_bytes(b):
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        return b.decode('latin1', errors='replace')


def _print_subprocess_output(stdout, stderr, tag, returncode):
    """Print both stdout and stderr from a preprocessing subprocess.
    Errors from RobustVideoMatting / face-parsing / etc. were previously
    swallowed, producing confusing `assert len(images)==len(masks)` crashes
    downstream.  Always show them."""
    if stdout:
        print(f"[{tag} stdout]\n{_decode_bytes(stdout)}")
    if stderr:
        print(f"[{tag} stderr]\n{_decode_bytes(stderr)}")
    print(f"[{tag}] returncode={returncode}")


def crop_image(image, x_min, y_min, x_max, y_max):
    """Crop ``image`` to ``[y_min:y_max, x_min:x_max]`` with zero-padding for
    any portion of the bbox that falls outside the image.

    Plan B (video-wide fixed outer bbox) sizes the bbox as
    ``bbox_scale * 2 * half_extent`` and expects the result to be exactly
    that square. The legacy clip-to-image-bounds behaviour collapsed the
    bbox onto the short edge whenever the head's motion envelope exceeded
    it, which made ``bbox_scale`` non-linear (it hit the bounds before the
    requested margin took effect). Padding keeps ``bbox_scale`` linear so
    operators can dial face-fraction-of-canvas directly.
    """
    h, w = image.shape[:2]
    pad_l = max(-x_min, 0)
    pad_r = max(x_max - w, 0)
    pad_t = max(-y_min, 0)
    pad_b = max(y_max - h, 0)
    if pad_l or pad_r or pad_t or pad_b:
        image = np.pad(image, [(pad_t, pad_b), (pad_l, pad_r), (0, 0)],
                       mode='constant')
        x_min += pad_l
        x_max += pad_l
        y_min += pad_t
        y_max += pad_t
    return image[y_min:y_max, x_min:x_max, :]


def squarefiy(image, size=512):
    h, w, c = image.shape
    if w != h:
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        image = np.pad(image, [(vp, vp), (hp, hp), (0, 0)], mode='constant')

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def crop_image_bbox(image, bbox):
    xb_min = bbox[0]
    xb_max = bbox[1]
    yb_min = bbox[2]
    yb_max = bbox[3]
    cropped = crop_image(image, xb_min, yb_min, xb_max, yb_max)
    return cropped

class Crop_and_matting(Dataset, ABC):
    def __init__(self, source, args):
        self.device = 'cuda:0'
        self.config = args
        self.source_dir=source
        self.name = args.name
        self.source = Path(source, args.name)
        os.makedirs(self.source,exist_ok=True)
        self.initialize()
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)
    
    def initialize(self):
        self.image_path = Path(self.source,'image')
        image_path=self.image_path

        if not image_path.exists() or len(os.listdir(str(image_path))) == 0:
            #video_file = self.source / 'video.mp4'
            # video_files = glob(os.path.join(self.source, '*.mp4'))
            video_file=os.path.join(self.source_dir, self.name,f'{self.name}.mp4')
            if self.config.fps==-1:
                cap = cv2.VideoCapture(video_file)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"VIDEL FPS: {fps}")
                self.config.fps=fps
            if not os.path.exists(video_file):
                logger.error(f'[ImagesDataset] Neither images nor a video was provided! Execution has stopped! {video_file}')
                exit(1)
            image_path.mkdir(parents=True, exist_ok=True)

            # Prescale to short_side = image_size before extracting frames.
            # Without this step the bbox stage (which inherits the source-
            # video resolution) makes the absolute face px size depend on
            # the input video's resolution, making the same `bb_scale` look
            # too tight on 1080p input and too loose on 360p input. Use
            # `-2` to preserve the aspect ratio while keeping dims even.
            target = int(self.config.image_size[0])
            no_prescale = bool(getattr(self.config, 'no_prescale', False))
            if no_prescale:
                vf = f'fps={self.config.fps}'
            else:
                vf = (
                    f"scale='if(gt(iw,ih),-2,{target})':"
                    f"'if(gt(iw,ih),{target},-2)',fps={self.config.fps}"
                )
            os.system(f'ffmpeg -i {video_file} -vf "{vf}" -start_number 0 -q:v 1 {image_path}/%05d.png')#%05d

        self.images = sorted(glob(f'{image_path}/*.jpg') + glob(f'{image_path}/*.png'),key=natural_sort_key)

    def process_face(self, image):
        lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        if detected_faces is None:
            lmks = None
        else:
            lmks = lmks[0]
        return lmks
    
    def change_file_name(self,image_path,save_path):
        images = sorted(glob(f'{image_path}/*.jpg') + glob(f'{image_path}/*.png'),key=natural_sort_key)
        masks = sorted(glob(f'{save_path}/*.png')+ glob(f'{save_path}/*.jpg'),key=natural_sort_key)
        if len(images) != len(masks):
            raise RuntimeError(
                f"mask count mismatch: {len(images)} images vs {len(masks)} masks. "
                f"The matting subprocess likely failed — re-run crop_and_matting.py "
                f"and watch for the `[rvm stderr]` block printed by robust_video_matting()."
            )
        
        file_type=masks[0].split(".")[-1]
        for i in range(len(images)):
            image_name=images[i].split('/')[-1].split(".")[0]
            os.rename(masks[i],os.path.join(save_path,image_name+"."+file_type))
            
    def robust_video_matting(self,image_path,save_path):
        import subprocess
        print(f"face masking...{image_path}")
        command = [
            'python',
            'preprocess/submodules/RobustVideoMatting/inference.py',
            '--variant', 'resnet50',
            '--checkpoint', 'preprocess/submodules/RobustVideoMatting/rvm_resnet50.pth',
            '--device', 'cuda:0',
            '--input-source', image_path,
            '--output-alpha', save_path,
            '--output-type', 'png_sequence'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        stdout, stderr = process.communicate()
        _print_subprocess_output(stdout, stderr, "rvm", process.returncode)
        if process.returncode != 0:
            raise RuntimeError(f"RobustVideoMatting inference.py exited with code {process.returncode}")

        self.change_file_name(image_path,save_path)

    
    def face_parsing(self,image_path,save_path):
        import subprocess
        print("face parsing...")
        command = [
            'python',
            'preprocess/submodules/face-parsing.PyTorch/test.py',
            "--dspth",image_path,
            "--respth",save_path
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        stdout, stderr = process.communicate()
        _print_subprocess_output(stdout, stderr, "face-parsing", process.returncode)
        if process.returncode != 0:
            raise RuntimeError(f"face-parsing test.py exited with code {process.returncode}")
        print("Finish face parsing.")
    
    def merge_maks(self,image_path,mask_path,seg_path):
        # Get the list of files
        rgb_files = sorted(os.listdir(image_path))
        mask_files = sorted(os.listdir(mask_path))

        seg_files = sorted(os.listdir(seg_path))

        print("Merging mask...")
        # Process each pair of files
        for rgb_file, mask_file, seg_file in tqdm(zip(rgb_files, mask_files, seg_files)):
            # Read the images
            rgb_img = cv2.imread(os.path.join(image_path, rgb_file), cv2.IMREAD_UNCHANGED)
            mask_img = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)
            
            # Create the alpha channel
            alpha_channel = np.ones(mask_img.shape, dtype=np.float32)

            # Set the alpha channel to 0 for the clothes area in the segmentation image (value 15)
            seg_img = cv2.imread(os.path.join(seg_path, seg_file), cv2.IMREAD_GRAYSCALE)
            if self.config.mask_clothes:
                
                alpha_channel[(seg_img == 16) | (seg_img == 0)] = 0.0
            else:
                alpha_channel[ (seg_img == 0)] = 0.0
            alpha_channel = (mask_img / 255.0) * alpha_channel
            
            # Apply alpha to the RGB image
            alpha_expanded = np.expand_dims(alpha_channel, axis=2)
            rgb_img = (rgb_img / 255.0 * alpha_expanded) * 255
            rgb_img = rgb_img.astype(np.uint8)

            # Merge the RGB image with the alpha channel
            alpha_channel = (alpha_channel * 255).astype(np.uint8)
            rgba_img = cv2.merge((rgb_img, alpha_channel))

            # Save the result
            output_path = os.path.join(image_path, rgb_file)
            cv2.imwrite(output_path, rgba_img)

        print("Processing complete!")

    def _compute_video_wide_outer_bbox(self):
        """Compute a single outer bbox shared by every frame.

        Per-frame following crops absorb the head's translation into a
        time-varying crop offset, which then gets baked out of FLAME
        ``translation_code`` — the very signal that head-motion-generation
        downstream models need. Holding the crop offset constant across
        frames keeps that translation in FLAME where it belongs.

        Pass 1: run face_alignment on every frame and collect the bbox
        endpoints (frames where detection fails are skipped — the union
        is dominated by the broader of the *successful* detections).
        Pass 2: take the union (min/max) over all successful frames so
        the bbox covers the head's full motion envelope across the video.
        Pass 3: inflate by ``bbox_scale``, square via the long edge, and
        parity-correct. The bbox is *not* clipped to image bounds;
        :func:`crop_image` handles out-of-bounds by zero-padding so
        ``bbox_scale`` stays linear (clipping the bbox here would collapse
        it onto the short edge whenever the head's motion envelope
        exceeded it, defeating the requested margin).
        """
        cfg = self.config
        logger.info('Computing video-wide outer bbox (FAN union over all frames)...')

        x_mins: list[float] = []
        x_maxs: list[float] = []
        y_mins: list[float] = []
        y_maxs: list[float] = []
        for imagepath in tqdm(self.images, desc='outer_bbox/fan'):
            image = cv2.imread(imagepath)
            if cfg.crop_range is not None:
                image = image[cfg.crop_range[0]:cfg.crop_range[1],
                              cfg.crop_range[2]:cfg.crop_range[3], :]
            lmk = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if lmk is None:
                continue
            lmk = np.asarray(lmk)
            x_mins.append(float(np.min(lmk[:, 0])))
            x_maxs.append(float(np.max(lmk[:, 0])))
            y_mins.append(float(np.min(lmk[:, 1])))
            y_maxs.append(float(np.max(lmk[:, 1])))

        if not x_mins:
            raise RuntimeError(
                'face_alignment detected no face in any frame. Verify the '
                'input video has a visible face throughout, then re-run '
                'crop_and_matting.py.')

        u_xmin, u_xmax = min(x_mins), max(x_maxs)
        u_ymin, u_ymax = min(y_mins), max(y_maxs)
        x_center = int(round((u_xmin + u_xmax) / 2.0))
        y_center = int(round((u_ymin + u_ymax) / 2.0))
        half_extent = max((u_xmax - u_xmin) / 2.0, (u_ymax - u_ymin) / 2.0)
        size = int(cfg.bbox_scale * 2 * half_extent)
        xb_min = x_center - size // 2
        xb_max = x_center + size // 2
        yb_min = y_center - size // 2
        yb_max = y_center + size // 2

        if (xb_max - xb_min) % 2 != 0:
            xb_min += 1
        if (yb_max - yb_min) % 2 != 0:
            yb_min += 1

        return np.array([xb_min, xb_max, yb_min, yb_max])

    def run(self):

        logger.info('Croping dataset (video-wide fixed outer bbox)...')
        cfg = self.config
        bbox = self._compute_video_wide_outer_bbox()
        logger.info(
            f'video-wide outer bbox (xmin,xmax,ymin,ymax)={bbox.tolist()}')

        for imagepath in tqdm(self.images, desc='outer_crop'):
            image = cv2.imread(imagepath)
            if cfg.crop_range is not None:
                image = image[cfg.crop_range[0]:cfg.crop_range[1],
                              cfg.crop_range[2]:cfg.crop_range[3], :]

            if cfg.crop_image:
                image = crop_image_bbox(image, bbox)
                if cfg.image_size[0] == cfg.image_size[1]:
                    image = squarefiy(image, size=cfg.image_size[0])
            elif image.shape[0] != cfg.image_size[0] or image.shape[1] != cfg.image_size[1]:
                image = cv2.resize(
                    image, (cfg.image_size[1], cfg.image_size[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

            cv2.imwrite(imagepath, image)

        if self.config.matting:
            logger.info("Matting dataset...")
            save_mask_path=os.path.join(self.source,"mask")
            os.makedirs(save_mask_path,exist_ok=True)
            save_seg_path=os.path.join(self.source,"seg")
            os.makedirs(save_mask_path,exist_ok=True)
            self.robust_video_matting(self.image_path,save_mask_path)
            
            
            self.face_parsing(self.image_path,save_seg_path)
            self.merge_maks(self.image_path,save_mask_path,save_seg_path)
            shutil.rmtree(save_mask_path)
            shutil.rmtree(save_seg_path)
        logger.info("Done!")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop and matting dataset')
    parser.add_argument('--source', type=str, required=True, help='Source directory')
    parser.add_argument('--name', type=str, required=True, help='Dataset name')
    parser.add_argument('--fps', type=int, default=-1, help='Frame per second')
    parser.add_argument('--crop_image', action='store_true', help='Crop images',default=False)
    parser.add_argument('--crop_range', type=int, nargs='+', default=None, help='Crop range')
    parser.add_argument("--image_size",type=int,nargs=2,default=[512,512],help='croped image size')
    parser.add_argument("--bbox_scale",type=float,default=2.2,help='bbox scale')#2.5
    parser.add_argument("--matting", action='store_true', help='Matting images')
    parser.add_argument("--mask_clothes",type=lambda x: x.lower() in ['true', '1'], default=True, help='remove cloth')
    parser.add_argument("--no_prescale", action='store_true',
                        help='Skip the short-side prescale during ffmpeg '
                             'extraction. Default behaviour scales the input '
                             'video so its short side equals --image_size '
                             'before frames are written, normalising the '
                             'apparent face size across input resolutions. '
                             'Set this when the input video already has the '
                             'expected short-side resolution AND the camera '
                             'intrinsics passed downstream are pinned to it.')

    args=parser.parse_args(sys.argv[1:])
    crop_and_matting = Crop_and_matting(args.source, args)
    crop_and_matting.run()