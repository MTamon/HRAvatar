

import os,logging,time,sys,uuid,shutil,copy
import torch,torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim,l2_loss,TVloss,masked_l1_loss,masked_ssim

from gaussian_renderer import network_gui
import gaussian_renderer
from scene import Head_Scene,GaussianHeadModel
from utils.general_utils import safe_state,save_image_L
from utils.graphics_utils import normal_from_depth_image
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams,init_args,add_more_argument
from metrics import evaluate
from render import render_set,render_multi_views
from scene.data_loader import TrackedData
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError: 
    TENSORBOARD_FOUND = False

def training(all_args, testing_epochs, saving_epochs, checkpoint_epochs, checkpoint, debug_from):
    first_iter = 0
    scene_name=(all_args.source_path).split(os.path.sep)[-1]
    print(f"Model initialization and Data reading....")
    
    os.makedirs(os.path.join(all_args.model_path,"logs"),exist_ok=True)
    
    train_dataset=TrackedData(args.source_path,args,split='train',pre_load=True)
    test_dataset=TrackedData(args.source_path,args,split='test',pre_load=True)
    
    
    gaussians = GaussianHeadModel(all_args.sh_degree,all_args)
    scene = Head_Scene(all_args, gaussians,dataset=train_dataset)
    
    if all_args.epochs==0:
        all_args.epochs=all_args.iterations//train_dataset.data_len
        saving_epochs.append(all_args.epochs)
    all_args.iterations=all_args.epochs*train_dataset.__len__()
    log_file_path=os.path.join(all_args.model_path,"logs",f"({time.strftime('%Y-%m-%d_%H-%M-%S')})_Epoch({all_args.epochs})_({scene_name}).log")
    
    logging.basicConfig(filename=log_file_path, 
                        level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Experiment Configuration: {all_args}")
    
    saving_iterations,testing_iterations,checkpoint_iterations=[i*train_dataset.data_len for i in saving_epochs],\
        [i*train_dataset.data_len for i in testing_epochs],[i*train_dataset.data_len for i in checkpoint_epochs]
        
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, all_args)
    device = gaussians.device
    
    bg_color= 1 if all_args.white_background else 0
    background = torch.tensor([bg_color]*3, dtype=torch.float32, device=device)
    tb_writer = prepare_output_and_logger(all_args)
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    render_temp_path=os.path.join(all_args.model_path,"train_temp_rendering")
    gt_temp_path=os.path.join(all_args.model_path,"train_temp_gt")
    if os.path.exists(render_temp_path):
        shutil.rmtree(render_temp_path)
    if os.path.exists(gt_temp_path):
        shutil.rmtree(gt_temp_path)
    os.makedirs(render_temp_path,exist_ok=True)
    os.makedirs(gt_temp_path,exist_ok=True)
    
    all_args.position_lr_max_steps=all_args.iterations-1000
    all_args.densify_until_iter=all_args.iterations-500
    gaussians.training_setup(all_args)
    
    ema_imloss_for_log,ema_exploss_for_log = 0.0,0.0
    progress_bar = tqdm(range(first_iter, all_args.iterations), desc="Training progress",position=0)
    first_iter += 1
    epoch_loss,epoch_losses=0.0,[]

    logging.info(f"Start trainning....")
    gaussians.set_eval(False)
    render=gaussian_renderer.render_with_deferred
    train_stack = [item for item in train_dataset]
    
    for iteration in range(first_iter, all_args.iterations + 1): 
        epoch=iteration//train_dataset.data_len       
        iter_start.record()
        gaussians.update_learning_rate(iteration,all_args)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a data
        if not train_stack:
            train_stack = [item for item in train_dataset]
        viewpoint_cam_param = train_stack.pop(randint(0, len(train_stack)-1))
        
        gaussians.Envmap.update()
        # Render
        if (iteration - 1) == debug_from:
            all_args.debug = True
            
        gt_image = viewpoint_cam_param.original_image.cuda(device)
        gt_alpha_mask=viewpoint_cam_param.gt_alpha_mask.cuda(device)
        if all_args.random_background:
            bg = torch.rand((3), device=gaussians.device)
            gt_image = gt_image * (1 - gt_alpha_mask) + bg[:,None,None] * gt_alpha_mask
        else:
            bg = background
        render_pkg = render(viewpoint_cam_param, gaussians, all_args, bg,iteration=iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        loss=0.0
        
        if getattr(all_args, "masked_loss", False):
            bg_w = getattr(all_args, "mask_bg_weight", 0.1)
            Ll1 = masked_l1_loss(image, gt_image, gt_alpha_mask, bg_weight=bg_w)
            image_loss = (1.0 - all_args.lambda_dssim) * Ll1 + all_args.lambda_dssim * (1.0 - masked_ssim(image, gt_image, gt_alpha_mask, bg_weight=bg_w))
        else:
            Ll1 = l1_loss(image, gt_image)
            image_loss = (1.0 - all_args.lambda_dssim) * Ll1 + all_args.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss+=getattr(all_args, "lambda_image", 1.0) * image_loss
        cam_o=viewpoint_cam_param.camera_center.cuda(device)
        
        
        if gaussians.with_param_net_smirk :
            jaw_pose_loss=all_args.lambda_jaw_pose*(gaussians.d_jaw_params**2).sum().mean()
            loss+=jaw_pose_loss

        if all_args.with_intrinsic_supervise and iteration>all_args.warm_up_iter:
            if viewpoint_cam_param.albedo is not None:
                persudo_albedo=viewpoint_cam_param.albedo.cuda(device)
                loss+=l1_loss(render_pkg["albedo"], persudo_albedo)*all_args.lambda_intrinsic_albedo

        if all_args.with_depth_supervise  :
            depth_render=render_pkg["depth"]
            normal_render=render_pkg["normal"]
            intrinsic_matrix, extrinsic_matrix = viewpoint_cam_param.intrinsic_matrix, viewpoint_cam_param.extrinsic_matrix
            normal_refer = normal_from_depth_image(depth_render[0], intrinsic_matrix.to(device), extrinsic_matrix.to(device)).permute(2,0,1)
            normal_refer=(normal_refer*(render_pkg["alpha"].detach()))
            if all_args.detach_normal_refer:
                normal_refer=normal_refer.detach()
            noraml_loss=((1-torch.sum(normal_refer*normal_render,dim=0))*(gt_alpha_mask>0.99)).mean()*all_args.lambda_normal
            loss+=noraml_loss
            
            
        if  iteration>all_args.warm_up_iter and all_args.with_envmap_consist:
            difmap,spemap=gaussians.Envmap.diffuse_map,gaussians.Envmap.specular_map
            spemap=torch.nn.functional.interpolate(spemap.permute(0,3,1,2),size=(difmap.shape[1], difmap.shape[2]), mode='bilinear',)
            envmap_consist_loss=l2_loss(difmap,spemap.permute(0,2,3,1))*all_args.lambda_envmap_consist
            loss+=envmap_consist_loss
            
        if all_args.with_tv_roughness  and iteration>all_args.warm_up_iter:
            tv_normal_loss=TVloss(render_pkg["roughness"][None],gt_alpha_mask[None]*render_pkg["alpha"][None])*all_args.lambda_tv_roughness
            loss+=tv_normal_loss

        loss.backward()

        iter_end.record()
        epoch_loss+=loss.item()
        if iteration % train_dataset.data_len == 0:
            tb_writer.add_scalar('train_epoch_losses', epoch_loss/train_dataset.data_len, epoch)
            logging.info(f"[Epoch {epoch}] loss: {epoch_loss/train_dataset.data_len}")
            logging.info(f"[Epoch {epoch}] Guassian points' number: {gaussians._xyz.shape[0]}")
            epoch_loss=0.0
        with torch.no_grad():
            if iteration%200==0 or iteration==1:
                torchvision.utils.save_image(image, os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name + ".png"))
                torchvision.utils.save_image(gt_image, os.path.join(gt_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name + ".png"))
                if iteration>all_args.warm_up_iter :
                    torchvision.utils.save_image((render_pkg["normal"]+1)/2, os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name + "_render_normal.png"))
                    if"albedo" in render_pkg.keys(): 
                        torchvision.utils.save_image((render_pkg["albedo"]), os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name + "_render_albedo.png"))

                if all_args.with_depth_supervise and iteration>all_args.warm_up_iter :
                    torchvision.utils.save_image((normal_refer+1)/2, os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name + "_refer_normal.png"))
                    depth_render=(depth_render-depth_render.min())/(depth_render.max()-depth_render.min())
                    save_image_L(depth_render,os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name + "_render_depth.png"))

                if  iteration>all_args.warm_up_iter:
                    save_image_L(render_pkg["roughness"],os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name + "_render_roughness.png"))
                    save_image_L(render_pkg["fresnel_reflectance"],os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name + "_render_reflectance.png"))
            # Progress bar
            ema_imloss_for_log = 0.4 * image_loss.item() + 0.6 * ema_imloss_for_log
            
            if iteration % 10 == 0:
                
                loss_dict = {
                    "Imloss": f"{ema_imloss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }

                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == all_args.iterations:
                progress_bar.close()

            # Log and save
            gaussians.set_eval(True)
            training_report(tb_writer, iteration, epoch, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (all_args, background),train_dataset,test_dataset)
            gaussians.set_eval(False)
            
            if (iteration in saving_iterations):
                print("\n[EPOCH {} - ITER {}] Saving Gaussians".format(epoch,iteration))
                logging.info("\n[EPOCH {} - ITER {}] Saving Gaussians".format(epoch,iteration))
                scene.save(epoch)

            # Densification
            if iteration < all_args.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                if gaussians.max_radii2D.shape[0]!=visibility_filter.shape[0]:
                    print(f"max_radii2D:{gaussians.max_radii2D.shape[0]},visibility_filter:{visibility_filter.shape[0]} \
                          ,xyz:{gaussians._xyz.shape[0]}")
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > all_args.densify_from_iter and iteration % all_args.densification_interval == 0:
                    size_threshold = 20 if iteration > all_args.opacity_reset_interval else None
                    gaussians.densify_and_prune(all_args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % all_args.opacity_reset_interval == 0 or (all_args.white_background and iteration == all_args.densify_from_iter):
                    gaussians.reset_opacity()
                        

            # Optimizer step
            if iteration < all_args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[EPOCH {} - ITER {}] Saving Checkpoint".format(epoch,iteration))
                logging.info("\n[EPOCH {} - ITER {}] Saving Checkpoint".format(epoch,iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(epoch) + ".pth")
    print("Training complete.")
    logging.info(f"Training complete.")
    
   
    if all_args.render_and_eval:
        gaussians.set_eval(True)
        
        with torch.no_grad():
            
            print("Rendering...")
            logging.info(f"Rendering training set....")
            render_set(all_args.model_path, "train", epoch, train_dataset, gaussians, all_args, bg,all_args)
            logging.info(f"Rendering testing set....")
            render_set(all_args.model_path, "test", epoch, test_dataset, gaussians, all_args, bg,all_args)
            
            logging.info(f"Rendering training set with multiview....")
            render_multi_views(all_args.model_path, "train", epoch, train_dataset, gaussians, all_args, bg)
            logging.info(f"Rendering testing set with multiview....")
            render_multi_views(all_args.model_path, "test", epoch, test_dataset, gaussians, all_args, bg)
            
            print("Evaluating on testing set....")
            logging.info(f"Evaluating on testing set....")
            evaluate([all_args.model_path])
            logging.info(f"Evaluating complete.")
        gaussians.set_eval(False)

    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/events", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        os.makedirs(os.path.join(args.model_path, "envents"), exist_ok = True)
        tb_writer = SummaryWriter(os.path.join(args.model_path,"envents"))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration,epoch, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Head_Scene, renderFunc, renderArgs,train_dataset,test_dataset,):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        logging.info(f"[EPOCH {epoch} - ITER {iteration}] Testing...")
        logging.info(f"[EPOCH {epoch} - ITER {iteration}] Guassian points' number:{scene.gaussians._xyz.shape[0]}")
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [test_dataset[idx]for idx in range(len(test_dataset))]}, 
                              {'name': 'train', 'cameras' : [train_dataset[idx % len(train_dataset)] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            logging.info(f"Start evaluate {config['name']} set...")
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(image.device), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[EPOCH {} - ITER {}] Evaluating {}: L1 {} PSNR {}".format(epoch,iteration, config['name'], l1_test, psnr_test))
                logging.info("\n[EPOCH {} - ITER {}] Evaluating {}: L1 {} PSNR {}".format(epoch,iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_epochs", nargs="+", type=int, default=[i*5 for i in range(1, 10)])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[15])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser=add_more_argument(parser)
    args = parser.parse_args(sys.argv[1:])
    args.save_epochs.append(args.epochs)
    args.test_epochs.append(args.epochs)
    
    args=init_args(args)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(args, args.test_epochs, args.save_epochs, args.checkpoint_epochs, args.start_checkpoint, args.debug_from,)

    # All done
    print("\nTraining complete.")
