import os
from os.path import join as p_join

primary_device = "cuda:0"

scenes = ["scene0000_00", "scene0059_00", "scene0106_00", 
          "scene0169_00", "scene0181_00", "scene0207_00"]
 
seed = int(os.environ["SEED"]) if "SEED" in os.environ else 0
scene_name = scenes[int(os.environ["SCENE_NUM"]) if "SCENE_NUM" in os.environ else 2] 

name = 'scannet'

h = 480
w = 640
trans_lr_base = 0.004
rot_lr_base = 0.001
num_tracking_iter = 100
num_ba_iters = 80
localmap_max_frames = 40

config = dict(
    vis_base_dir = f'output/{name}_seed{seed}/{scene_name}',
    seed = seed,
    primary_device='cuda',
    use_wandb = False,
    wandb = dict(
        name = name,
        project_name = "GauS_SLAM_ScanNet",
    ),
          
    render = dict(
        method = '2dgs', # ["2dgs", "3dgs", "radegs"]
        use_sa = True, 
        use_weight_norm = True, 
        enable_exposure = False,
        eps = 1e-6,
        depth_far = 1e2,
        depth_near = 1e-2,
    ),
    
    frontend = dict(
        num_tracking_iters = num_tracking_iter,
        num_mapping_iters = localmap_max_frames,
        converged_th = -1,
        tau_k = 0.05, # The proportion of new observed scene, used for keyframe selection.
        tau_l = h * w * 1.5,  # Maximum number of Gaussians in the local map
        max_frames = localmap_max_frames, # Maximum number of frames in the local map
        vel_pose_init  = True,
        enable_retracking = False,  
        additional_densify = False, 
    ), 
    backend = dict(
        num_ba_iters = num_ba_iters,
        num_frame_saved = localmap_max_frames // 4,
        num_covis_submaps = 30,
        sleep_time = 0.1, # Prevent continuous backend operations from degrading frontend efficiency.
        
        mesh_vis = False,
        render_vis = False, 
        common_vis = True, 
        gs_densify = False,
        random_process = True, # random optimization when backend is not busy
        final_refinement = -1,
    ),

    densify=dict(
        use_edge_growth = False,
        densify_interval=20,
        method='splatam',
        sil_thres=0.6,
        edge_thres=0.4,
        opacity_cuil=0.05,
        scale_cuil=5e-4,
        scale_max=0.1,
        num_addpts= h * w,
        percent_dense=0.01,
        densify_grad_threshold=0.0002,
        extent=2,
    ),    
    
    loss = dict(
        ignore_outliners = False,
        use_normal_loss = False,
        silmask_th = 0.90,
        tracking = dict(
            color=0.5,
            depth=1.0,
            normal = 0,
            dist = 0,
        ),
        mapping = dict(
            color=0.5,
            depth=1.0,
            normal = 0,
            dist = 0.1,
        ),
    ),
    data=dict(
        dataset_name="scannet",
        basedir="./data/scannet",
        gradslam_data_cfg="./configs/data/scannet.yaml",
        sequence=scene_name,
        desired_image_height=h,
        desired_image_width=w,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
    ),
    gaussians=dict(
        gaussian_distribution = "anisotropic", # ["isotropic", "anisotropic"]
        training_args=dict(
            xyz_lr= 0.0001,
            feature_lr=0.0025,
            rgb_lr=0.0025,
            rotation_lr= 0.001,
            opacity_lr=0.05,
            scaling_lr=0.001,
        ),

    ),
    cameras=dict(
        adam_betas = (0.7, 0.99),
        frontend_lr=dict(
            cam_rot_lr_init = rot_lr_base, # 0.0004,  # / 100
            cam_rot_lr_final = rot_lr_base / 10,
            cam_rot_lr_max_step = num_tracking_iter, 
            cam_trans_lr_init = trans_lr_base, # 0.002, # / 100
            cam_trans_lr_final = trans_lr_base / 10,
            cam_trans_lr_max_step = num_tracking_iter,
            exposure_lr_init = 0.0001,
            exposure_lr_final = 0.0001,
            exposure_lr_max_step = 100,
        ),
        backend_lr=dict(
            cam_rot_lr_init = rot_lr_base / 2, # 0.0004,  # / 100
            cam_rot_lr_final = rot_lr_base / 10,
            cam_rot_lr_max_step = 2*num_ba_iters,
            cam_trans_lr_init = trans_lr_base / 2, # 0.002, # / 100
            cam_trans_lr_final = trans_lr_base / 10,
            cam_trans_lr_max_step = 2*num_ba_iters,
            exposure_lr_init = 0.0001,
            exposure_lr_final = 0.0001,
            exposure_lr_max_step = 100,
        )
    ),
    viz=dict(
        viz_w=600,  
        viz_h=340,
        view_scale=2,
        mesh_every=5,
        gen_animation = False, 
        video_freq=30,
        cam_loc = ([[ 1.0, -0.0,  0.0,-3.08],
                    [-0.0, -1.0, -0.0, 1.14],
                    [ 0.0, -0.0, -1.0, 5.83],
                    [ 0.0,  0.0,  0.0, 1.0 ]]),
    ),
    eval=dict(
        save_renders = False, 
        eval_mesh = False, 
        save_mesh = False, 
        mesh_interval = 5, 
        voxel_size = 0.01,
    ),
)