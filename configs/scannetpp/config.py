import os
from os.path import join as p_join

primary_device = "cuda:0"

scenes = ["b20a261fdf", "8b5caf3398", "fb05e13ad1", "2e74812d00", "281bc17764", ]
seed = int(os.environ["SEED"]) if "SEED" in os.environ else 0
scene_name = scenes[int(os.environ["SCENE_NUM"]) if "SCENE_NUM" in os.environ else 0] 

name = 'scannetpp'
use_train_split = True
scene_num_frames = [-1, -1, 250, 250, 250]
num_frames = scene_num_frames[int(os.environ["SCENE_NUM"]) if "SCENE_NUM" in os.environ else 0]

h = 584
w = 876
trans_lr_base = 0.04
rot_lr_base = 0.01
num_tracking_iter = 150
num_mapping_iters = 60
num_ba_iters = 120
localmap_max_frames = 20
edge_growth = False

config = dict(
    vis_base_dir = f'output/{name}_seed{seed}/{scene_name}',
    seed = seed,
    primary_device='cuda',
    use_wandb = False,
    wandb = dict(
        name = name,
        project_name = "GauS_SLAM_ScanNetpp",
    ),
          
    render = dict(
        method = '2dgs', 
        use_sa = True, 
        use_weight_norm = True, 
        enable_exposure = False,
        eps = 1e-6,
        depth_far = 1e2,
        depth_near = 1e-2,
    ),
    
    frontend = dict(
        num_tracking_iters = num_tracking_iter,
        num_mapping_iters = num_mapping_iters,
        tau_k = 0.01, # The proportion of new observed scene, used for keyframe selection.
        tau_l = h * w * 2.5,  # Maximum number of Gaussians in the local map
        max_frames = localmap_max_frames, # Maximum number of frames in the local map
        vel_pose_init  = True, 
        enable_retracking = True, 
        additional_densify = edge_growth, 
    ), 
    backend = dict(
        num_ba_iters = num_ba_iters,
        num_frame_saved = 15,
        num_covis_submaps = 20,
        sleep_time = 0.1, # Prevent continuous backend operations from degrading frontend efficiency.

        mesh_vis = False,
        render_vis = False, 
        common_vis = True, 
        gs_densify = False,
        random_process = True, # random optimization when backend is not busy
        final_refinement = -1,
    ),

    densify=dict(
        use_edge_growth = edge_growth,
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
        dataset_name="scannetpp",
        basedir="./data/scannetpp",
        sequence=scene_name,
        ignore_bad=False,
        use_train_split=use_train_split,
        desired_image_height=584,
        desired_image_width=876,
        start=0,
        end=num_frames,
        stride=1,
        num_frames=num_frames,
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
            cam_rot_lr_init = rot_lr_base / 20, # 0.0004,  # / 100
            cam_rot_lr_final = 0,
            cam_rot_lr_max_step = 2*num_ba_iters,
            cam_trans_lr_init = trans_lr_base / 20, # 0.002, # / 100
            cam_trans_lr_final = 0,
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
        mesh_every=2,
        gen_animation = False, 
        video_freq=15,
        # cam_loc = ([[ 0.08, -0.99,  0.0,-2.75],
        #             [-0.99, -0.08, -0.0, 1.93],
        #             [ 0.0, -0.0, -1.0, 5.55],
        #             [ 0.0,  0.0,  0.0, 1.0 ]])  
    ),
    eval=dict(
        save_renders = False, 
        eval_mesh = False, 
        save_mesh = False, 
        mesh_interval = 5, 
        voxel_size = 0.01,
    ),
)