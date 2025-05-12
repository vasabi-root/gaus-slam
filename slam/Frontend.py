import pytorch3d.ops
import pytorch3d.structures
import pytorch3d.transforms
from scene import Gaussians, LocalMap, Frame
import torch 
from render import Renderer_mapping, Renderer_tracking, Renderer_BA, Renderer_view
from utils.common_utils import get_pointcloud, get_pts_from_depth
import random 
import json 
from slam.Loss import get_loss
from slam.Densify import add_new_gaussians, prune_gaussians
import matplotlib.pyplot as plt
import os 

import torch.multiprocessing as mp
import copy
import matplotlib.pyplot as plt
from queue import Queue
import time
import open3d as o3d
from pytorch3d.ops import iterative_closest_point
import pytorch3d
import numpy as np
import math 
from open3d_ui import Vis_Render
import pdb

class Frontend:
    def __init__(self, config, to_backend: Queue, wandb_run) -> None:
        
        self.config = config 
        self.intrinsics = torch.tensor(config['cameras']['intrinsics'], dtype=torch.float32, device='cuda')
        self.map = Gaussians(config)
        self.local_frames = []
        
        self.cur_lmid = 0
        self.to_backend = to_backend
        
        self.local_frame_ids = []
        
        self.vel = torch.eye(4).cuda().float()
        self.num_tracking_iters = config['frontend']['num_tracking_iters']
        self.num_mapping_iters = config['frontend']['num_mapping_iters']
        self.tau_k = config['frontend']['tau_k']
        self.tau_l = config['frontend']['tau_l']
        self.max_frames = config['frontend']['max_frames']
        self.additional_densify = config['frontend']['additional_densify']
        self.use_wandb = self.config['use_wandb']
        self.vel_pose_init = self.config['frontend']['vel_pose_init']
        self.enable_retracking = self.config['frontend']['enable_retracking']
        self.wandb_run = wandb_run

        #self.vis_render = Vis_Render(config, os.path.join(config['vis_base_dir'], "frontend"))
        self.local_frames_vis = []

        self.numpts_rec = []
        self.depth_l1_rec = []
        self.means_of_moving = 0
        self.tracking_iter_time_sum = 0
        self.tracking_iter_time_count = 0
        self.mapping_iter_time_sum = 0
        self.mapping_iter_time_count = 0
        self.tracking_frame_time_sum = 0
        self.tracking_frame_time_count = 0
        self.mapping_frame_time_sum = 0
        self.mapping_frame_time_count = 0
        
        self.tracking_flag = True
        self.avg_depth_l1 = 0.05 
        
    def create_map(self):
        frame: Frame = self.local_frames[0]
        pcd, initial_scale = get_pointcloud(frame.gt_color, 
                                            frame.gt_depth, 
                                            self.intrinsics, 
                                            c2w=torch.eye(4).cuda().float(), 
                                            compute_mean_sq_dist=True,
                                            sample_num=None)
    
        self.map.create_from_pcd(pcd, initial_scale)
        self.mapping()
    
    def tracking(self, frame: Frame):

        for iter in range(self.num_tracking_iters):
            tracking_start_time = time.time()
            frame.transform.optimizer.zero_grad(set_to_none=True)
            
            render_pkg = Renderer_tracking(self.config, self.map, frame)
            loss = get_loss(self.config, render_pkg, frame, tracking=True)
            loss.backward()
            
            with torch.no_grad():
                frame.transform.optimizer.step()
                frame.transform.update_learning_rate()
            tracking_end_time = time.time()
            self.tracking_iter_time_sum += tracking_end_time - tracking_start_time
            self.tracking_iter_time_count += 1
        
        last_alpha = render_pkg['render_alpha'].detach()
        last_depth = render_pkg['render_depth'].detach()
        mask = torch.logical_and((last_alpha.view(-1) > 0.9), (frame.gt_depth.view(-1) > 0.0001))
        avg_depth_l1 = torch.abs(last_depth.view(-1) - frame.gt_depth.view(-1))[mask].sum() / mask.sum()       
        return avg_depth_l1.item()
    
    def mapping(self, frames=None):
        if frames is None: frames = self.local_frames
        for iter in range(self.num_mapping_iters):
            
            mapping_start_time = time.time()
            frame: Frame = random.choice(frames) # random.choice(frames) # cameras[time_idx]
            self.map.optimizer.zero_grad(set_to_none=True)
            if self.config['render']['enable_exposure']:
                frame.exposure.optimizer.zero_grad(set_to_none=True)
            render_pkg = Renderer_mapping(self.config, self.map, frame)
            loss = get_loss(self.config, render_pkg, frame)
            loss.backward()
            with torch.no_grad():
                self.map.optimizer.step()
                frame.mapping_times += 1
                if self.config['render']['enable_exposure'] and frame.mapping_times > 10:
                    frame.exposure.optimizer.step()
                    frame.exposure.update_learning_rate()

                if self.additional_densify and (frame.mapping_times + 1) % self.config['densify']['densify_interval'] == 0:
                    add_new_gaussians(self.config, self.map, frame, render_pkg)
                    
            mapping_end_time = time.time()
            self.mapping_iter_time_sum += mapping_end_time - mapping_start_time
            self.mapping_iter_time_count += 1
    
    def process_frame(self, time_idx, gt_color, gt_depth, gt_pose):
        """
        Main processing pipeline of the Frontend
        Frontend process one frame at a time
        """
        cur_frame = Frame(self.config, time_idx, gt_color, gt_depth, gt_pose, self.cur_lmid, frame_type=2)
        self.local_frames.append(cur_frame)
        
        # first frame
        if len(self.local_frames) == 1:
            cur_frame.frame_type = 0  # RKF
            cur_frame.start_optimizer(torch.eye(4, dtype=torch.float32, device='cuda'), 
                                         self.config['cameras']['frontend_lr'])
            self.create_map()
            return 
        # initialize for tracking
        tracking_start_time = time.time()
        last_frame: Frame = self.local_frames[-2]
        
        if not self.vel_pose_init: self.vel = torch.eye(4).float().cuda()
        initial_w2c = self.vel @ last_frame.get_w2c.detach()
        lr_dict: dict = self.config['cameras']['frontend_lr'].copy()
        # tracking 
        cur_frame.start_optimizer(initial_w2c, lr_dict)
        depth_l1 = self.tracking(cur_frame)
        # checking whether tracking is lost
        self.depth_l1_rec.append(depth_l1)
        
        tracking_flag = depth_l1 < self.avg_depth_l1 * 5 if self.enable_retracking else True 
        if tracking_flag: self.avg_depth_l1 = 0.9 * self.avg_depth_l1 + 0.1 * depth_l1
        
        tracking_end_time = time.time()
        self.tracking_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_frame_time_count += 1
               
        is_refkf = (not tracking_flag) or len(self.local_frames) > self.max_frames or self.map.size > self.tau_l
        
        if not tracking_flag: 
            cur_frame.start_optimizer(last_frame.get_w2c.detach(), lr_dict)
            self.vel = torch.eye(4).cuda().float()
            print("Tracking failed, reset localmap!!!")
            
        if not is_refkf: 
            
            self.vel = (cur_frame.get_w2c @ torch.linalg.inv(last_frame.get_w2c)).detach()
            render_pkg = Renderer_view(self.config, self.map, w2c=cur_frame.get_w2c.detach())
            alpha_map = render_pkg['render_alpha']
            if ((alpha_map < 0.5).sum() > alpha_map.numel() * self.tau_k):
                cur_frame.frame_type = 1 # keyframe
                # self.save_render_pkg(render_pkg, cur_frame.gt_color, cur_frame.gt_depth, cur_frame.time_idx)
                mapping_start_time = time.time()
                cur_frame.is_keyframe = True
                add_new_gaussians(self.config, self.map, cur_frame, render_pkg=render_pkg)
                self.mapping()
                prune_gaussians(self.config, self.map)
                mapping_end_time = time.time()
                self.mapping_frame_time_sum += mapping_end_time - mapping_start_time
                self.mapping_frame_time_count += 1
            self.local_frames_vis.append(cur_frame)
        # to generate new keyframes
        if is_refkf:
            #self.update_vis()
            # send local map to backend
            local_map_params = self.map.extract_params()
            lm = LocalMap(self.config, self.cur_lmid, self.local_frames, local_map_params, tracking_ok=self.tracking_flag)
            self.to_backend.put(copy.deepcopy(lm))
            self.cur_lmid = self.cur_lmid + 1
            
            # reset local map
            cur_frame = Frame(self.config, time_idx, gt_color, gt_depth, gt_pose, self.cur_lmid, frame_type=0)
            cur_frame.start_optimizer(torch.eye(4, dtype=torch.float32, device='cuda'), lr_dict)
            cur_frame.transform.iteration_times = self.num_tracking_iters
            cur_frame.transform.update_learning_rate(step=False)
            self.local_frames = [cur_frame]
            self.local_frames_vis = [cur_frame]
            self.create_map() 
            self.tracking_flag = tracking_flag

            # waiting backend
            while self.to_backend.qsize() > 1:
                print("backend too busy !!!")
                time.sleep(1)
                
        if self.use_wandb:
            self.wandb_run.log({"Frontend_numpts":self.map.size, "frame_idx":cur_frame.time_idx})
    
    def process_final(self, ):
        if len(self.local_frames) > 1:
            local_map_params = self.map.extract_params()
            lm = LocalMap(self.config, self.cur_lmid, self.local_frames, local_map_params)
            self.cur_lmid = self.cur_lmid + 1
            self.to_backend.put(copy.deepcopy(lm))
        
    def update_common_visualization(self, ):
        vis_base_dir = self.config['vis_base_dir']
        os.makedirs(vis_base_dir, exist_ok=True)
        self.numpts_rec.append(self.map.size)
        
        plt.plot(range(len(self.numpts_rec)), self.numpts_rec)
        plt.savefig(f"{vis_base_dir}/frontend_numpts.png")
        plt.close()

        plt.plot(range(len(self.depth_l1_rec)), self.depth_l1_rec)
        plt.savefig(f"{vis_base_dir}/depth_l1.png")
        plt.close()

    def update_vis(self, ):
        if self.vis_render.queue.empty():
            self.vis_render.reset()
        else:
            while not self.vis_render.queue.empty(): time.sleep(1)

        for frame in self.local_frames_vis:
            frame: Frame
            self.vis_render.update_frame(self.map, frame.get_w2c, frame.frame_type, frame.time_idx)


class mp_Frontend(Frontend):
    def __init__(self, config, dataflow: mp.Queue, to_backend: mp.Queue, event, wandb_run):
        self.dataflow = dataflow
        self.event = event
        super().__init__(config, to_backend, wandb_run)

    def run(self, ):
        should_finish = 0
        total_time = 0
        
        while True:
            
            if should_finish and self.dataflow.empty():
                break 
            
            if not self.dataflow.empty():
                
                msg = self.dataflow.get()
                
                if isinstance(msg, str) and msg == "finish":
                    should_finish = 1
                    self.process_final()
                    continue

                if msg.get("data"):
                    total_time_start = time.time()
                    time_idx, gt_color, gt_depth, gt_pose = msg["data"]
                    self.process_frame(time_idx=time_idx, 
                                       gt_color=gt_color.clone(), 
                                       gt_depth=gt_depth.clone(), 
                                       gt_pose=gt_pose.clone())
                    total_time_end = time.time()
                    total_time += total_time_end - total_time_start
                    if time_idx % 10 == 0:
                        self.update_common_visualization()
                
                del msg
        
        self.to_backend.put("finish")
        self.event.wait()
        print("Frontend process finished !!!")
        num_frames = self.tracking_frame_time_count
        tracking_iter_time_avg = self.tracking_iter_time_sum / self.tracking_iter_time_count
        tracking_frame_time_avg = self.tracking_frame_time_sum / num_frames
        mapping_iter_time_avg = self.mapping_iter_time_sum / self.mapping_iter_time_count
        mapping_frame_time_avg = self.mapping_frame_time_sum / num_frames
        avg_time = total_time / num_frames 
        
        result = {
            "tracking_iter_time(ms)": tracking_iter_time_avg * 1000,
            "tracking_frame_time(s)": tracking_frame_time_avg,
            "mapping_iter_time(ms)": mapping_iter_time_avg * 1000, 
            "mapping_frame_time(s)": mapping_frame_time_avg,
            "frame_time": avg_time, 
        }
        
        vis_base_dir = self.config['vis_base_dir']
        os.makedirs(vis_base_dir, exist_ok=True)
        with open(os.path.join(vis_base_dir, "time.json"), "w") as f:
             json.dump(result, f)
        print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
        print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
        print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
        print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
        print(f"Mapping time: {self.mapping_frame_time_count} s")
        
        