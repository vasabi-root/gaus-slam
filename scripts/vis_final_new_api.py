import sys
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
print("System Paths:")
for p in sys.path:
    print(p)

from scene import Gaussians, Frame
from render import Renderer_view
import torch
from utils.common_utils import get_pointcloud
import numpy as np
import time
from scene import load_scence, Gaussians, Frame
import argparse
import os 
import open3d as o3d
from open3d.visualization import gui, rendering

class Visualization:
    
    def __init__(self, config: dict) -> None:
        
        self.config = config
        self.viz_scale = config['viz']['view_scale']
        config['viz']['viz_h'] = 1552
        config['viz']['viz_w'] = 2560
        self.viz_h = int(config['viz']['viz_h'])
        self.viz_w = int(config['viz']['viz_w'])
    
        self.intrinsics = np.array([
            [self.viz_w/2 * 1.83, 0, self.viz_w/2],
            [0, self.viz_w/2 * 1.83, self.viz_h/2],
            [0, 0, 1],
        ])
        config['cameras']['intrinsics'] = self.intrinsics
        config['cameras']['height'] = self.viz_h
        config['cameras']['width'] = self.viz_w
        
        self.is_created = False
        self.pcd = None
        gui.Application.instance.initialize()
        self.win = gui.Application.instance.create_window("Viz", self.viz_w, self.viz_h)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.win.renderer)
        self.widget.scene.set_background([1, 1, 1, 1])
        self.win.add_child(self.widget)
    
    def create_visualization(self, gaussians: Gaussians, first_w2c):
        extrinsic = first_w2c.detach().contiguous().double().cpu().numpy()
        extrinsic[:3, 3] += np.array([0, 0, 0.1])
        
        Rc2w = extrinsic[:3, :3].T
        Tc2w = -extrinsic[:3, :3] @ extrinsic[:3, 3:]
        insight_point_c = np.array([0, 0, 0]).T
        insight_point_w = Rc2w @ insight_point_c + Tc2w  # Исправление формы
        dummy_pcd = o3d.geometry.PointCloud()
        dummy_pcd.points = o3d.utility.Vector3dVector(insight_point_w)
        dummy_pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(insight_point_w))
        mat_dummy = rendering.MaterialRecord()
        mat_dummy.shader = "defaultUnlit"
        self.widget.scene.add_geometry("dummy", dummy_pcd, mat_dummy)
        
        with torch.no_grad():
            render_pkg = Renderer_view(self.config, gaussians, w2c=first_w2c)
            color = render_pkg['render_color'].permute(1, 2, 0)
            depth = render_pkg['render_depth'].permute(1, 2, 0)
            depth = torch.nan_to_num(depth / render_pkg['render_alpha'].permute(1, 2, 0), 0, 0)
            point_cloud = get_pointcloud(color, depth, self.intrinsics, w2c=first_w2c)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(point_cloud.points.contiguous().double().cpu().numpy())
        self.pcd.colors = o3d.utility.Vector3dVector(point_cloud.colors.contiguous().double().cpu().numpy())
        mat_pcd = rendering.MaterialRecord()
        mat_pcd.shader = "defaultUnlit"
        mat_pcd.point_size = self.viz_scale
        self.widget.scene.add_geometry("pcd", self.pcd, mat_pcd)
        
        # Вычисление bounding box
        bbox = self.pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # Для отладки
        
        # Установка камеры
        self.widget.setup_camera(self.intrinsics, extrinsic, self.viz_w, self.viz_h, bbox)
        
        # Установка центра вращения (замените на ваш desired_center)
        center = np.asarray(self.pcd.get_center())  # Центр point cloud
        eye = -Rc2w @ Tc2w  # Позиция камеры в world
        up = Rc2w @ np.array([0, -1, 0])  # Предполагая Y-up, скорректируйте
        self.widget.look_at(center, eye, up)
        
        self.is_created = True
        
    def update(self, gaussians: Gaussians):
        if not self.is_created: return
        cam_params = self.widget.scene.camera.get_projection_matrix()  # Не extrinsic, но для render используем view matrix
        view_matrix = np.array(self.widget.scene.camera.get_view_matrix())  # 4x4 view (c2w inverse)
        w2c = np.linalg.inv(view_matrix)  # w2c from view_matrix (c2w)
        w2c = torch.tensor(w2c).cuda().float()
        with torch.no_grad():
            ts = time.time()
            render_pkg = Renderer_view(self.config, gaussians, w2c=w2c)
            td = time.time()
            color = render_pkg['render_color'].permute(1, 2, 0)
            depth = render_pkg['render_depth'].permute(1, 2, 0)
            depth = torch.nan_to_num(depth / render_pkg['render_alpha'].permute(1, 2, 0), 0, 0)
            point_cloud = get_pointcloud(color, depth, self.intrinsics, w2c=w2c)
            if point_cloud.points.shape[0] > 0:
                self.widget.scene.remove_geometry("pcd")
                self.pcd.points = o3d.utility.Vector3dVector(point_cloud.points.contiguous().double().cpu().numpy())
                self.pcd.colors = o3d.utility.Vector3dVector(point_cloud.colors.contiguous().double().cpu().numpy())
                mat_pcd = rendering.MaterialRecord()
                mat_pcd.shader = "defaultUnlit"
                mat_pcd.point_size = self.viz_scale
                self.widget.scene.add_geometry("pcd", self.pcd, mat_pcd)
        gui.Application.instance.post_to_main_thread(self.win, self.win.force_redraw)
        return td - ts
            
    def destroy(self):
        gui.Application.instance.quit()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="Path to experiment file")
    args = parser.parse_args()

    result_path = os.path.join(args.m, "save")
    config, gaussians, w2cs, gt_w2cs = load_scence(result_path)
    
    vis = Visualization(config)
    vis.create_visualization(gaussians, w2cs[0])
    
    total_time = 0
    cnt = 0
    
    def tick():
        total_time += vis.update(gaussians)
        cnt += 1
        if cnt % 100 == 0:
            print("fps:", 1 / (total_time / cnt))
        return True
    
    vis.win.set_on_key(tick)
    gui.Application.instance.run()