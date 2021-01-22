#!/usr/bin/env python3
import traceback
import sys

try:
    import open3d as o3d
except ImportError:
    traceback.print_exc()
    print("Importing open3d failed, please run the following command or visit https://pypi.org/project/open3d/")
    print()
    print(sys.executable  + " -m pip install open3d")


class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.depth_map = None
        self.rgb = None
        self.pcl = None
        # self.depth_scale = depth_scale
        # self.depth_trunc = depth_trunc
        # self.rgbd_mode = rgbd_mode
        # self.pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsic_file)

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.isstarted = False


        # intrinsic = o3d.camera.PinholeCameraIntrinsic()
        # print(str(type(intrinsic)))
        # open3d::camera::PinholeCameraIntrinsic instrinsic;  # TODO: Try this
        # instrinsic.SetIntrinsics(480, 272, 282.15, 321.651, 270, 153.9);
        # self.vis.add_geometry(self.pcl)


    # def dummy_pcl(self):
    #     x = np.linspace(-3, 3, 401)
    #     mesh_x, mesh_y = np.meshgrid(x, x)
    #     z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    #     z_norm = (z - z.min()) / (z.max() - z.min())
    #     xyz = np.zeros((np.size(mesh_x), 3))
    #     xyz[:, 0] = np.reshape(mesh_x, -1)
    #     xyz[:, 1] = np.reshape(mesh_y, -1)
    #     xyz[:, 2] = np.reshape(z_norm, -1)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(xyz)
    #     return pcd

    def rgbd_to_projection(self, depth_map, rgb):
        self.depth_map = depth_map
        self.rgb = rgb
        rgb_o3d = o3d.geometry.Image(self.rgb)
        depth_o3d = o3d.geometry.Image(self.depth_map)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d)
        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors
        return self.pcl


    def visualize_pcd(self):
        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()

    # def depthmap_to_projection(self, depth_map):
    #     # depth_map[depth_map == 65535] = 0
    #     # self.depth_map = 65535 // depth_map
    #     self.depth_map = depth_map
    #     img = o3d.geometry.Image(self.depth_map)
    #
    #     if not self.isstarted:
    #         print("inhere")
    #         self.pcl = o3d.geometry.PointCloud.create_from_depth_image(img, self.pinhole_camera_intrinsic)
    #         self.vis.add_geometry(self.pcl)
    #         self.isstarted = True
    #     else:
    #         print("not inhere")
    #         pcd = o3d.geometry.PointCloud.create_from_depth_image(img, self.pinhole_camera_intrinsic, depth_scale = 1000, depth_trunc = 50)
    #         self.pcl.points = pcd.points
    #         self.vis.update_geometry(self.pcl)
    #         self.vis.poll_events()
    #         self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()



def depthmap_to_projection(depth_map, intrinsic, stride = 1, depth_scale = 1, depth_trunc = 96):
    # o3d_intrinsic_matrix = o3d.camera.PinholeCameraIntrinsicParameters()
    # pinhole_camera_intrinsic
    # o3d_intrinsic_matrix.set_intrinsics(depth_map.shape[1], depth_map.shape[0],
    #                                     intrinsic[0, 0], intrinsic[1, 1],
    #                                     intrinsic[0, 2], intrinsic[1, 2])

    img = o3d.geometry.Image((depth_map))
    pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic("intrinsic.json")
    print(pinhole_camera_intrinsic.intrinsic_matrix)

    # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_map, pinhole_camera_intrinsic, depth_scale=depth_scale,
    #                                                       depth_trunc=depth_trunc, stride=stride)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(img, pinhole_camera_intrinsic)

    return pcd

def visualize(pcd):

    # o3d.visualization.draw_geometries([pcd], zoom=0.3412,
    #                               front=[0.4257, -0.2125, -0.8795],
    #                               lookat=[2.6172, 2.0475, 1.532],
    #                               up=[-0.0694, -0.9768, 0.2024])
    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([pcd])

# def depthmap_to_projection(depth_map, M): # can add step size which does subsampling
#     c_x = M[2,2]
#     c_y = M[2,2]
#     f_x = M[0,0]
#     f_y = M[1,1]
#     point_cloud = []
#     rows, cols = depth_map.shape
#     for u in range(rows):
#         for v in range(cols):
#             if(depth_map[u,v] == 65535 or depth_map[u,v] == 0) # Not sure about zero
#                 continue
#             z = depth_map[u,v]
#             x = (u - c_x) * z / f_x
#             y = (v - c_y) * z / f_y
#             point_cloud.append((x,y,z))
#     return point_cloud
            

