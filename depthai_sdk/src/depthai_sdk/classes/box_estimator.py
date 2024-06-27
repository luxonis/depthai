import numpy as np
import cv2
import random
from typing import Tuple
import open3d as o3d
import json
from depthai_sdk.logger import LOGGER

N_POINTS_SAMPLED_PLANE = 3
MAX_ITER_PLANE = 300

class BoxEstimator:
    def __init__(self, median_window=3, calib_json_path: str = None, threshold=50):
        """
        Box estimator helper class. Currently it's applicable for scanning only one box at a time.

        Args:
            threshold (int, optional): Distance threshold for plane fitting. Defaults to 50 mm.
        """
        self.top_side_pcl = None

        self.ground_plane_eq = None
        self.threshold = threshold

        self.height = None
        self.width = None
        self.length = None

        self.bounding_box = None
        self.rotation_matrix = None
        self.translate_vector = None

        # Median filter
        self.median_window = median_window
        self.prev_dimensions = []

        self.pcd = o3d.geometry.PointCloud()
        self.plane_pcd = o3d.geometry.PointCloud()
        self.box_pcd = o3d.geometry.PointCloud()

        self.corners = None

        # Try to find plane_eq.json file (where user executed script)
        if calib_json_path is None:
            calib_json_path = 'plane_eq.json'

        try:
            with open(calib_json_path, 'r') as f:
                data = json.load(f)
                if 'eq' in data:
                    self.ground_plane_eq = np.array(data['eq'], dtype=np.float64)
                else:
                    raise ValueError("Invalid JSON file")
        except FileNotFoundError:
            LOGGER.info("No plane_eq.json file found. Please run calibrate() method to calibrate the ground plane.")

    def get_outliers(self, points) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get outliers and inliers from a point cloud
        """
        A,B,C,D = self.ground_plane_eq
        # Calculate the denominator (constant for all points)
        denominator = np.sqrt(A**2 + B**2 + C**2)
        threshold = 45 # mm
        distances = np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D) / denominator
        return (points[distances > self.threshold], points[distances <= self.threshold])

    def is_calibrated(self) -> bool:
        return self.ground_plane_eq is not None

    def calibrate(self, points: np.ndarray):
        """
        Calibrate the ground plane equation using a point cloud

        Args:
            points: Point cloud

        Returns:
            None
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        pcd = pcd.voxel_down_sample(voxel_size=10)
        pcd = pcd.remove_statistical_outlier(30, 0.1)[0]

        # Get plane segmentation
        plane_eq, plane_inliers = pcd.segment_plane(self.threshold, N_POINTS_SAMPLED_PLANE, MAX_ITER_PLANE)
        inlier_points_num = len(plane_inliers)
        inline_percentage = inlier_points_num / len(points)
        if inline_percentage < 0.8:
            LOGGER.error("Not enough inliers found. Please try again.")
            return False

        self.ground_plane_eq = plane_eq
        with open('plane_eq.json', 'w') as f:
            # Convert np.array to list
            json.dump({'eq': self.ground_plane_eq.tolist()}, f)
        return True


    def get_plane_mesh(self, size=10, divisions=10) -> Tuple:
        """
        Create a mesh representation of a plane given its equation Ax + By + Cz + D = 0.

        Args:
            size: Size of the plane (default: 10)
            divisions: Number of divisions in each dimension (default: 10)

        Returns:
            positions: List of 3D points
            indices: List of indices for triangles
            normals: List of normal vectors for each vertex
        """
        # Normalize the normal vector
        A,B,C,D = self.ground_plane_eq
        normal = np.array([A, B, C])
        normal = normal / np.linalg.norm(normal)

        # Create a grid of points
        x = np.linspace(-size/2, size/2, divisions)
        y = np.linspace(-size/2, size/2, divisions)
        X, Y = np.meshgrid(x, y)

        # Calculate Z values
        if C != 0:
            Z = (-A*X - B*Y - D) / C
        elif B != 0:
            Z = (-A*X - C*Y - D) / B
            Y, Z = Z, Y
        else:
            Z = (-B*Y - C*Z - D) / A
            X, Z = Z, X

        # Create positions
        positions = np.stack((X, Y, Z), axis=-1).reshape(-1, 3).tolist()

        # Create indices
        indices = []
        for i in range(divisions - 1):
            for j in range(divisions - 1):
                square_start = i * divisions + j
                indices.extend([
                    square_start, square_start + 1, square_start + divisions,
                    square_start + 1, square_start + divisions + 1, square_start + divisions
                ])

        # Create normals
        normals = [normal.tolist() for _ in range(len(positions))]

        return positions, indices, normals

    def process_points(self, points_roi: np.ndarray) -> Tuple:
        self.pcd.points = o3d.utility.Vector3dVector(points_roi)
        self.pcd = self.pcd.voxel_down_sample(voxel_size=10)
        self.pcd = self.pcd.remove_statistical_outlier(30, 0.1)[0]

        self.box_pcl, self.plane_pcl = self.get_outliers(np.asarray(self.pcd.points))

        # Remove outliers
        self.plane_pcd = self.plane_pcd.remove_statistical_outlier(30, 0.1)[0]
        self.box_pcd = self.box_pcd.remove_statistical_outlier(30, 0.1)[0]


        if len(self.box_pcl) < 100:
            return None, None  # No box

        self.get_box_top(self.ground_plane_eq)
        dimensions = self.get_dimensions()
        self.prev_dimensions.append(dimensions)
        corners = self.get_3d_corners()
        return self._filtered_dimensions(), corners

    @property
    def box_pcl(self):
        return np.asarray(self.box_pcd.points)
    @box_pcl.setter
    def box_pcl(self, points):
        self.box_pcd.points = o3d.utility.Vector3dVector(points)
    @property
    def plane_pcl(self):
        return np.asarray(self.plane_pcd.points)
    @plane_pcl.setter
    def plane_pcl(self, points):
        self.plane_pcd.points = o3d.utility.Vector3dVector(points)

    def _filtered_dimensions(self):
        # If too many dimensions, remove the oldest one
        if len(self.prev_dimensions) > self.median_window:
            self.prev_dimensions.pop(0)

        # Get median of the dimensions
        length = np.median([d[0] for d in self.prev_dimensions])
        width = np.median([d[1] for d in self.prev_dimensions])
        height = np.median([d[2] for d in self.prev_dimensions])
        return length, width, height


    def create_rotation_matrix(self, vec_in, vec_target):
        v = np.cross(vec_in, vec_target)
        s = np.linalg.norm(v)
        c = np.dot(vec_in, vec_target)
        v_mat = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
        R = np.eye(3) + v_mat + (np.matmul(v_mat, v_mat) * (1 / (1 + c)))
        self.rotation_matrix = R
        return R

    def get_box_top(self, plane_eq):
        rot_matrix = self.create_rotation_matrix(plane_eq[0:3], [0, 0, 1])
        self.plane_pcl = self.rotate_points(self.plane_pcl, rot_matrix)
        avg_z = np.average(self.plane_pcl[:, 2])

        translate_vector = [0, 0, -avg_z]
        self.translate_vector = np.array(translate_vector)

        self.plane_pcl = self.translate_points(self.plane_pcl, translate_vector)
        self.box_pcl = self.rotate_points(self.box_pcl, rot_matrix)
        self.box_pcl = self.translate_points(self.box_pcl, translate_vector)

        top_plane_eq, top_plane_inliers = self.fit_plane_vec_constraint([0, 0, 1], self.box_pcl, 3, 30)

        top_plane = self.box_pcl[top_plane_inliers]
        # self.side_planes = self.box_pcl[np.setdiff1d(np.arange(len(self.box_pcl)), top_plane_inliers)]
        self.top_side_pcl = top_plane
        self.height = abs(top_plane_eq[3])
        return self.height

    def get_dimensions(self):
        upper_plane_points = self.top_side_pcl
        coordinates = np.c_[upper_plane_points[:, 0], upper_plane_points[:, 1]].astype('float32')
        rect = cv2.minAreaRect(coordinates)
        self.bounding_box = cv2.boxPoints(rect)
        self.width, self.length = rect[1][0], rect[1][1]

        return self.length, self.width, self.height

    def get_3d_lines(self, corners):
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
        ]
        line_segments = []
        for line in lines:
            line_segments.append(corners[line[0]])
            line_segments.append(corners[line[1]])
        return np.array(line_segments)

    def get_3d_corners(self):
        bounding_box = self.bounding_box
        points_floor = np.c_[bounding_box, np.zeros(4)]
        points_top = np.c_[bounding_box, -self.height * np.ones(4)]
        box_points = np.concatenate((points_floor, points_top))

        self.corners = box_points
        return box_points

    def inverse_corner_points(self):
        # Apply the inverse of the translation and rotation to get back to the original coordinates
        inverse_translation = -self.translate_vector
        inverse_rot_mat = np.linalg.inv(self.rotation_matrix)

        box_points = self.translate_points(self.corners, inverse_translation)
        box_points = self.rotate_points(box_points, inverse_rot_mat)
        return box_points

    def fit_plane_vec_constraint(self, norm_vec, pts, thresh=0.05, n_iterations=300):
        best_eq = []
        best_inliers = []

        n_points = pts.shape[0]
        for _ in range(n_iterations):
            id_sample = random.sample(range(0, n_points), 1)
            point = pts[id_sample]
            d = -np.sum(np.multiply(norm_vec, point))
            plane_eq = [*norm_vec, d]
            pt_id_inliers = self.get_plane_inliers(plane_eq, pts, thresh)
            if len(pt_id_inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers = pt_id_inliers

        return best_eq, best_inliers

    def get_plane_inliers(self, plane_eq, pts, thresh=0.05):
        dist_pt = self.get_pts_distances_plane(plane_eq, pts)
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
        return pt_id_inliers

    def get_pts_distances_plane(self, plane_eq, pts):
        dist_pt = (plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1]
                   + plane_eq[2] * pts[:, 2] + plane_eq[3]) \
                  / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)
        return dist_pt

    def rotate_points(self, points, rotation_matrix):
        return np.dot(points, rotation_matrix.T)

    def translate_points(self, points, translate_vector):
        return points + translate_vector