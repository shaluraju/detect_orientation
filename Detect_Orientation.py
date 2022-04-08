#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import open3d
import cv2
import matplotlib.pyplot as plt
import copy
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from pickletools import uint8
from utils.lib_io import read_yaml_file
from utils.lib_ransac import PlaneModel, RansacPlane
from utils.lib_geo_trans import world2pixel
from utils_rgbd.lib_rgbd import CameraInfo, resize_color_and_depth
from utils_rgbd.lib_open3d import wrap_open3d_point_cloud_with_my_functions
from utils_rgbd.lib_plot_rgbd import drawMaskFrom2dPoints, draw3dArrowOnImage
wrap_open3d_point_cloud_with_my_functions()


MAX_OF_MAX_PLANE_NUMBERS = 1

def subtract_points(points, indices_to_subtract):
    ''' Subtract points by indices and return a new sub cloud. '''
    all_indices = np.arange(len(points))
    rest_indices = np.setdiff1d(all_indices, indices_to_subtract)
    points = points[rest_indices]
    return 

def calc_opposite_point(p0, p1, length=5.0, to_int=True):
    ''' p0 and p1 are two points.
        Create a point p2 
        so that the vector (p2, p0) and (p0, p1) are at the same line.
        length is the length of p2~p0.
    '''
    x0, y0 = p0
    x1, y1 = p1
    theta = np.arctan2(y1 - y0, x1 - x0)
    x2 = x0 - length * np.cos(theta)
    y2 = y0 - length * np.sin(theta)
    if to_int:
        x2, y2 = int(x2), int(y2)
    return (x2, y2)

class PlaneParam(object):
    ''' The parameters of the detected plane are stored in this class. '''

    def __init__(
            self, w, pts_3d_center, normal_vector, pts_2d_center, mask_color):
        self.w = w
        self.pts_3d_center = pts_3d_center
        self.normal_vector = normal_vector
        self.pts_2d_center = pts_2d_center
        self.mask_color = mask_color

    def resize_2d_params(self, ratio):
        self.pts_2d_center *= ratio

    def print_params(self, index=""):
        ''' Print the plane parameters. '''
        print("-----------------------------------")
        print("Plane {}: ".format(index))
        print("     weights: {}".format(self.w))
        print("     normal: {}".format(self.normal_vector))
        print("     3d center: {}".format(self.pts_3d_center))
        print("     2d center: {}".format(self.pts_2d_center))
        print("     mask color: {}".format(self.mask_color))

class PlaneDetector(object):
    def __init__(self, config_file_path, camera_info_file_path):

        # -- Load config file.
        self._cfg = read_yaml_file(
            config_file_path, is_convert_dict_to_class=True)

        # -- Load camera intrinsics from file.
        self._cam_intrin = CameraInfo(camera_info_file_path)
        self._shape = self._cam_intrin.get_img_shape()

        # -- Algorithm settings.

        # Settings for reducing image size.
        self._cam_intrin_resized = copy.deepcopy(self._cam_intrin)
        self._cam_intrin_resized.resize(self._cfg.img_resize_ratio)
        self._shape_resized = self._cam_intrin_resized.get_img_shape()

        #assert(self._cfg.max_number_of_planes <= MAX_OF_MAX_PLANE_NUMBERS)

        # -- Visualization settings.
        self._cmap = plt.get_cmap(self._cfg.visualization["color_map_name"])

    def detect_planes(self, depth_img, color_img=None):
        '''
        Arguments:
            depth_img {np.ndarry, np.uint16}:
                Undistorted depth image.
            color_img {None} or {np.ndarry, np.uint8, bgr, undistorted}:
                Color image is only for visualiation purpose.
                If None, color_img will be created as a black image.
        '''

        #-- Check input.
        if len(depth_img.shape) != 2:
           raise RuntimeError("Depth image should have 2 channels.")
        if color_img is None:  # Use black image instead.
            r, c = depth_img.shape
            color_img = np.zeros(shape=(r, c, 3), dtype=np.uint8)

        # -- Resize image.
        color_img_resized, depth_img_resized = resize_color_and_depth(
            color_img, depth_img, self._cfg.img_resize_ratio)

        # -- Compute point cloud.
        pcd = self._create_point_cloud(color_img_resized, depth_img_resized)
        if self._cfg.debug["draw_3d_point_cloud"]:
            pcd.draw()
        points = pcd.get_xyzs()
        # points.shape=(N, 3). Each row is a point's 3d position of (x, y, z).

        # -- Detect plane one by one until there is no plane.
        planes = []
        for i in range(self._cfg.max_number_of_planes):
            print("-------------------------")
            print("Start detecting {}th plane ...".format(i))

            # Detect plane by RANSAC.
            is_succeed, plane_weights, plane_pts_indices = \
                self._detect_plane_by_RANSAC(points)
            if not is_succeed:
                break

            # Store plane result.
            plane_points = points[plane_pts_indices]
            planes.append(self._Plane(plane_weights, plane_points))

            # Use the remaining point cloud to detect next plane.
            points = subtract_points(points, plane_pts_indices)
        print("-------------------------")
        print("Plane detection completes. Detect {} planes.".format(len(planes)))


        # -- Return.
        return planes

    class _Plane(object):
        def __init__(self, plane_weights, plane_points):
            self.weights = plane_weights
            self.points = plane_points

    def _create_point_cloud(self, color_img_resized, depth_img_resized):
        ''' Create point cloud from color and depth image.
        Return:
            pcd {open3d.geometry.PointCloud}
        '''

        rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
            color= open3d.geometry.Image(cv2.cvtColor(color_img_resized, cv2.COLOR_BGR2RGB)),
            depth=open3d.geometry.Image(depth_img_resized),
            depth_scale=1.0/self._cfg.depth_unit,
            depth_trunc=self._cfg.depth_trunc,
            convert_rgb_to_intensity=False)

        cam_intrin = self._cam_intrin_resized.to_open3d_format()
        pcd = open3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            cam_intrin)

        if self._cfg.cloud_downsample_voxel_size > 0:
            pcd = open3d.geometry.voxel_down_sample(
                pcd, voxel_size=self._cfg.cloud_downsample_voxel_size)

        return pcd

    def _detect_plane_by_RANSAC(self, points):
        ''' Use RANSAC to detect plane from point pcd.
        The plane weights(parameters) w means:
            w[0] + w[1]*x + w[2]*y + w[3]*z = 0
        Arguments:
            points {np.ndarray}: (N, 3).
        Return:
            is_succeed {bool}: Is plane detected successfully.

        '''
        FAILURE_RETURN = False, None, None
        cfg = self._cfg.RANSAC_config

        print("\nRANSAC starts: Source points = {}".format(len(points)))
        ransac = RansacPlane()
        is_succeed, plane_weights, plane_pts_indices = ransac.fit(
            points,
            model=PlaneModel(),
            n_pts_fit_model=3,
            n_min_pts_inlier=cfg["min_points"],
            max_iter=cfg["iterations"],
            dist_thresh=cfg["dist_thresh"],
            is_print_res=cfg["is_print_res"],
        )

        if not is_succeed:
            print("RANSAC Failed.")
            return FAILURE_RETURN
        print("RANSAC succeeds: Points of plane = {}".format(
            plane_pts_indices.size))

        # Let the plane norm pointing to the camera.
        #       which means that the norm's z component should be negative.
        if plane_weights[-1] > 0:
            plane_weights *= -1
        return is_succeed, plane_weights, plane_pts_indices

    def _compute_planes_info(self, planes, color_img, depth_image):
        '''
        Arguments:
            planes {list of `class _Plane`}
        Returns:
            list_plane_params {list of `class PlaneParam`}
            planes_mask {image}: Mask of the detected planes.
                Each plane corresponds to one color region of the mask.
                This is a colored mask, with 3 channels and np.uint8 datatype.
            planes_img_viz {image}: image visualization of the detected planes.
                For each plane region, a color is superposed onto the origin image.
                Besides, an arrow is drawn to indicate the plane direction.
        '''

        shape, shape_resized = self._shape, self._shape_resized

        intrin_mat = self._cam_intrin.intrinsic_matrix(type="matrix")
        intrin_mat_resized = self._cam_intrin_resized.intrinsic_matrix(
            type="matrix")  # For the resized smaller image.
        resize_ratio = self._cfg.img_resize_ratio
        cfg_viz = self._cfg.visualization

        # -- Initialize the output variables.
        merged_masks = np.zeros(
            (shape_resized[0], shape_resized[1], 3), np.uint8)
        planes_img_viz = color_img.copy()
        list_plane_params = []

        # -- Process each plane.
        for i, plane in enumerate(planes):
            w, pts_3d = plane.weights, plane.points

            # Project 3d points to 2d by using
            # the resized intrinsics,
            # so the created mask is small, and costs less time.
            pts_2d_resized = world2pixel(
                pts_3d,
                T_cam_to_world=np.identity(4),
                camera_intrinsics=intrin_mat_resized).T

            # Calculate teh depth difference between either
            # side of the image    
            col,row = depth_image.shape[0:2] 
            mask_image = np.zeros((col,row,3), np.uint8)
            driver = Driver()

            mask_image = driver.steering(pts_2d_resized, depth_image, mask_image)

        return list_plane_params, mask_image

    def _get_ith_color(self, i):
        color_tuple_float = self._cmap(
            float(i)/MAX_OF_MAX_PLANE_NUMBERS)[:3]
        # color_tuple_uint8 = tuple(int(c*255) for c in color_tuple_float)
        color_array_uint8 = (
            np.array(color_tuple_float)*255).astype(np.uint8)
        return color_array_uint8


class Calling_PlaneDetector():

    def __init__(self):

        self.img_depth = None
        self.img_color = None
        self.bridge = CvBridge()
        self.rate = rospy.Rate(0.8)
        self.bride = CvBridge()
        self.planes = []

        self.pub_mask = rospy.Publisher("/mask_image", Image, queue_size = 2)
        self.pub_depth = rospy.Publisher("/depth_image", Image, queue_size = 2)

        rospy.Subscriber("/device_0/sensor_0/Depth_0/image/data", Image, self.image_callback_depth)
        rospy.Subscriber("/device_0/sensor_1/Color_0/image/data", Image, self.image_callback_color)

    def image_callback_depth(self, msg):
        #print("Got the depth Image")
        self.img_depth = self.bridge.imgmsg_to_cv2(msg,'16UC1')

    def image_callback_color(self, msg):
        #print("Got the Color Image")
        self.img_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def Run_Detection(self):

        while not rospy.is_shutdown():

            self.rate.sleep()
            #print("Working")
            #assert(type(self.img_depth) == np.uint8)
            config_file = "config/plane_detector_config.yaml"
            camera_info_file_path = "data/cam_params_realsense.json"
            detector = PlaneDetector(config_file, camera_info_file_path)
            #-----

            # -- Detect planes.
            # -- Process planes to obtain desired plane parameters.
        
            self.planes = detector.detect_planes(self.img_depth, color_img=None) # Add self.img_color if needed

            list_plane_params, mask_image = \
            detector._compute_planes_info(planes = self.planes, color_img = self.img_color, depth_image = self.img_depth)
            # -- Print result.
            for i, plane_param in enumerate(list_plane_params):
                plane_param.print_params(index=i+1)

            # -- Plot result.
            plt.subplot(1, 2, 1)
            plt.imshow(self.img_depth)
            plt.title("Depth Image.")
            plt.subplot(1, 2, 2)
            plt.imshow(mask_image)
            plt.title("Plane Detected and Masked into Halfs.")
            plt.show()

            #self.pub_mask.publish(mask_image)
            #self.pub_depth.publish(planes_img_viz)


class Driver():

    def steering(self, pixel_index, depth_image, mask_image): 

        tolerance = 0.05
        if type(pixel_index) == list:
            pixel_index = np.array(pixel_index)
        
        pixels = pixel_index.T
        left_depth = 0
        right_depth = 0
        right_count = 0
        left_count = 0
        mid = depth_image.shape[1]//2
        print("mid:",mid)
        #print(depth_image.shape)
        xs, ys = pixels[0].astype(np.int16), pixels[1].astype(np.int16) 
        print("Max Xs:", max(xs))
        print("Max Ys:", max(ys))
        for i in range(xs.shape[0]):
            mask_image[ys[i], xs[i]] = (255, 0, 0)
            dep = depth_image[ys[i],xs[i]]/1000
            if xs[i] > mid: 
                #print("xs:",xs[i])
                right_depth += dep
                right_count += 1
                mask_image[ys[i], xs[i]] = (255, 0, 0)
            else: 
                left_depth += dep
                left_count += 1
                mask_image[ys[i], xs[i]] = (0, 0, 255)
        
        right_depth /= right_count # mean depth
        left_depth /= left_count

        #print("left:",round(left_depth,3))
        #print("right:",round(right_depth,3))
        difference = round(left_depth - right_depth, 3)
        print("Difference: ", difference)
        if abs(difference) < tolerance:
            print("Perfectly Oriented")
        else:
            print("Steer_Left") if difference < 0 else print("Steer_Right")
        return mask_image


if __name__ == '__main__':

    rospy.init_node('image_listener', anonymous = True)
    my_plane = Calling_PlaneDetector()
    my_plane.Run_Detection()
