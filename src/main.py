import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField, Image
import std_msgs
import timeit
from matplotlib import pyplot as plt
import math
import struct
import copy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import cv2

# python3 main.py  --eps 0.01 --min_points 10 --ransac_n_cluster 40  --ransac_n_cluster_pallet_detection 40 --eps_refine 0.03 --min_points_refine 10 --max_dis 4



# L515 (1m  - 2m)
# python3 main.py  --eps 0.01 --min_points 10 --ransac_n_cluster 40  --ransac_n_cluster_pallet_detection 40 

# L515 (2m  - 3m)
# python3 main.py  --eps 0.05 --min_points 10 --ransac_n_cluster 40  --ransac_n_cluster_pallet_detection 40 --eps_refine 0.025 --min_points_refine 10

# L515 (3m  - 4m)
# python3 main.py  --eps 0.05 --min_points 10 --ransac_n_cluster 40  --ransac_n_cluster_pallet_detection 40 --eps_refine 0.03 --min_points_refine 10 --max_dis 4



# L515 (up 1m: 1m-2.5m)
# python3 main.py  --eps 0.01 --min_points 10 --ransac_n_cluster 40  --pallet_height 1 --ransac_n_cluster_pallet_detection 25 --eps_refine 0.025

# L515 (up 1m: 2.5m-4m)
# python3 main.py  --eps 0.05 --min_points 10 --ransac_n_cluster 40  --pallet_height 1 --ransac_n_cluster_pallet_detection 25 --eps_refine 0.025 --max_dis 4


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh



def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])



def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    
    # # parameters of pallet:
    parser.add_argument('--pallet_height', type=float, default=0.2, help='max height of pallet')
    parser.add_argument('--sensor_height', type=float, default=0.5, help='prox height of sensor')
    parser.add_argument('--max_dis', type=float, default=3.5, help='max dis of range')
    
    # # parameters for point_cloud_preprocessing: 
    parser.add_argument('--voxel_size', type=float, default=0.005, help='voxel_size for voxel filter')
    parser.add_argument('--distance_threshold', type=float, default=0.01, help='distance_threshold for segment_plane')
    parser.add_argument('--distance_threshold_RS_M1', type=float, default=0.03, help='distance_threshold for segment_plane for RS_M1')
    
    parser.add_argument('--ransac_n_ground', type=int, default=100, help='ransac_n for segment_plane of ground')
    parser.add_argument('--num_iterations', type=int, default=100, help='num_iterations for segment_plane')
    parser.add_argument('--gournd_theta_threshold', type=float, default=10, help='theta threshold to be considered as ground')
    
    # # parameters for geometric_pallet_segmentation: 
    parser.add_argument('--nb_points', type=int, default=50, help='Number of points within the radius')
    # default for D455
    # no need for L515
    # 25 for RS_M1
    parser.add_argument('--radius', type=float, default=0.1, help='Radius of the sphere')
    # default for D455
    # no need for L515
    # 0.2 for RS_M1
    parser.add_argument('--ransac_n_cluster', type=int, default=50, help='ransac_n for segment_plane of cluster')    
    parser.add_argument('--ransac_n_cluster_pallet_detection', type=int, default=25, help='ransac_n for segment_plane of cluster')    
    parser.add_argument('--eps', type=float, default=0.01, help='the distance to neighbours in a cluster')
    # 0.01 for D455
    # 0.05 for L515 and ransac_n_cluster 40
    # 0.03 for RS_M1 and ransac_n_cluster 40
    parser.add_argument('--min_points', type=int, default=10, help='the minimun number of points required to form a cluster')
    parser.add_argument('--eps_threshold', type=float, default=8, help='epsilo threshold to be considered as vertical to ground')
    # cluster_dbscan and requires two parameters. eps defines the distance to neighbours in a cluster and min_points defines the minimun number of points required to form a cluster
    
    # # parameters for geometric_pallet_segmentation: voxel_size_refine
    # parser.add_argument('--voxel_size_refine', type=float, default=0.01, help='voxel_size for voxel filter')
    # parser.add_argument('--every_k_points', type=int, default=2, help='downsample the point cloud by collecting every n-th points')
    parser.add_argument('--voxel_size_refine', type=float, default=0.01, help='voxel_size for voxel filter')
    parser.add_argument('--every_k_points', type=int, default=2, help='downsample the point cloud by collecting every n-th points')
    
    parser.add_argument('--eps_refine', type=float, default=0.03, help='the distance to neighbours in a cluster')
    parser.add_argument('--min_points_refine', type=int, default=10, help='the minimun number of points required to form a cluster')
    # 0.02
    
    parser.add_argument('--centroid_dis_min', type=float, default=0.3, help='the distance threshold for centroid')
    parser.add_argument('--centroid_dis_max', type=float, default=0.5, help='the distance threshold for centroid')
    
    parser.add_argument('--centroid_height_threshold', type=float, default=0.05, help='the threshold distance for centroid to be considered as same height')
    
    
    args = parser.parse_args()
    return args



class Segmentation(object):
    def __init__(self, args):
        # # parameters of pallet:
        self.opt = args
        
        self.segmation = False
        self.ground_plane_model = None
        self.V2C = np.array([0, -1, 0,
                             0, 0, -1,
                             1, 0, 0,]).reshape((3,3))
        self.C2V = np.linalg.inv(self.V2C)
        
        rospy.init_node('PalletDetection')
        self.bridge = CvBridge()
        # self.rgb_img = None
        self.frame_number = 0
        self.frame_success = 0
        self.img_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.lidar_sub = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)
        # self.lidar_sub = rospy.Subscriber('/camera/depth/color/points', data_class=PointCloud2, queue_size=1, callback=self.callback)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.lidar_sub, self.img_sub], 1, 0.1)
        self.ts.registerCallback(self.callback)

        self.pointcloud_repub = rospy.Publisher('/camera/depth/color/points_repub', PointCloud2, queue_size=1)
        self.pointcloud_pub = rospy.Publisher('/camera/depth/color/points_voxel', PointCloud2, queue_size=1)
        self.img_pub = rospy.Publisher('/camera/color/image_show', Image, queue_size=1)
    
    def callback(self, lidar_msg, img_msg):
        rospy.loginfo('Receiving MSG !')
        self.frame_number += 1
        start = timeit.default_timer()
        self.header = std_msgs.msg.Header()
        self.header.stamp = lidar_msg.header.stamp
        self.header.frame_id = lidar_msg.header.frame_id
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('rgba', 12, PointField.UINT32, 1),
                        ]

        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(img_msg, 'passthrough').astype(np.uint8).copy() 
            # rgb_img = self.bridge.compressed_imgmsg_to_cv2(img_msg, 'passthrough')
        except CvBridgeError as e:
            print(e)
        
        # receiving point cloud
        pointcloud_list = []
        for point in pcl2.read_points(lidar_msg, skip_nans=True, field_names=("x", "y", "z")):
            pointcloud_list.append(point[2])
            pointcloud_list.append(-point[0])
            pointcloud_list.append(-point[1])
        
        pointcloud_array = np.array(pointcloud_list).reshape((-1, 3))
        pointcloud_array = pointcloud_array[np.where(pointcloud_array[:, 0] < self.opt.max_dis)]

        # preprocessing point cloud
        pcd = self.point_cloud_preprocessing(pointcloud_array)
        # if not Succesfully Segment the Ground return
        if self.segmation == False:
            rospy.loginfo('********* Sorry No Pallet Detected *********')
            return
        
        # # geometric_pallet_segmentation
        is_segmet, pointcloud_array = self.geometric_pallet_segmentation(pcd)
        if is_segmet == False:
            rospy.loginfo('********* Sorry No Pallet Detected *********')
            return
        
        pointcloud_repub_msg = pcl2.create_cloud_xyz32(self.header, pointcloud_array)
        self.pointcloud_repub.publish(pointcloud_repub_msg)
        
        # return
        
        # # # geometric_pallet_detection
        is_detected, pallet_pcd = self.geometric_pallet_detection(pointcloud_array)
        if is_detected == False:
            rospy.loginfo('********* Sorry No Pallet Detected *********')
            return
        
        pointcloud_repub_msg = pcl2.create_cloud_xyz32(self.header, np.asarray(pallet_pcd.points))
        
        self.pointcloud_pub.publish(pointcloud_repub_msg)
        success_rate = self.frame_success / self.frame_number
        rospy.loginfo('Successful Rate: %2f; (%d / %d )', success_rate, self.frame_success, self.frame_number)
        
        self.lookup_tabel(np.asarray(pallet_pcd.points))
        
        # pointcloud_msg = pcl2.create_cloud_xyz32(header, pointcloud_array)
        # self.pointcloud_pub.publish(pointcloud_msg)
        
        stop = timeit.default_timer()
        print('det_time: ', stop - start)


# L515 (1m  - 2m)
# python3 main.py  --eps 0.01 --min_points 10 --ransac_n_cluster 40  --ransac_n_cluster_pallet_detection 40 

# L515 (2m  - 3m)
# python3 main.py  --eps 0.05 --min_points 10 --ransac_n_cluster 40  --ransac_n_cluster_pallet_detection 40 --eps_refine 0.025 --min_points_refine 10

# L515 (3m  - 4m)
# python3 main.py  --eps 0.05 --min_points 10 --ransac_n_cluster 40  --ransac_n_cluster_pallet_detection 40 --eps_refine 0.03 --min_points_refine 10 --max_dis 4

     
    def lookup_tabel(self, pointcloud_array):
        median_x = np.median(pointcloud_array[:, 0])
        # print('median distance: ', median_x)
        
        if median_x >= 3:
            # self.opt.eps = 0.05
            # self.opt.eps_refine = 0.03
            # self.opt.min_points_refine = 10
            self.opt.max_dis = 4
        elif 2<= median_x <3:
            # self.opt.eps = 0.05
            # self.opt.eps_refine = 0.025
            # self.opt.min_points_refine = 10
            self.opt.max_dis = 3.5
        elif 1.5 <= median_x <2:
            # self.opt.eps = 0.01
            # self.opt.eps_refine = 0.03
            # self.opt.min_points_refine = 10
            self.opt.max_dis = 3
        elif median_x <1.5:
            self.opt.max_dis = 2
            

    def point_cloud_preprocessing(self, pointcloud_array):
        pointcloud_array = pointcloud_array[np.where(pointcloud_array[:, -1] < self.opt.sensor_height)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud_array)
        
        # # first Voxel filter to downsample
        point_cloud_np = pcd.voxel_down_sample(self.opt.voxel_size)
    
        # # plane segmentation to find ground
        plane_model, inliers =  point_cloud_np.segment_plane(distance_threshold=self.opt.distance_threshold, 
                                                             ransac_n=self.opt.ransac_n_ground, 
                                                             num_iterations=self.opt.num_iterations)
        [a, b, c, d] = self.ground_plane_model = plane_model
        self.abc_sqrt = np.sqrt(a**2+b**2+c**2)
        
        rospy.loginfo(f"Ground plane equation:({a:.2f}x)+({b:.2f}y)+({c:.2f}z)+({d:.2f})=0")
        
        theta_z = np.arccos( c / self.abc_sqrt )

        if theta_z <= self.opt.gournd_theta_threshold * np.pi / 180:
            self.segmation = True
            rospy.loginfo('Succesfully Segment the Ground')
            outlier_cloud = point_cloud_np.select_by_index(inliers, invert=True)
            outlier_cloud_np = np.asarray(outlier_cloud.points).reshape((-1, 3))
            distance = abs( outlier_cloud_np[:, 0] *a + outlier_cloud_np[:, 1] * b + outlier_cloud_np[:, 2] * c + d) /self.abc_sqrt
            outlier_cloud_np = outlier_cloud_np[np.where(distance < 1.5 * self.opt.pallet_height)]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(outlier_cloud_np)
            return pcd            
        else:
            rospy.loginfo('Failed Segment the Ground')
            self.segmation = False
            return pcd
        
        # [a, b, c, d] = self.ground_plane_model =  [0, 0, 1, self.opt.sensor_height]
        # self.abc_sqrt = np.sqrt(a**2+b**2+c**2)
        # self.segmation = True
        # return pcd
            

    def geometric_pallet_segmentation(self, voxel_down_pcd):
        is_segment = False
        # 提取所有垂直地面的cluster
        pcd, ind = voxel_down_pcd.remove_radius_outlier(nb_points=self.opt.nb_points, radius=self.opt.radius)
        
        # # DB_Scan for clustering
        labels = np.array(pcd.cluster_dbscan(eps=self.opt.eps, min_points=self.opt.min_points)) 
        # #  a array with shape (n,): n is the size of point cloud array

        max_label = labels.max()
        rospy.loginfo(f"point cloud has {max_label + 1} clusters")
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([pcd])
        
        # # check each cluster whether they are vertical to the ground
        point_cloud_array = np.empty((0,3))
        for label_num in range(0, max_label + 1):
    
            index = np.where(labels == label_num)[0].tolist()
            
            if len(index)  < self.opt.ransac_n_cluster:
                continue
            segment_pcd = pcd.select_by_index(index, invert=False)
            
            plane_model, inliers =  segment_pcd.segment_plane(distance_threshold=self.opt.distance_threshold, 
                                                            ransac_n=self.opt.ransac_n_cluster_pallet_detection, 
                                                            num_iterations=self.opt.num_iterations)
            [a, b, c, d] = plane_model
            
            # # theta with respect to the ground plane model    
            theta_z = np.arccos( (a*self.ground_plane_model[0] + b*self.ground_plane_model[1] + c*self.ground_plane_model[2]) 
                                / np.sqrt(a**2+b**2+c**2) 
                                / np.sqrt(self.ground_plane_model[0]**2+self.ground_plane_model[1]**2+self.ground_plane_model[2]**2))
            
            # # check if is vertical to the ground
            if (90 + self.opt.eps_threshold) * np.pi / 180 >= theta_z >= (90 - self.opt.eps_threshold) * np.pi / 180:
                is_segment = True
                point_cloud_array = np.concatenate((point_cloud_array, np.asarray(segment_pcd.select_by_index(inliers).points)), axis=0) 
                     
        return is_segment, point_cloud_array
    
    def geometric_pallet_detection(self, point_cloud_array):
        # # DB_Scan for clustering
        is_detected, pallet_pcd = False, None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_array)
        # labels = np.array(pcd.cluster_dbscan(eps=self.opt.eps, min_points=self.opt.min_points)) 
        # max_label = labels.max()
        # max_idx = []
        # for label_num in range(0, max_label + 1):
        #     index = np.where(labels == label_num)[0].tolist()
        #     if len(index) > len(max_idx):
        #         max_idx = index
        # segment_pcd = pcd.select_by_index(index, invert=False)
        plane_model, inliers =  pcd.segment_plane(distance_threshold=self.opt.distance_threshold, 
                                                            ransac_n=self.opt.ransac_n_cluster_pallet_detection, 
                                                            # ransac_n=int(len(index) / 1.5), 
                                                            num_iterations=self.opt.num_iterations)
        [a, b, c, d] = plane_model
        abc_sqrt = np.sqrt(a**2+b**2+c**2)
        theta_a = np.arccos(a/ abc_sqrt)  * 180 / np.pi
        theta_b = np.arccos(b/ abc_sqrt)  * 180 / np.pi
        theta_c = np.arccos(c/ abc_sqrt)  * 180 / np.pi
        theta_z = np.arccos( (a*self.ground_plane_model[0] + b*self.ground_plane_model[1] + c*self.ground_plane_model[2]) 
                                / np.sqrt(a**2+b**2+c**2) 
                                / np.sqrt(self.ground_plane_model[0]**2+self.ground_plane_model[1]**2+self.ground_plane_model[2]**2))
        # rospy.loginfo(f"Segmentation Plane equation:({a:.2f}x)+({b:.2f}y)+({c:.2f}z)+({d:.2f})=0")
        # dis = abs(d / abc_sqrt)
        # dis = np.sqrt(np.median(point_cloud_array[:,0])**2+np.median(point_cloud_array[:,1])**2+np.median(point_cloud_array[:,2])**2)
        disX = np.median(point_cloud_array[:,0])
        disY = np.median(point_cloud_array[:,1])
        disZ = np.median(point_cloud_array[:,2])
        if (90 + self.opt.eps_threshold) * np.pi / 180 >= theta_z >= (90 - self.opt.eps_threshold) * np.pi / 180:
            rospy.loginfo('************* Pallet Detected *************')
            rospy.loginfo(f" Distance: {disX:.2f} (meter), {disY:.2f} (meter), {disZ:.2f} (meter) ")
            # rospy.loginfo(f" Distance: {dis:.2f} (meter) ")
            self.frame_success += 1
            rospy.loginfo(f" Angle: {theta_a:.2f} (degree), {theta_b:.2f} (degree), {theta_c:.2f} (degree) ")
            is_detected = True
            pallet_pcd = pcd.select_by_index(inliers)
            
            # text1 = f"Distance:{dis:.2f}(meter) " 
            text1 = f"Distance:{disX:.2f}(meter), {disY:.2f}(meter), {disZ:.2f}(meter) " 
            text2 = f"Angle:{theta_a:.2f}(degree), {theta_b:.2f}(degree), {theta_c:.2f}(degree)"
            h, w, _ = self.rgb_img.shape
            cv2.putText(self.rgb_img, text1,(int(0.1*w), int(0.09 * h)),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(self.rgb_img, text2,(int(0.1*w), int(0.2 * h)),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(self.rgb_img, 'passthrough'))
        return is_detected, pallet_pcd
        
            
        
        
if __name__ == "__main__":
    import argparse
    args = parse_config()
    segment = Segmentation(args)
    rospy.spin()