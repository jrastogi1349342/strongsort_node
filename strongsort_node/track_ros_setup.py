#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.clock import Clock
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
import message_filters
from pathlib import Path

from strongsort_node.track import StrongSortPublisher

FILE = Path(__file__).resolve()
# print("FILE:", FILE)
# print("FILE Parents:", FILE.parents[0])
ROOT = FILE.parents[0]  # yolov7 strongsort root directory
WEIGHTS = ROOT / 'weights'


from yolov7.utils.general import check_requirements


class StrongSortSetup(Node):
    def __init__(self):
        super().__init__('strongsort_node_setup')
        check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_id', 0), # change this in launch file for each robot
                ('max_num_robots', 2), 
                ('camera_type', 'gray'), # gray or color
                ('yolo_weights', 'yolov7.pt'),
                ('strong_sort_weights', 'osnet_x0_25_msmt17.pt'),
                ('config_strongsort', 'src/strongsort_node/strong_sort/configs/strong_sort.yaml'), # may be changed
                ('video_topic', "rm_vlc_leftfront/image"),
                ('name_space', "hl2"), 
                ('img_size', [640, 640]),
                ('conf_thres', 0.5),
                ('iou_thres', 0.1),
                ('max_det', 1000),
                ('device', '0'),
                ('show_video', False), # changed from false
                ('save_text', False),
                ('save_conf', False),
                ('save_crop', False),
                ('save_vid', False),
                ('no_save', False),
                ('classes', [-1]),
                ('agnostic_nms', False),
                ('augment', False),
                ('visualize', False),
                ('update', False),
                ('project', 'runs/track'),
                ('name', "exp"),
                ('exist_ok', False),
                ('line_thickness', 3),
                ('hide_labels', False),
                ('hide_conf', False),
                ('hide_class', False),
                ('half', False),
                ('dnn', False)
            ]
        )
        
        self.strongsort_params = {}
        
        self.strongsort_params['robot_id'] = self.get_parameter('robot_id').value
        self.strongsort_params['max_num_robots'] = self.get_parameter('max_num_robots').value
        
        self.strongsort_params['video_topic'] = self.get_parameter('video_topic').value
        self.strongsort_params['name_space'] = self.get_parameter('name_space').value

        self.strongsort_params['camera_type'] = self.get_parameter('camera_type').value
        self.strongsort_params['yolo_weights'] = self.get_parameter('yolo_weights').value
        self.strongsort_params['strong_sort_weights'] = self.get_parameter('strong_sort_weights').value
        self.strongsort_params['config_strongsort'] = self.get_parameter('config_strongsort').value
        self.strongsort_params['img_size'] = self.get_parameter('img_size').value
        self.strongsort_params['conf_thres'] = self.get_parameter('conf_thres').value
        self.strongsort_params['iou_thres'] = self.get_parameter('iou_thres').value
        self.strongsort_params['max_det'] = self.get_parameter('max_det').value
        self.strongsort_params['device'] = self.get_parameter('device').value
        self.strongsort_params['show_video'] = self.get_parameter('show_video').value
        self.strongsort_params['save_text'] = self.get_parameter('save_text').value
        self.strongsort_params['save_conf'] = self.get_parameter('save_conf').value
        self.strongsort_params['save_crop'] = self.get_parameter('save_crop').value
        self.strongsort_params['save_vid'] = self.get_parameter('save_vid').value
        self.strongsort_params['no_save'] = self.get_parameter('no_save').value
        self.strongsort_params['classes'] = self.get_parameter('classes').value
        self.strongsort_params['agnostic_nms'] = self.get_parameter('agnostic_nms').value
        self.strongsort_params['augment'] = self.get_parameter('augment').value
        self.strongsort_params['visualize'] = self.get_parameter('visualize').value
        self.strongsort_params['update'] = self.get_parameter('update').value
        self.strongsort_params['project'] = self.get_parameter('project').value
        self.strongsort_params['name'] = self.get_parameter('name').value
        self.strongsort_params['exist_ok'] = self.get_parameter('exist_ok').value
        self.strongsort_params['line_thickness'] = self.get_parameter('line_thickness').value
        self.strongsort_params['hide_labels'] = self.get_parameter('hide_labels').value
        self.strongsort_params['hide_conf'] = self.get_parameter('hide_conf').value
        self.strongsort_params['hide_class'] = self.get_parameter('hide_class').value
        self.strongsort_params['half'] = self.get_parameter('half').value
        self.strongsort_params['dnn'] = self.get_parameter('dnn').value        
        

        # To align with how original code works
        if self.strongsort_params['classes'] == [-1]: 
            self.strongsort_params['classes'] = None
            
        
        self.mot_publishers = StrongSortPublisher(self.strongsort_params, self)


        # print(f"/{name_space}{video_source}\t/{name_space}/stereo/depth\t/{name_space}/stereo/left/camera_info\t/{name_space}/odom")
        
        # TODO maybe add namespace here, but works 
        # Gets camera info, runs Yolo and StrongSORT, populates self.best_cosplace_results_dict
        video_sub_sync = message_filters.Subscriber(self, 
                                               Image, 
                                                f"/{self.strongsort_params['name_space']}{self.strongsort_params['video_topic']}", 
                                                qos_profile=rclpy.qos.QoSProfile(
                                                    reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                                    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                                    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                                                    depth=5, 
                                                    )
                                                )
        
        # 34.5 cm distance to object, 46.5 cm width of object --> 34 degree offset from center for the two centered cameras
        # Should use this angle to get angle/distance from center of HL2 but since distance between cameras is 0.0986 m, which 
        # is negligible
        # 17.25 in dist to wall, 29.65 in wall height from 0 to 640 px
        depth_sub_sync = message_filters.Subscriber(
            self, 
            Image, 
            f"/{self.strongsort_params['name_space']}/stereo/depth", 
            qos_profile=rclpy.qos.QoSProfile(
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                depth=5, 
            )
        )
        
        cam_info_sync = message_filters.Subscriber(
            self, 
            CameraInfo, 
            f"/{self.strongsort_params['name_space']}/stereo/left/camera_info", 
            qos_profile=rclpy.qos.QoSProfile(
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                depth=5, 
            )
        )
        
        odom_sync = message_filters.Subscriber(
            self, 
            Odometry, 
            f"/{self.strongsort_params['name_space']}/odom", 
            qos_profile=rclpy.qos.QoSProfile(
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                depth=5, 
            )
        )
         
        #  Colocalization will eventually go here
                
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [video_sub_sync, depth_sub_sync, cam_info_sync, odom_sync], 
            queue_size=10, 
            slop=0.5, 
            allow_headerless=True)
        self.ts.registerCallback(self.mot_publishers.video_callback)
        
        
        # left_img_sync = message_filters.Subscriber(
        #     self, 
        #     Image, 
        #     "/hl2/stereo/left/image_rect", 
        #     qos_profile=rclpy.qos.QoSProfile(
        #         reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
        #         history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        #         durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        #         depth=5, 
        #     )
        # )
        
        # right_img_sync = message_filters.Subscriber(
        #     self, 
        #     Image, 
        #     "/hl2/stereo/right/image_rect", 
        #     qos_profile=rclpy.qos.QoSProfile(
        #         reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
        #         history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        #         durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        #         depth=5, 
        #     )
        # )
        
        # self.ts_scratch = message_filters.ApproximateTimeSynchronizer([left_img_sync, right_img_sync], queue_size=10, slop=0.5, allow_headerless=True)
        # self.ts_scratch.registerCallback(self.mot_publishers.depth_scratch_sync_callback)
        # self.i = 1
        
        
if __name__ == '__main__':
    rclpy.init(args=None)
    setup = StrongSortSetup()
    
    setup.get_logger().info('StrongSORT initialization done.')
    rclpy.spin(setup)
    
    setup.destroy_node()
    rclpy.shutdown()
