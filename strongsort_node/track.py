#!/usr/bin/env python3
from rclpy.qos import qos_profile_sensor_data
from rclpy.clock import Clock
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped
from strongsort_msgs.msg import MOTGlobalDescriptor, MOTGlobalDescriptors, UnifiedObjectIDs
from tf2_ros import TransformBroadcaster
import math
from scipy.spatial.transform import Rotation as R

qos_pf = qos_profile_sensor_data


import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image as pil_image


from strongsort_node.cosplace import CosPlace
# TODO use this after integration into Nathan's repo
# from cslam.cslam.vpr.cosplace import CosPlace 

FILE = Path(__file__).resolve()
# print("FILE:", FILE)
# print("FILE Parents:", FILE.parents[0])
ROOT = FILE.parents[0]  # yolov7 strongsort root directory
WEIGHTS = ROOT / 'weights'

# print("ROOT:", ROOT)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.general import (check_img_size, non_max_suppression, cv2,xyxy2xywh, xywh2xyxy, 
                                  clip_coords, increment_path)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

class StrongSortPublisher(object): 
    def __init__(self, params, node):
        self.params = params
        self.node = node
        
        # self.cosplace_desc = CosPlace(self.params, self.node)
        self.cosplace_desc = CosPlace()
        
        # Dictionary mapping object ID to MOTGlobalDescriptor object
        self.results_dict = {}
        
        self.last_sent_detections_time = -1.0
        # Sends MOTGlobalDescriptors object every 0.1 seconds to all new detections found in the
        # last 0.1 seconds 
        # NOTE: will not send to agents out of range 
        self.last_seen_det_clock = self.node.create_timer(0.1, self.last_seen_callback, clock=Clock())
                
        # Sends all descriptors with non-unified IDs to the other robot in callback of self.latest_class_sub
        # subscriber (i.e. new info found)
        self.mot_pub = self.node.create_publisher(
            MOTGlobalDescriptors, 
            f"/mot/descriptors", 
            100)
        
        self.tf_broadcaster = TransformBroadcaster(self.node)
        
        self.unified_id_mapping = {}
        
        self.unified_ids_sub = self.node.create_subscription(
            UnifiedObjectIDs, 
            '/mot/unified', 
            self.unified_info_callback, 
            10
        )

        self.orig_init()
        
    def orig_init(self): 
        self.bridge = CvBridge()
        
        exp_name = self.params['yolo_weights']
        exp_name = self.params['name'] if self.params['name'] else exp_name + "_" + self.params['strong_sort_weights'].stem
        self.save_dir = increment_path(Path(self.params['project']) / exp_name, exist_ok=self.params['exist_ok'])  # increment run
        self.save_dir = Path(self.save_dir)
        (self.save_dir / 'tracks' if self.params['save_text'] else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(self.params['device'])
        
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.model = attempt_load(Path(self.params['yolo_weights']), map_location=self.device)  # load FP32 model
        self.names, = self.model.names,
        stride = self.model.stride.max().cpu().numpy()  # model stride
        self.img_size = check_img_size(self.params['img_size'][0], s=stride)  # check image size

        # Dataloader
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # initialize StrongSORT
        self.cfg = get_config()
        self.cfg.merge_from_file(self.params['config_strongsort'])

        # Create as many strong sort instances as there are video sources
        self.strongsort_list = []
        self.strongsort_list.append(
            StrongSORT(
                self.params['strong_sort_weights'],
                self.device,
                self.params['half'],
                max_dist=self.cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=self.cfg.STRONGSORT.MAX_AGE,
                n_init=self.cfg.STRONGSORT.N_INIT,
                nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=self.cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=self.cfg.STRONGSORT.EMA_ALPHA,
            )
        )
        
        self.strongsort_list[0].model.warmup()
        self.outputs = [None]
        
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run tracking
        self.dt, self.seen = [0.0, 0.0, 0.0, 0.0], 0
        self.curr_frames, self.prev_frames = [None], [None] 
        
    # method from [https://github.com/WongKinYiu/yolov7/blob/u5/utils/plots.py#L474] for availabling '--save-crop' argument
    def save_one_box(self, xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
        # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
        xyxy = torch.tensor(xyxy).view(-1, 4)
        b = xyxy2xywh(xyxy)  # boxes
        if square:
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
        b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
        xyxy = xywh2xyxy(b).long()
        clip_coords(xyxy, im.shape)
        crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
        if save:
            file.parent.mkdir(parents=True, exist_ok=True)  # make directory
            f = str(Path(increment_path(file)).with_suffix('.jpg'))
            # cv2.imwrite(f, crop)  # https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
            pil_image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(f, quality=95, subsampling=0)
        return crop
    
    # Range of approximately 2.5 m, or 8 ft, with stereoParams.maxDisparity = 48
    def get_depth(self, depth_msg, left_cam_info_msg): 
        img = self.bridge.imgmsg_to_cv2(depth_msg, "mono8")
        
        focal_len = 4.87 # mm +/- 5% (of one grayscale camera --> not sure if this is needed)
        
        
        baseline_dist = 0.0986 # meters (distance between the two stereo cameras)
        focal_len_left_cam = (left_cam_info_msg.k[0] + left_cam_info_msg.k[4]) / 2
        
        # 8.85 is approximate scaler for when stereoParams.maxDisparity = 48 --> distance in meters
        depth = baseline_dist * focal_len_left_cam * 8.85 / img
        
        return depth
    
    # Take median of depth, and remove a bit of left and right sides because both cameras can't see 
    # that part so depth is completely inaccurate
    def depth_median(self, depth, tlw, tlh, brw, brh):
        tlh = 30 if tlh < 30 else tlh
        brh = 450 if brh > 450 else brh
        
        return np.median(depth[tlw:brw, tlh:brh])
    
    def angle_to_object(self, odom_quat, pitch_degrees, yaw_degrees): 
        pitch_rad = math.pi * pitch_degrees / 180
        pitch_mtx = np.array([[math.cos(pitch_rad), 0, math.sin(pitch_rad)], 
                             [0, 1, 0], 
                             [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]])
    
        yaw_rad = math.pi * yaw_degrees / 180
        yaw_mtx = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0], 
                             [math.sin(yaw_rad), math.cos(yaw_rad), 0], 
                             [0, 0, 1]])
        
        obj_rot_mtx = R.from_matrix(yaw_mtx * pitch_mtx)
        
        odom_rot_mtx = R.from_quat([odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w])
        
        # print(f"Obj rotation matrix: {obj_rot_mtx.as_euler('xyz', degrees=True)}\tOdom rotation matrix: {odom_rot_mtx.as_euler('xyz', degrees=True)}")
        
        combined_rot_mtx = obj_rot_mtx * odom_rot_mtx
        return combined_rot_mtx.as_euler('xyz', degrees=True)
        
    # video_sub_sync, depth_sub_sync, cam_info_sync, odom_sync
    @torch.no_grad()
    def video_callback(self, img_msg, depth_msg, cam_info, odom_msg): 
        s = ''
        t1 = time_synchronized()

        if self.params['camera_type'] == 'gray': 
            img = self.bridge.imgmsg_to_cv2(img_msg, "mono8") # (640, 480) from HL2 

            norm = img / 255 # Normalize
            norm_expanded = np.expand_dims(norm, axis = 0) # (1, 640, 480)
            np_stack = np.repeat(norm_expanded, 3, axis = 0) # (3, 640, 480)
            np_exp = np.expand_dims(np_stack, axis = 0) # (1, 3, 640, 480)
            im = torch.from_numpy(np_exp).to(0) # torch.Size([1, 3, 480, 640])

        elif self.camera_type == 'color': # TODO test with Vision 60 - (960 x 540)?
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8") # presumably (640, 480, 3) 
            norm = img / 255 # Normalize
            norm_transpose = np.transpose(norm, (2, 0, 1))
            norm_expanded = np.expand_dims(norm_transpose, axis = 0) # (1, 3, 640, 480)
            im = torch.from_numpy(norm_expanded).to(0)
            
        else: 
            return 
        
        im = im.half() if self.params['half'] else im.float()  # uint8 to fp16/32

        t2 = time_synchronized()
        self.dt[0] += t2 - t1

        pred = self.model(im)
        t3 = time_synchronized()
        self.dt[1] += t3 - t2
        
        # Inference
        self.visualize = increment_path(self.save_dir / Path(self.path[0]).stem, mkdir=True) if self.params['visualize'] else False

        # Apply NMS
        # self.conf_thres, self.iou_thres, self.classes
        pred = non_max_suppression(
            pred[0], 
            self.params['conf_thres'], 
            self.params['iou_thres'], 
            self.params['classes'], 
            self.params['agnostic_nms'])
        self.dt[2] += time_synchronized() - t3
        
        # disparity = self.bridge.imgmsg_to_cv2(orig_msg, "bgr8") 
        depth = self.get_depth(depth_msg, cam_info)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            self.seen += 1

            if self.params['camera_type'] == 'gray': 
                copy = img.copy()
                copy_exp = np.expand_dims(copy, axis = 2)
                im0 = np.repeat(copy_exp, 3, axis = 2)
            elif self.camera_type == 'color':
                im0 = img.copy()

            p = Path("path")  # to Path
            s += f'{i}: '
            txt_file_name = p.name
            save_path = str(self.save_dir / p.name) + str(i)  # im.jpg, vid.mp4, ...

            self.curr_frames[i] = im0

            txt_path = str(self.save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if self.params['save_crop'] else im0  # for save_crop

            if self.cfg.STRONGSORT.ECC:  # camera motion compensation
                self.strongsort_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])

            if det is not None and len(det):
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                
                # pass detections to strongsort
                t4 = time_synchronized()
                self.outputs[i] = self.strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_synchronized()
                self.dt[3] += t5 - t4

                # draw boxes for visualization
                if len(self.outputs[i]) > 0:
                    for _, (output, conf_tensor) in enumerate(zip(self.outputs[i], confs)):
                        conf = conf_tensor.item()
    
                        bboxes = output[0:4]
                        id = int(output[4])
                        cls = int(output[5])
                        
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        
                        tlw = int(output[1]) if int(output[1]) < im0.shape[0] else im0.shape[0] - 1
                        tlh = int(output[0]) if int(output[0]) < im0.shape[1] else im0.shape[1] - 1
                        brw = int(output[3]) if int(output[3]) < im0.shape[0] else im0.shape[0] - 1
                        brh = int(output[2]) if int(output[2]) < im0.shape[1] else im0.shape[1] - 1
                        
                        # if id > self.latest_det_dict[self.params['robot_id']]: 
                        #     self.latest_det_dict[self.params['robot_id']]
                              
                        # Negate pitch
                        pitch_to_obj = 67 * (320 - ((tlw + brw) / 2)) / 320
                        yaw_to_obj = 34 * (((tlh + brh) / 2) - 240) / 240
                        median_depth_val = self.depth_median(depth, tlw, tlh, brw, brh)
                        # print(f"Pitch angle: {pitch_to_obj}\tYaw angle: {yaw_to_obj}\tDepth: {median_depth_val}")
                        
                        # TODO test this
                        # overall_angles_to_obj = self.angle_to_object(odom_msg.pose.pose.orientation, pitch_to_obj, yaw_to_obj)

                        # print(f"Angles: {overall_angles_to_obj}")
                        # disparity = cv2.rectangle(disparity, (tlh, tlw), (brh, brw), (255, 255, 255), 2)

                        # ID 0 is on the left of ID 1
                        # TODO test this
                        colocalization = TransformStamped(
                            header=Header(
                                stamp=self.node.get_clock().now().to_msg(), 
                                frame_id=img_msg.header.frame_id # this doesn't change
                            ))
                        quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                        quat_scipy = R.from_quat([0.0, 0.0, 0.0, 1.0])
                        quat_scipy_inv = quat_scipy.inv().as_quat()
                        quat_inv = Quaternion(x=quat_scipy_inv[0], y=quat_scipy_inv[1], z=quat_scipy_inv[2], w=quat_scipy_inv[3])
                        if self.params['robot_id'] == 0: 
                            colocalization.child_frame_id = "B_odom"
                            colocalization.transform = Transform(
                                translation=Vector3(x=1.0, y=0.0, z=0.0), # meters?
                                rotation=quat
                            )
                        else: 
                            colocalization.child_frame_id = "A_odom"
                            colocalization.transform = Transform(
                                translation=Vector3(x=-1.0, y=0.0, z=0.0), # meters?
                                rotation=quat_inv
                            )

                        if id in self.results_dict: 
                            old_info = self.results_dict[id]
                            
                            # Localization info needs to be as recent as possible
                            self.results_dict[id].header.stamp = self.node.get_clock().now().to_msg()
                            self.results_dict[id].keyframe_id = 0 # msg.keyframe_id, change 
                            self.results_dict[id].curr_confidence = conf
                            self.results_dict[id].pitch = pitch_to_obj
                            self.results_dict[id].yaw = yaw_to_obj
                            # self.results_dict[id].angle_x = overall_angles_to_obj[0]
                            # self.results_dict[id].angle_y = overall_angles_to_obj[1]
                            # self.results_dict[id].angle_z = overall_angles_to_obj[2]
                            self.results_dict[id].distance = median_depth_val
                            self.results_dict[id].pose = odom_msg.pose.pose
                            self.results_dict[id].colocalization = colocalization
                            
                            # Feature descriptor should have highest confidence
                            if conf > old_info.max_confidence: 
                                cropped = im0[tlw:brw, tlh:brh]
                                new_embedding = self.cosplace_desc.compute_embedding(cropped)
                                # print(f"Embedding for existing obj: {new_embedding}")
                                self.results_dict[id].max_confidence = conf
                                self.results_dict[id].best_descriptor = [float(x) for x in new_embedding]
                        else: 
                            cropped = im0[tlw:brw, tlh:brh]
                            new_embedding = self.cosplace_desc.compute_embedding(cropped)
                            # print(f"Embedding for new obj: {new_embedding}")
                            self.results_dict[id] = MOTGlobalDescriptor(
                                header=Header(
                                    stamp=self.node.get_clock().now().to_msg(), 
                                    frame_id=img_msg.header.frame_id # this doesn't change
                                ),
                                robot_id=self.params['robot_id'], 
                                robot_id_to=1, # change
                                keyframe_id=0, # msg.keyframe_id
                                obj_id=id, 
                                obj_class_id=cls, 
                                max_confidence=conf,
                                best_descriptor=new_embedding,
                                curr_confidence=conf, 
                                pitch=pitch_to_obj, 
                                yaw=yaw_to_obj,
                                # angle_x=overall_angles_to_obj[0], 
                                # angle_y=overall_angles_to_obj[1], 
                                # angle_z=overall_angles_to_obj[2], 
                                distance=median_depth_val, 
                                pose=odom_msg.pose.pose,
                                colocalization = colocalization
                            )
                            
                        self.unified_id_mapping.update({id: id})
                        
                        if self.params['save_text']:
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (self.frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if self.params['save_vid'] or self.params['save_crop'] or self.params['show_video']:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            unified_id = self.unified_id_mapping[id]
                            label = None if self.params['hide_labels'] else (f'{unified_id} {self.names[c]}' if self.params['hide_conf'] else \
                                (f'{unified_id} {conf:.2f}' if self.params['hide_class'] else f'{unified_id} {self.names[c]} {conf:.2f}'))
                            plot_one_box(bboxes, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                            if self.params['save_crop']:
                                txt_file_name = txt_file_name if (isinstance(self.path, list) and len(self.path) > 1) else ''
                                self.save_one_box(bboxes, imc, file=self.save_dir / 'crops' / txt_file_name / self.names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                self.strongsort_list[i].increment_ages()
                print('No detections')

            # Stream results
            if self.params['show_video']:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if self.params['save_vid']:
                if self.vid_path[i] != save_path:  # new video
                    self.vid_path[i] = save_path
                    if isinstance(self.vid_writer[i], cv2.VideoWriter):
                        self.vid_writer[i].release()  # release previous video writer
                    if self.vid_cap:  # video
                        fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                self.vid_writer[i].write(im0)

            self.prev_frames[i] = self.curr_frames[i]
            
    def last_seen_callback(self): 
        stamp = self.node.get_clock().now().to_msg()
        self.mot_pub.publish(MOTGlobalDescriptors(
            header=Header(stamp=stamp), 
            descriptors=list(self.results_dict.values())))
        
        self.results_dict.clear()        
        
    def broadcast_transform_callback(self): 
        '''Send callback relating transform from agent 1 to agent 2 
        every 0.1 seconds, if robot has ID 0 only because sending from 
        both is unnecessary\n
        Update translation and rotation as necessary (ie not static)
        '''
        if self.params['robot_id'] == 0: 
            t1 = TransformStamped()
            t2 = TransformStamped()

            t1.header.stamp = self.node.get_clock().now().to_msg()
            t2.header.stamp = t1.header.stamp
            
            t1.header.frame_id = 'world'
            t1.child_frame_id = 'A_odom'
            
            t2.header.frame_id = 'world'
            t2.child_frame_id = 'B_odom'

            t1.transform.translation.x = 0.0 # meters
            t1.transform.translation.y = 0.0
            t1.transform.translation.z = 0.0
            t1.transform.rotation.x = 0.0
            t1.transform.rotation.y = 0.0
            t1.transform.rotation.z = 0.0
            t1.transform.rotation.w = 1.0

            t2.transform.translation.x = 0.3333 # meters
            t2.transform.translation.y = 0.0
            t2.transform.translation.z = 0.0
            t2.transform.rotation.x = 0.0
            t2.transform.rotation.y = 0.0
            t2.transform.rotation.z = 0.0
            t2.transform.rotation.w = 1.0

            # Send the transformations
            self.tf_broadcaster.sendTransform(t1)
            self.tf_broadcaster.sendTransform(t2)
        
            
    # UnifiedObjectIDs message
    def unified_info_callback(self, msg): 
        if msg.robot_id == self.params['robot_id']: 
            for i in range(len(msg.obj_ids)): 
                self.unified_id_mapping.update({msg.obj_ids[i]: msg.unified_obj_ids[i]})
                

    # Callback for commented out ApproximateTimeSynchronizer
    def depth_scratch_sync_callback(self, left_img_msg, right_img_msg): 
        left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "mono8")
        right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "mono8")
        
        left = pil_image.fromarray(left_img)
        left.save(f"/home/jrastogi/Documents/test_imgs_videos/hololens/rectified_stereo/left_{self.i:03}.png") # this works, just change path to desired

        right = pil_image.fromarray(right_img)
        right.save(f"/home/jrastogi/Documents/test_imgs_videos/hololens/rectified_stereo/right_{self.i:03}.png") # this works, just change path to desired
        print(f"Saved frame {self.i}")

        self.i += 1
        