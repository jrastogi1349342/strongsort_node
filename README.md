# Decentralized Multi-Agent Multi-Object Tracking (Yolov7 + StrongSORT with OSNet)



## Description and Design Considerations

This repository contains a highly configurable multi-agent two-stage-tracker, accounting for inter-agent object associations, with communication through ROS2 Humble. It requires the `strongsort_msgs` package, linked [here](https://github.com/jrastogi1349342/strongsort_msgs), which contains the custom messages used for communication. The pipeline works as follows: 

* For each agent launched, `launch/strongsort_node_launch.py` launches two processes: the `StrongSortSetup` and `AssociationsROSDriver` classes. These two classes initialize parameters and launch the two driver classes: `StrongSortPublisher` and `ObjectAssociation`.
* In the `StrongSortPublisher` class, each agent runs multi-object tracking through [YOLOv7](https://github.com/WongKinYiu/yolov7) (currently pretrained on the COCO dataset), and the [StrongSORT](https://github.com/dyhBUPT/StrongSORT) and [OSNet](https://github.com/KaiyangZhou/deep-person-reid) tracking pipeline. A compact feature descriptor of each detection is extracted through the [CosPlace](https://github.com/gmberton/CosPlace) Visual Geo-localization Algorithm. The agent detects the object and computes its relative location to it (depth, pitch, yaw). The information about a singular detection is packaged in a `MOTGlobalDescriptor` object, and each set of detections is packaged in a `MOTGlobalDescriptors` object and published to the `/mot/descriptors` topic. 
    * This code subscribes to 4 topics for each agent: 
        * Left stereo camera: `{namespace}{camera}` topic (where the default `{camera}` is `/rm_vlc_leftfront/image`)
        * Disparity map from the stereo cameras: `{namespace}/stereo/depth` topic
        * Camera info of left stereo camera: `{namespace}/stereo/left/camera_info` topic
        * Odometry: `{namespace}/odom` topic
    * On the Hololens 2, all cameras are grayscale. I used numpy to stack the channels to the correct dimension for YOLOv7. However, this 'hack' has experimentally has shown poor performance for smaller objects, especially ones that are against a similar-colored background. Future work includes training an object detection model designed for grayscale images, and using more relevant ontologies than the Microsoft COCO dataset. 
        * Note that the depth, yaw, and pitch calculations are specifically written for the Hololens 2. 
    * The StrongSORT + OSNet object tracking pipeline seems to lose occluded objects within a few seconds. This becomes a problem with my clustering pipeline, due to my assumption that "Each agent only has one track for each detected object" (assumption 4 of the clustering process, seen below). Future work includes integrating features from other multi-object tracking pipelines to improve the tracking quality. 
        * I have not tested its performance against partially or completely camouflaged objects. Future work includes testing this performance and if it performs poorly, finding or developing improvements. 
    * I did not use LiDaR or GPS information because I assumed agents would be operating in constrained environments. For agents that have a reliable LiDaR, such as the Vision 60, future work involves integrating camera and LiDaR depth information for more reliable object tracking. 
    * For the depth information from camera-only systems, I assume that the disparity map is created with the parameter `stereoParams.maxDisparity = 48` in the VPI Stereo Disparity calculation. This provides a reasonably accurate distance calculation from approximately 0.3 to 3 meters. By reducing the value of the `maxDisparity` parameter, the minimum and maximum detectable distance increase; I did not test the max distance it can accurately detect. Future work includes potentially calculating the disparity multiple times, using a different `maxDisparity` value, to cover a broader range of distances. 
        * I use the median of the distances to each pixel on the object (image cropped to the bounding box) to reduce outliers. Future work may involve using segmentation techniques to increase accuracy and robustness. 
        * This approach still requires stereo cameras, so depth cannot be calculated from the side cameras of the Hololens 2 or Vision 60. Future work involves using alternative methods to reliably calculate distances to objects, such as potentially neural network approaches. 
    * Pitch refers to the angle up and down from the horizontal line that bisects the image. It is positive in the top half of the image. 
    * Yaw refers to the angle left and right from the vertical line that bisects the image. It is positive in the right half of the image. 
    * Distance, pitch, and yaw are converted into `xyz` coordinates, from the perspective of the camera. They maintain the same coordinate system as the camera, where positive `x` is inwards, `y` is down, and `z` is right from camera perspective. This also means that `x` is much larger than `y` and `z`. 
    * I use the exact same CosPlace model as Swarm SLAM, so when it is integrated, both pipelines can use the same model. 
    * I simulate colocalization with the `broadcast_transform_callback` function. The agents start with known transforms from the fixed frame, `'world'`, to `{namespace}_odom`, and use `world` to calculate transforms between agents in different times. 
    * The `self.unified_id_mapping` dict saves the results of inter-agent object associations. It currently only changes the detection label in the video feed for visualization. Future work involves classifying objects (or groups of objects, such as a soldier with a gun) as threats, through action detection or other techniques. 

- In the `ObjectAssociation` class, each agent subscribes to the `/mot/descriptors` topic and adds all the information about each new detection to a Disjoint Set data structure, which uses union by rank and path compression for efficiency. Then, every `self.params['sort.re_cluster_secs']` seconds, a broker is chosen to perform the inter-agent object association. This broker iterates over all the clusters from each agent in range that aren't "full" (have an associated detection from each agent), and performs a custom version of single-linkage agglomerative clustering. It checks that the two prospective detections are in the same place in space (`check_same_location` method) and that they have very similar feature extractions. If a pair of clusters are similar enough, by these metrics, they are clustered together and an unified labeling scheme is constructed. If the broker has seen the object, the broker's label becomes the global label; otherwise, it becomes the next descending integer, starting from -1. 
    * I use the same neighbor management module as the Swarm SLAM repository, though I assumed that all agents would remain in range at all times. Future work involves being more faithful to the original Swarm SLAM  repository to accurately send all messages to all other agents, regardless of if they are in range or out of range of any other subset of agents at any time. 
    * I implemented two different strategies for broker selection: 
        * Highest `robot_id` in range
            * Swarm SLAM uses the lowest `robot_id` in range, and since the computationally expensive part in both algorithms requires at least two agents, this guarantees the brokers are not the same for Swarm SLAM and this repository. 
        * Most CPU usage available
            * The clustering process is currently primarily CPU intensive. It is not parallelized, and each agent already has the detections in RAM. 
    * I chose a Disjoint Set data structure because each detection cannot be associated with multiple agents at once. 
    * In the process of unifying two clusters, I also delete most of the information regarding the 'child' cluster, since it isn't used anymore. Any new detections just update the parent cluster.  
    * Each cluster also has an associated Kalman Filter with it. After clustering, any agent that next detects that object can update the location, from the perspective of the `broker_id`. The `Q` and `R` matrices are very hacky and could be better tuned. 
        * The Kalman Filter also contains two modifications from a standard version: 
            * The `R` matrix depends on the confidence of the detection.
            * The change in time, `dt`, is the difference between when the next `MOTGlobalDescriptors` message is received and the last time the location was updated, instead of being fixed. 
        * If the specific detection is spotted in one frame, I run the `predict` and `update` functions of the Kalman Filter to improve the accuracy of the location and covariance matrices. 
        * If the specific detection is not spotted in one frame, I run the `predict` function of the Kalman Filter to simulate the object's location in space at that timeframe. 
    * The assumptions made in the clustering process are very rigid and 'perfect', as shown below: 
        * Every class is extremely different from every other class. 
            * This implies that if two detections of the same object have different class labels (which happens occasionally), they cannot be clustered together. Future work involves using metric learning to compute a metric of how similar two classes are. Alternatively, you may be able to use a vision-language model to develop a semantic understanding of the object, and its associated properties (i.e. size, location, etc). 
        * Co-localization is extremely accurate. 
            * I currently have not seen how good co-localization is, so I don't know if this is a valid assumption or not. Future work involves testing the capabilities of co-localization. If it isn't enough, the distance metrics will have to be tuned accordingly, or adjusted to a function of distance from the other agent. 
        * Objects look similar from all angles. 
            * The front and back of a person look very different from each other, so they wouldn't be matched properly. Future work involves somehow adding invariance to object orientation for the purpose of object tracking, if possible. 
        * Each agent only has one track for each detected object. 
            * In practice, StrongSORT inevitably loses and re-intializes a track for each object, due to occlusions, so the re-added track cannot be clustered as the same object. Future work includes accounting for this assumption. 

Note that this is only one step in the robot's stack for collaborative human-robot interaction, so it needs to be very robust and efficient. As a result, inter-agent communication needs to be decentralized. The next step on the stack would be classifying subsets of the clustered tracks as threats/not threats, through Action Detection, semantic understanding of the tracked checks, and/or some other technique(s). 

There may be other design decisions that I forgot to include. If you are wondering about why I did something, please email me at jai1rastogi@gmail.com. 

## Known Unaddressed Edge Cases
* YOLO occasionally jitters and provides a detection for one frame that doesn't match the others. This gets propagated in my clustering algorithm, and future work includes post-detection sanity checks to reduce the chance of object mis-detections. 
* The functionality for sending messages to each other agent likely doesn't work if agents move in and out of range of each other. 
* In the `check_same_location` method in the `ObjectAssociation` class, computing `lookup_transform_full` sometimes results in an "Requires extrapolation into the past" error and fails to compute. In my testing, it only occurred if an object had not been detected in at least 5 or 6 seconds. One potential solution may be to use the `self.pose_dict` dictionary and current transforms to get around the fact that 
* I was unable to test the negative number functionality for label association, since I only had two agents (I didn't know how to use the Vision 60) and this functionality relies on having at least 3 agents. 
* Something becomes occluded, strongsort loses it, and then it gets reassigned to a different ID and clustering doesn't work

Note that there might be more edge cases. 

## Future Work for this Project
Use a find feature (i.e. Ctrl-F) to search the term "future work" under the "Description and Design Considerations" heading. I listed many different potential approaches, which vary between the level of an undergraduate and someone with a masters/PhD. 

## Installation Instructions

1. Clone the following repositories recursively inside `{ros_ws}/src`, where `{ros_ws}` is your ROS2 workspace:

`git clone https://github.com/jrastogi1349342/strongsort_msgs`

`git clone --recurse-submodules https://github.com/jrastogi1349342/strongsort_node`

If you already cloned `strongsort_node` and forgot to use `--recurse-submodules`, you can run `git submodule update --init` from the `strongsort_node/` directory. 

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/jrastogi1349342/strongsort_node/blob/main/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

`pip install -r requirements.txt`

3. From `{ros_ws}`, run `colcon build --packages-select strongsort_msgs strongsort_node` 


## How to Run
1) Launch HL2SS on each Hololens 2, and note the IP address
2) Open the `holo-ros` directory in the Dev Container in VS Code, and for each Hololens 2, launch `ros2 launch holo_ros hololens_driver.launch.py host:={IP address} namespace:={namespace}` from the `dev@{device}:/home/ws` directory in terminal
    1) Keep in mind that this will run the strongsort_node repository on your local machine
3) For each Hololens, run the following command from your `{ros_ws}`: 
`ros2 launch strongsort_node strongsort_node_launch.py name_space:={namespace} robot_id:={robot_id} max_nb_robots:={max_robots}`

Here is an example, using 2 agents on my laptop: 
1) The IP addresses of the two Hololenses are 131.218.141.219 and 131.218.141.231. 
2) Open the Dev Container, and run the following commands, each in a different bash shell (e.g. `dev@arl-Precision-7760:/home/ws$`) in the dev container: 
    1) `ros2 launch holo_ros hololens_driver.launch.py host:=131.218.141.231 namespace:=A`
    2) `ros2 launch holo_ros hololens_driver.launch.py host:=131.218.141.219 namespace:=B`
3) In two different terminals, from your local `{ros_ws}` (not in the dev container): 
    1) `ros2 launch strongsort_node strongsort_node_launch.py name_space:=A robot_id:=0 max_nb_robots:=2`
    2) `ros2 launch strongsort_node strongsort_node_launch.py name_space:=B robot_id:=1 max_nb_robots:=2`


## Tracking sources

In the `parameters` list in `strongsort_node_list.py`, change `video_topic` to the desired ROS2 topic. If you want to add new parameters and pass them to the pipeline, feel free to do so. You can also let the user add parameter values from the launch command directly if desired; an example of this is `name_space:=B` (i.e., the name space is set to "B"). 


## Select object detection and ReID model

The first time you run the overall system, as above, the `.pt` file for YOLOv7 and StrongSORT will be downloaded to your `{ros_ws}` directory. 

### YOLOv7

The default YOLOv7 model is `yolov7.pt`. If you want to use a different version, change the `parameters` list by adding the `yolo_weights` key and your desired YOLOv7 family model. It will be automatically downloaded. 

Note that there is a clear trade-off between model inference speed and accuracy.

### StrongSORT

The above applies to StrongSORT models as well. The default StrongSORT ReID model is `osnet_x0_25_msmt17.pt`. If you want to use a different version, choose a ReID model [model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO), and change the `parameters` list by using the `strong_sort_weights` key. 


## Filter tracked classes

By default the tracker tracks all MS COCO classes.

If you want to track a subset of the MS COCO classes, add an array with all desired class numbers to the `classes` key in the `parameters` list. 

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov7 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero.


## Contact for Multi-Agent Multi-Object Tracking Pipeline
For any questions regarding multi-agent multi-object tracking, please contact Jai Rastogi at jai1rastogi@gmail.com. Please note that he may not work on this repository beyond Summer 2024. 

## Contact for StrongSORT Pipeline

For Yolov7 DeepSort OSNet bugs and feature requests please visit [GitHub Issues](https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet/issues). For business inquiries or professional support requests regarding StrongSORT, please send an email to: yolov5.deepsort.pytorch@gmail.com

## Sources
[1] Berton, Gabriele, Carlo Masone, and Barbara Caputo. "Rethinking visual geo-localization for large-scale applications." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[2] Du, Yunhao, et al. “Strongsort: Make deepsort great again.” IEEE Transactions on Multimedia 25 (2023): 8725-8737. 

[3] Lajoie, Pierre-Yves, and Giovanni Beltrame. "Swarm-slam: Sparse decentralized collaborative simultaneous localization and mapping framework for multi-robot systems." IEEE Robotics and Automation Letters 9.1 (2023): 475-482. 

[4] Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[5] Zhou, Kaiyang, et al. "Omni-scale feature learning for person re-identification." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
