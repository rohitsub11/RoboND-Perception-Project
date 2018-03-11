## Project: Perception Pick & Place
### Rohit Subramanian

---
[//]: # (Image References)
[input]: ./images/input_image.png
[img_table]: ./images/img_table.png
[objects]: ./images/objects.png
[cluster]: ./images/clustering.png
[training]: ./images/training.png
[gazebo]:./images/gazebo_image.png
[test1]:./images/test1_list1.png
[test2]:./images/test2_list2.png
[test3]:./images/test3_list3.png
[confusion_norm]: ./images/normalized_confusion.png
[confusion_nonnorm]: ./images/nonnormal_confusion.png

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

The following is a screenshot of the Gazebo workspace
![gazebo]
### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
The goal of this exercise was to create 'object' and 'table' point cloud separately.


1. Load the ROS msg Point Cloud data `pc2.PointCloud2` as `pcl_msg` into the `pcl_callback` function and convert it to PCL data. The input point cloud data with noise from RViz is shown below:
 ![input]
```python
# Convert ROS msg to PCL data
cloud = ros_to_pcl(pcl_msg)
 ```
2. Use the  "Statistical Outlier Filter" and "VoxelGrid Downsampling Filter" to get ride of the outliers and downsampling the point cloud data. This is shown in my code as follows:
```python
# Statistical Outlier Filter
outlier_filter = cloud.make_statistical_outlier_filter()
## Set the number of neighboring points to analyze for any given point
outlier_filter.set_mean_k(20)
## Any point with a mean distance larger than global will be considered out
outlier_filter.set_std_dev_mul_thresh(0.3)
cloud_filtered = outlier_filter.filter()

# VoxelGrid Downsampling Filter
vox = cloud_filtered.make_voxel_grid_filter()
## Choose a voxel (also known as leaf) size
LEAF_SIZE = 0.01
## Set the voxel (or leaf) size  
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
## Call the filter function to obtain the resultant downsampled point cloud
cloud_vox = vox.filter()
```
3. Use "Pass Through Filter" to crop the given 3D point cloud into different regions of interest. This is shown in my code as follows:
```python
# PassThrough Filter
## Create a PassThrough filter object.
passthrough = cloud_vox.make_passthrough_filter()
## Assign z axis and range to the passthrough filter object.
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.2
passthrough.set_filter_limits(axis_min, axis_max)
## Use the filter function to obtain the resultant point cloud. 
cloud_passthrough = passthrough.filter()

## Use Another PassThrough Filter to select the final regions of interest.
passthrough = cloud_passthrough.make_passthrough_filter()
## Assign y axis and range to the passthrough filter object.
filter_axis = 'y'
passthrough.set_filter_field_name(filter_axis)
axis_min = -0.5
axis_max = 0.5
passthrough.set_filter_limits(axis_min, axis_max)
## Use the filter function to obtain the resultant point cloud. 
cloud_passthrough = passthrough.filter()
```
4. Use RANSAC Plane Segmentation to identify points in your dataset that belong to a particular model. This is shown in my code as follows:
```python
# RANSAC Plane Segmentation
## Create the segmentation object
seg = cloud_passthrough.make_segmenter()
## Set the model you wish to fit 
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
## Max distance for a point to be considered fitting the model
## Experiment with different values for max_distance 
## for segmenting the table
max_distance = 0.01
seg.set_distance_threshold(max_distance)
## Call the segment function to obtain set of inlier indices and model coefficients
inliers, coefficients = seg.segment()
```
5. Extract Indices and Objects. This is shown in my code as follows:
```python
# Extract inliers (table) and outliers (objects)
## Extract inliers
cloud_table = cloud_passthrough.extract(inliers, negative=False)
## Extract outliers
cloud_objects = cloud_passthrough.extract(inliers, negative=True)
```
The following image shows extracted objects:
![objects]

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
Use Euclidean Clustering to seperate different segmenetations.
```python
# Euclidean Clustering
white_cloud = XYZRGB_to_XYZ(cloud_objects)
tree = white_cloud.make_kdtree()
## Create a cluster extraction object
ec = white_cloud.make_EuclideanClusterExtraction()
## Set tolerances for distance threshold as well as minimum and maximum cluster size (in points)
ec.set_ClusterTolerance(0.01)
ec.set_MinClusterSize(25)
ec.set_MaxClusterSize(3000)
## Search the k-d tree for clusters
ec.set_SearchMethod(tree)
## Extract indices for each of the discovered clusters
cluster_indices = ec.Extract()

# Create Cluster-Mask Point Cloud to visualize each cluster separately
## Assign a color corresponding to each segmented object in scene
cluster_color = get_color_list(len(cluster_indices))
color_cluster_point_list = []
for j, indices in enumerate(cluster_indices):
    for i, indice in enumerate(indices):
        color_cluster_point_list.append([white_cloud[indice][0],
                                        white_cloud[indice][1],
                                        white_cloud[indice][2],
                                         rgb_to_float(cluster_color[j])])
## Create new cloud containing all clusters, each with unique color
cluster_cloud = pcl.PointCloud_PointXYZRGB()
cluster_cloud.from_list(color_cluster_point_list)
```

After trial and error, I found appropriate values for the max and min cluster sizes as well as cluster tolerance and the result is shown in the following image:
![cluster]

#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Following the "[Object Recognition](https://classroom.udacity.com/nanodegrees/nd209/parts/586e8e81-fc68-4f71-9cab-98ccd4766cfe/modules/e5bfcfbd-3f7d-43fe-8248-0c65d910345a/lessons/81e87a26-bd41-4d30-bc8b-e747312102c6/concepts/8389976c-550a-4fde-a530-65ca24b31f05)" steps to implement object recognition to the filtered 'objects' point cloud data. The snippets of code are shown below:
```python
# Classify the clusters! (loop through each detected cluster one at a time)
detected_objects_labels = []
detected_objects = []
for index, pts_list in enumerate(cluster_indices):
    ## Grab the points for the cluster
    pcl_cluster = cloud_objects.extract(pts_list)
    # Convert the cluster from pcl to ROS using helper function
    ros_cluster = pcl_to_ros(pcl_cluster)
    ## Extract histogram features
    # complete this step just as you did before in capture_features.py
    chists = compute_color_histograms(ros_cluster, True)
    normals = get_normals(ros_cluster)
    nhists = compute_normal_histograms(normals)
    feature = np.concatenate((chists, nhists))

    ## Make the prediction, retrieve the label for the result and add it to detected_objects_labels list
    prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
    label = encoder.inverse_transform(prediction)[0]
    detected_objects_labels.append(label)
```

Capture point cloud features based on the project models instead of the exercise models. First, I changed the models in `capture_features.py`:
```
models = [
    'biscuits',
    'book',
    'eraser',
    'glue',
    'snacks',
    'soap',
    'soap2',
    'soda_can',
    'sticky_notes']
```
In a terminal window, run:
```
$ cd ~/catkin_ws
$ roslaunch sensor_stick training.launch
```
In another terminal window, run:
```
$ cd ~/catkin_ws
$ rosrun sensor_stick capture_features.py
```
After the features were captured, run:
```
$ rosrun sensor_stick train_svm.py
```
The training results are:

![training]
![confusion_norm]
![confusion_nonnorm]

The accuracy was 91% (+/- 0.05). 
It also created the model file `model.sav` which i renamed to `model_new.sav` to separate from exercise models.

The `pr2_mover()` function is to load parameters and request PickPlace service. It create ROS messages containing the details of each object (name, pick_pose, etc.) and writes these messages out to different `output_*.yaml` files corresponding to different 'pick list' scenarios.


### Pick and Place Setup
### Project Process
In a terminal window, run:
```
$ roslaunch pr2_robot pick_place_project.launch
```
In another terminal winodw, run:
```
rosrun pr2_robot object_recognition.py
```

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

To change the different objects settings, change the `test*.world` and `pick_list_*.yaml` in line 13 and 39 of `./launch/pick_place_project.launch` to 1, 2 and 3, respectively.

For `test1.world` and `pick_list_1.yaml`, the result is:
![test1]

For `test2.world` and `pick_list_2.yaml`, the result is:
![test2]

For `test3.world` and `pick_list_3.yaml`, the result is:
![test3]

The `perception_pipeline.py` is able to achieve 100%(3/3) of objects from `pick_list_1.yaml` for `test1.world`, close to 100%(5/5) of items from `pick_list_2.yaml` for `test2.world` and almost 100%(8/8) of items from `pick_list_3.yaml` in `test3.world`.


