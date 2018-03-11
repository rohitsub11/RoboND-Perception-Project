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


1. Load the ROS msg Point Cloud data `pc2.PointCloud2` as `pcl_msg` into the `pcl_callback` function and convert it to PCL data. 
Stepping through the `pcl_callback` function, we first conver the ros message to pcl with `ros_to_pcl` helper function. The input point cloud data with noise from RViz is shown below:
 ![input]

2. Use the  "Statistical Outlier Filter" and "VoxelGrid Downsampling Filter" to get ride of the outliers and downsampling the point cloud data. Since there is noise in the data, statistical outlier filter is applied to remove this noise. The filter assumes a normal distribution and compares the neighboring points to the mean before classifying those points lying outside a threshold as outliers and discards them. In `perception_pipeline.py` this is done in lines `59-64`.

Voxel grid downsampling is then performed. Since the RGB-D camera provides a data-dense input represtations, we need to cut it down for processing. We define a box of a certain size in 3D world of pixels. Find the center, take the average value over all the pixels inside the box, assign it the the center and discard the rest. This is a trade-off that when balanced properly leads to a recognizable and accurate pointcloud representation of an object that isn't as data dense as it used to be. In `perception_pipeline.py` this is done in lines `68-77`.

3. Use "Pass Through Filter" to crop the given 3D point cloud into different regions of interest. The passthrough filter acts as a cropping tool to cut off pixels beyond a given setting in the x,y,z space. For example in this project, if we know the height of the table that is in front of your robot (the top of which is making up it's workspace) you can discard pixels that come from below the height of the table (in the 'z' direction) over a specified range. This effectively eliminates everything below the table from the picture. In `perception_pipeline.py` this is done in lines `81-104`.

4. Use RANSAC Plane Segmentation to identify points in your dataset that belong to a particular model. After the previous steps, the pixels forming the tabletop and objects sitting on it remain. To separate these from each other, Random Sample Consensus Algorithm is used. It looks for the components of geometric shapes in order to find the plane shape of the table and divide the pixels into ones that fit the plane (inliers) and ones that don't (outliers). In `perception_pipeline.py` this is done in lines `108-128`.

5. Extract Indices and Objects. 
The following image shows extracted objects:
![objects]

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

The downside of RANSAC model fitting is that you have to compute it over the entire pointcloud once per object in order to filter out all the other objects and leave just one object of interest. This is not optimal. Also many objects in the scene look similar to each other geometrically (cylindrical objects like cups or cans) and therefore would be mistaken for each other by RANSAC. So we use other approaches to define our objects of interest. 

So we apply K-D Tree Clustering. This requires x,y,z data only, no color. So a white-cloud is formed using a xyzrgb_to_xyz helper function. Clusters are found and extracted after min/max tolerances and cluster size are set. Colors are assigned to each cluster to distinguish them from one another. We have segmented our data into individual objects. For each point-cloud object we have RGB color data as well as 3-D spatial information. Now we need to identify what those objects are. In `perception_pipeline.py` this is done in lines `131-163`.

After trial and error, I found appropriate values for the max and min cluster sizes as well as cluster tolerance and the result is shown in the following image:
![cluster]

#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Following the "[Object Recognition](https://classroom.udacity.com/nanodegrees/nd209/parts/586e8e81-fc68-4f71-9cab-98ccd4766cfe/modules/e5bfcfbd-3f7d-43fe-8248-0c65d910345a/lessons/81e87a26-bd41-4d30-bc8b-e747312102c6/concepts/8389976c-550a-4fde-a530-65ca24b31f05)" steps to implement object recognition to the filtered 'objects' point cloud data. The SVM classifier works based on the features of the data identified in the HSV color space and by the shape given by the surface normals. By 'binning up' our color data into color histogram bins, we can obtain a unique histogram signature of each object to learn from. Similarly, the distribution of surface normals contains shape information for each object that can be learned from. A support vector machine is used to characterize the parameter space into distinct classes. This algorithm draws decision boundaries around data groups, evaluates the level of success of these boundaries compared to a ground-truth label (requires labeled data for training) and then iteratively improves on the best location for the boundaries. After the feature space has been subdivided in this way, new unlabeled data can be categorized simply via reference to where it falls in the space due to its features. In other words it can be used for object recognition.

First, I changed the models in `capture_features.py`:
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

The confusion matrix below demonstrates the accuracy of recognition across every type of object trained. An overall accuracy of 91% (+/- 0.05) was attained.

![training]
![confusion_norm]
![confusion_nonnorm]

It also created the model file `model.sav` which i renamed to `model_new.sav` to separate from exercise models.

The flow of object recognition is implemented in the following manner: Instantiate two empty lists: one for detected objects and one for labels. Loop over the list of euclidean-clustered objects, extracting those points which correspond to our extracted_outliers. Convert to ROS from PCL. Compute associated feature vectors using compute_color_histograms and compute_normal_histograms functions. Normalize and concatenate these into a feature vector. Use the SVM boundaries to predict the identity of the object based on it's feature vector. Assign a label based on the prediction. Add the labeled object to the detected object list. Publish the detected objects list.


### Pick and Place Setup
### Project Process
After the pcl_callback function comes the pr2_mover function. This function loads the parameters with regard to the scene/object data and requests a pick/place service. A number of variables are initialized including the object name, group, pick_pose, place_pose and the arm needed to grab the object. The centroid of each object is calculated as the mean of it's numpy points array. This will be used later to define the pick_pose. The pick list is then parsed and looped over in order to assign appropriate data to the variables we defined above for each object. The pick_list object name data field is compared to the label of a point cloud object and when they match, the scalar-version of the centroid data is assigned as that object's pick-pose data. The place pose is defined based on the object's group. Arm data is assigned accordingly. This data is then written out in yaml dictionary format for a pick_place routine to use in directing robot motion.

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

To change the different objects settings, change the `test*.world` and `pick_list_*.yaml` in line 13 and 39 of `./launch/pick_place_project.launch` to 1, 2 and 3, respectively.

For `test1.world` and `pick_list_1.yaml`, the result is:
![test1]

For `test2.world` and `pick_list_2.yaml`, the result is:
![test2]

For `test3.world` and `pick_list_3.yaml`, the result is:
![test3]

The `perception_pipeline.py` is able to achieve 100%(3/3) of objects from `pick_list_1.yaml` for `test1.world`, close to 100%(5/5) of items from `pick_list_2.yaml` for `test2.world` and almost 100%(8/8) of items from `pick_list_3.yaml` in `test3.world`.


