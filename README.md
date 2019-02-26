## **Mask RCNN for Spacenet Off Nadir Building Detection**

This repository reuses Matterport's Mask RCNN implementation. Kindly refer their implementation https://github.com/matterport/Mask_RCNN for detailed documentation on many aspects of this code.

#### **Current Score:**

mAP | F1 
--- | --- 
30.4 | 37.9 

### **Dataset**

You can download the data via the following links:

**Training data imagery**:

    aws s3 cp s3://spacenet-dataset/SpaceNet_Off-Nadir_Dataset/SpaceNet-Off-Nadir_Train/ . --exclude "*geojson.tar.gz" --recursive
    
**Training data labels**:

    aws s3 cp s3://spacenet-dataset/SpaceNet_Off-Nadir_Dataset/SpaceNet-Off-Nadir_Train/geojson.tar.gz .
    
Refer the SpaceNet Off Nadir challenge page ([link](https://spacenetchallenge.github.io/Challenges/challengesSummary.html)) for more details

Latest weights can be found [here](https://drive.google.com/open?id=1CExnB6BaZ8sjA7JIpVcuQLCgoHCjWqHd)

### **Sample results**

Sample results of using this model on Nadir (left, nadir angle=13&deg;), Off Nadir (center, nadir angle=27&deg;) and Very Off Nadir (right, nadir angle=50&deg;) images are shown below.

<p align="center">
   <img src="https://github.com/ash1995/Mask-RCNN-for-Off-Nadir-Building-Detection/blob/master/example_images/Atlanta_nadir13_catid_1030010002B7D800_748451_3735939.png" alt="Example result of MaskRCNN on SpaceNet"/ width=275 img align="left">
  <img src="https://github.com/ash1995/Mask-RCNN-for-Off-Nadir-Building-Detection/blob/master/example_images/Atlanta_nadir27_catid_1030010003472200_739451_3740439.png" alt="Example result of MaskRCNN on SpaceNet"/ width=275 img align="center">
  <img src="https://github.com/ash1995/Mask-RCNN-for-Off-Nadir-Building-Detection/blob/master/example_images/Atlanta_nadir50_catid_10300100039E6200_746201_3721539.png" alt="Example result of MaskRCNN on SpaceNet"/ width=275 img align="right">
</p>


### **Notes**:
1. Currently, the model requires the training data to be in jpg. By default, the images in the SpaceNet dataset are in geotiff. You can do the conversion via `gdal_translate` from the GDAL library. 
2. Expected data format: MS COCO
3. There is some issue with using the default cocoeval.py script for evaluating this dataset. Refer this [notebook](https://github.com/ash1995/Mask-RCNN-for-Off-Nadir-Building-Detection/blob/master/samples/sate/Calculate_metrics.ipynb) for calculating metrics. 
