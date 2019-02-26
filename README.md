## **Mask RCNN for Spacenet Off Nadir Building Detection**

This repository reuses Matterport's Mask RCNN implementation. Kindly refer their implementation https://github.com/matterport/Mask_RCNN for detailed documentation on many aspects of this code.

**Current Score:**

mAP | F1 
--- | --- 
30.4 | 37.9 

Download data via:

**Training data imagery**:

    aws s3 cp s3://spacenet-dataset/SpaceNet_Off-Nadir_Dataset/SpaceNet-Off-Nadir_Train/ . --exclude "*geojson.tar.gz" --recursive
    
**Training data labels**:

    aws s3 cp s3://spacenet-dataset/SpaceNet_Off-Nadir_Dataset/SpaceNet-Off-Nadir_Train/geojson.tar.gz .
    
Refer the SpaceNet Off Nadir challenge page ([link](https://spacenetchallenge.github.io/Challenges/challengesSummary.html)) for more details

Latest weights can be found [here](https://drive.google.com/open?id=1CExnB6BaZ8sjA7JIpVcuQLCgoHCjWqHd)

Example output images using this model are shown below.

<p align="center">
   <img src="https://github.com/ash1995/Mask-RCNN-for-Off-Nadir-Building-Detection/blob/master/example_images/Atlanta_nadir13_catid_1030010002B7D800_748451_3735939.png" alt="Example result of MaskRCNN on SpaceNet"/ width=650>
  <img src="https://github.com/ash1995/Mask-RCNN-for-Off-Nadir-Building-Detection/blob/master/example_images/Atlanta_nadir27_catid_1030010003472200_739451_3740439.png" alt="Example result of MaskRCNN on SpaceNet"/ width=650>
  <img src="https://github.com/ash1995/Mask-RCNN-for-Off-Nadir-Building-Detection/blob/master/example_images/Atlanta_nadir50_catid_10300100039E6200_746201_3721539.png" alt="Example result of MaskRCNN on SpaceNet"/ width=650>
</p>


**Notes**:
1. Currently, the model requires the training data to be in jpg. By default, the images in the SpaceNet dataset are in geotiff. You can do the conversion via `gdal_translate` from the GDAL library. 
2. Expected data format: MS COCO
