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

**Notes**:
1. Currently, the model requires the training data to be in jpg. By default, the images in the SpaceNet dataset are in geotiff. You can do the conversion via `gdal_translate` from the GDAL library. 
2. Expected data format: MS COCO
