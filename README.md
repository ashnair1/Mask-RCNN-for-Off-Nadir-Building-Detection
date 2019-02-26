## **Mask RCNN for Spacenet Off Nadir Building Detection**

This repository reuses Matterport's Mask RCNN implementation. Kindly refer their implementation https://github.com/matterport/Mask_RCNN for detailed documentation on many aspects of this code.


Download data via:

**Training data imagery**:

    aws s3 cp s3://spacenet-dataset/SpaceNet_Off-Nadir_Dataset/SpaceNet-Off-Nadir_Train/ . --exclude "*geojson.tar.gz" --recursive
    
**Training data labels**:

    aws s3 cp s3://spacenet-dataset/SpaceNet_Off-Nadir_Dataset/SpaceNet-Off-Nadir_Train/geojson.tar.gz .
    
Refer the SpaceNet Off Nadir challenge page ([link](https://spacenetchallenge.github.io/Challenges/challengesSummary.html)) for more details

Latest weights can be found [here](https://drive.google.com/open?id=1CExnB6BaZ8sjA7JIpVcuQLCgoHCjWqHd)


