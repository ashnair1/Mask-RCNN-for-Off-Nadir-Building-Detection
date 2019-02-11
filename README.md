## **Mask RCNN for Spacenet Off Nadir Building Detection**

This repository reuses Matterport's Mask RCNN implementation. Kindly refer their implementation https://github.com/matterport/Mask_RCNN for detailed documentation on many aspects of this code.

Spacenet data in MS COCO format can be found [here](https://drive.google.com/open?id=1Mx_QThYvQ3t2vS71EaIAhsPGQ9YGIeqv#) <br>
Latest weights can be found [here](https://drive.google.com/open?id=1CExnB6BaZ8sjA7JIpVcuQLCgoHCjWqHd)

**Warning**: The format of the dataset differs from the standard MS COCO format with regards to the image id. In COCO annotations the image id is an int but in this particular dataset the image id is a string. A slight modification will have to be made in pycocotools's coco.py to allow for string ids.
