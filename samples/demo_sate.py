import os
import sys
import glob
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import pickle
import image_slicer
from image_slicer import join
from PIL import Image

def bbox_iou(bboxes1, bboxes2,bb_format="LTRB"):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

        LTRB format:

        p1 *-----
           |     |
           |_____* p2


        XYWH format:

        p1 *--w---
           |     |
           |     h
           |     |
           |_____* 

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """
    if type(bboxes1) is list:
        bboxes1 = np.array(bboxes1)
    if type(bboxes2) is list:
        bboxes2 = np.array(bboxes2)

    if bb_format == "LTRB":
        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)
    elif bb_format == "XYWH":
        x11, y11, w1, h1 = tf.split(bboxes1, 4, axis=1)
        x21, y21, w2, h2 = tf.split(bboxes2, 4, axis=1)

        x12 = x11 + w1
        y12 = y11 - h1

        x22 = x21 + w2
        y22 = y21 - h2

    xI1 = tf.maximum(x11, tf.transpose(x21))
    xI2 = tf.minimum(x12, tf.transpose(x22))

    yI1 = tf.minimum(y11, tf.transpose(y21))
    yI2 = tf.maximum(y12, tf.transpose(y22))

    inter_area = tf.maximum((xI2 - xI1), 0) * tf.maximum((yI1 - yI2), 0)

    bboxes1_area = (x12 - x11) * (y11 - y12)
    bboxes2_area = (x22 - x21) * (y21 - y22)

    union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.

    iou = inter_area / (union+0.0001)

    with tf.Session() as sess:
        iou_val = sess.run(iou)


    return iou_val 


def main():

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on satellite images.')
    parser.add_argument("--size",
                        metavar="size",
                        default="small",
                        help="'large' or 'small' images")

    args = parser.parse_args()
    print("Image Type: ", args.size)



    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    # from mrcnn.visualize import save_image # added by JX

    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    import sate

    #matplotlib inline

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_sate_b2_0071.h5")
    #COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_sate_0055.h5")
    #COCO_MODEL_PATH = os.path.join(ROOT_DIR, "crowdai_pretrained_weights.h5")
    
    # # Download COCO trained weights from Releases if needed
    # if not os.path.exists(COCO_MODEL_PATH):
    #     utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    # IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    InIMAGE_DIR = "/home/ashwin/Desktop/Test/input"
    OutIMAGE_DIR = "/home/ashwin/Desktop/Test/output"
    if not os.path.isdir(OutIMAGE_DIR):
        os.makedirs(OutIMAGE_DIR)

    if args.size == 'large':
        Image.MAX_IMAGE_PIXELS = 1600 * 1600 * 10 * 10
        img_name = os.listdir(InIMAGE_DIR)[0]
        img = os.path.join(InIMAGE_DIR, img_name)
        num_tiles = 64
        print("Slicing image into {} slices".format(num_tiles))
        tiles = image_slicer.slice(img, num_tiles)

    ## Configurations

    class InferenceConfig(sate.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    ## Create Model and Load Trained Weights

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    model_json = model.KM.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    print("Loading weights from:",COCO_MODEL_PATH)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)#, exclude=["mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask"])



    ## Class Names
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    # class_names = ['Plane', 'Ships', 'Running_Track', 'Helicopter', 'Vehicles', 'Storage_Tanks', 'Tennis_Court',
    #                'Basketball_Court', 'Bridge', 'Roundabout', 'Soccer_Field', 'Swimming_Pool', 'baseball_diamond',
    #                'Buildings', 'Road', 'Tree', 'People', 'Hangar', 'Parking_Lot', 'Airport', 'Motorcycles', 'Flag',
    #                'Sports_Stadium', 'Rail_(for_train)', 'Satellite_Dish', 'Port', 'Telephone_Pole',
    #                'Intersection/Crossroads', 'Shipping_Container_Lot', 'Pier', 'Crane', 'Train', 'Tanks', 'Comms_Tower',
    #                'Cricket_Pitch', 'Submarine', 'Radar', 'Horse_Track', 'Hovercraft', 'Missiles', 'Artillery',
    #                'Racing_Track', 'Vehicle_Sheds', 'Fire_Station', 'Power_Station', 'Refinery', 'Mosques', 'Helipads',
    #                'Shipping_Containers', 'Runway', 'Prison', 'Market/Bazaar', 'Police_Station', 'Quarry', 'School',
    #                'Graveyard', 'Well', 'Rifle_Range', 'Farm', 'Train_Station', 'Crossing_Point', 'Telephone_Line',
    #                'Vehicle_Control_Point', 'Warehouse', 'Body_Of_water', 'Hospital', 'Playground', 'Solar_Panel']

    class_names = ['BG','building']

    # model_filename = "MRCNN_spacemodel.pkl"

    # pickle.dump(model,open(model_filename, 'wb'))

    if args.size == 'large':
        for tile in tiles:
            image = skimage.io.imread(tile.filename)
            # remove alpha channel
            image = image[:, :, :3]
    
            results = model.detect([image], verbose=1)
            r = results[0]
        
            fig = visualize.save_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    
            plt.imsave(tile.filename,fig)
            tile.image = Image.open(tile.filename)

        image = join(tiles)
        image.save(os.path.join(OutIMAGE_DIR,img_name))


    if args.size == 'small':        
        gt_flag = 0

        # TODO: If any tif files are present, convert them to jpg 

        ## Run Object Detection
        filenames = glob.glob(os.path.join(InIMAGE_DIR, "*.jpg"))
        with open('./coco/annotations_batch1/train.json') as f: # Not needed for inference
            gj = json.load(f)
        for counter, fl in enumerate(filenames):
            print("counter = {:5d}".format(counter))
            image_name = fl.split('/')[-1]
            output_path = os.path.join(OutIMAGE_DIR, image_name)
            if os.path.isfile(output_path):
                continue

            # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
            image = skimage.io.imread(fl)
            # remove alpha channel
            image = image[:, :, :3]

            results = model.detect([image], verbose=1)


            r = results[0]
            # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

        #####################################################################################################
            if gt_flag == 1:
                bbox_actual = []


                print("Image ID:",image_name[:-4])
                for ind in gj['annotations']:
                    if ind['image_id'] == image_name[:-4]:
                        bbox_actual.append(ind['bbox'])
            
        #print("Number of proposals =",len(r['rois']))
        #print("Number of ground truth labels =",len(bbox_actual))

        #print("Proposal1= ",r['rois'][0])
        #print("ActualBBox1= ",bbox_actual[0])

        # Ideally it should be a square matrix with high IOU values in the diagonal
        #iou_mat = bbox_iou(r['rois'],bbox_actual,bb_format="XYWH")
        #print(iou_mat)

        #plt.figure()

        #plt.imshow(iou_mat, cmap="hot",interpolation="nearest")
        #plt.show()
        
        #####################################################################################################


            fig = visualize.save_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        # fig.savefig(output_path, bbox_inches='tight', pad_inches=0)

            print("Number of detected buildings =",len(r['rois']))
        
            if gt_flag == 1:
                print("Number of ground truth buildings",len(bbox_actual))



        # print(fig.shape)
            plt.imsave(output_path, fig)
        # fig.imsave(output_path)


if __name__ == "__main__":
    main()


