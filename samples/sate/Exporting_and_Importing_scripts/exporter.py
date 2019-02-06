import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import imageio
import tensorflow as tf
from tensorflow.python.framework import graph_util
from keras import backend as K

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from inference_config import InferenceConfig
import mrcnn.model as modellib
import mrcnn.utils as utils
import mrcnn.visualize as visualize
import matplotlib.pyplot as plt
import math


# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
# Directory to trained model
train_log_dirpath = os.getcwd()#os.path.join(ROOT_DIR, "logs")
print(train_log_dirpath)
model_filepath = os.path.join(ROOT_DIR, train_log_dirpath,"mask_rcnn_sate_spaceaug_0045.h5")#None# os.path.join(ROOT_DIR, "logs","trained_model","my_model.h5")
print(model_filepath)
# name of the pb file we want to output
filename = 'mrcnn_model4.pb'



# Display Inference Config
inference_config = InferenceConfig()
inference_config.display()


# Build the inference model
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=train_log_dirpath)
# Get path to saved weights. Either set a specific path or find last trained weights
model_filepath = model_filepath if model_filepath else model.find_last()[1]
#print(model_filepath)
# Load trained weights (fill in path to trained weights here)
assert model_filepath, "Provide path to trained weights"
model.load_weights(model_filepath, by_name=True)
print("Model loaded.")


# Get keras model and save
model_keras= model.keras_model
# All new operations will be in test mode from now on.
K.set_learning_phase(0)
# Create output layer with customized names
num_output = 7
pred_node_names = ["detections", "mrcnn_class", "mrcnn_bbox", "mrcnn_mask",
                       "rois", "rpn_class", "rpn_bbox"]
pred_node_names = ["output_" + name for name in pred_node_names]


pred = [tf.identity(model_keras.outputs[i], name = pred_node_names[i])
        for i in range(num_output)]
sess = K.get_session()
# Get the object detection graph
od_graph_def = graph_util.convert_variables_to_constants(sess,
                                                         sess.graph.as_graph_def(),
                                                         pred_node_names)

model_dirpath = os.path.dirname(model_filepath)
pb_filepath = os.path.join(model_dirpath, filename)
print('Saving frozen graph {} ...'.format(os.path.basename(pb_filepath)))

frozen_graph_path = pb_filepath
with tf.gfile.GFile(frozen_graph_path, 'wb') as f:
    f.write(od_graph_def.SerializeToString())
print('{} ops in the frozen graph.'.format(len(od_graph_def.node)))