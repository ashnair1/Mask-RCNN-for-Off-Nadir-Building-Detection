import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
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



# Display Inference Config
inference_config = InferenceConfig()
inference_config.display()



argparser = argparse.ArgumentParser(
    description='Import a model for inference')

argparser.add_argument('--weight_path',
                       help='Location of the weight file',
                       type=str, default= '/home/ashwin/Desktop/SpaceNet/Mask_RCNN/samples/sate/mask_rcnn_sate_spaceaug_0045.h5')

argparser.add_argument('--model_name', help='Name of the .pb file',
                       type=str, default= 'mrcnn_model.pb')


def main(args):

	model_filepath = args.weight_path
	filename = args.model_name

	train_log_dirpath = os.path.abspath(model_filepath+"../..")
	

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


if __name__ == '__main__':
	args = argparser.parse_args()
	main(args)