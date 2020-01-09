import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
%matplotlib inline

#@title helper functions
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, sess):
  global image_tensor, detection_boxes, detection_scores, detection_classes, num_detections
  with sess.as_default():
      image_np = load_image_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np, axis=0)

      # image_np_expanded = image
 
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      output_dict = {
          'detection_boxes':boxes,
          'detection_scores':scores,
          'detection_classes':classes,
          'num_detections':num
      }
  return output_dict

def visualize(image_np, output_dict, category_index, min_score_thresh=0.1,
              skip_scores=False, skip_labels=False):
  vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(output_dict['detection_boxes']),
            np.squeeze(output_dict['detection_classes']).astype(np.int32),
            np.squeeze(output_dict['detection_scores']),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2, 
            max_boxes_to_draw=None,
            min_score_thresh=min_score_thresh,
            skip_scores=skip_scores,
            skip_labels=skip_labels)
  return image_np

def drawBox(image, box):
  image_np = image.copy()
  vis_util.draw_bounding_box_on_image(image_np, box[0], box[1], box[2], box[3], thickness=2)
  return image_np

def drawBoxes(image, box):
  image_np = image.copy()
  vis_util.draw_bounding_boxes_on_image(image_np, box, thickness=2)
  return image_np

def drawBoxesFromArray(image_np, boxes):
  image_np = image_np.copy()
  for box in boxes:
      vis_util.draw_bounding_boxes_on_image(image_np, box, thickness=2,
                                            use_normalized_coordinates=False)
  return image_np


#################################################
PATH_TO_CKPT = '/content/ssd-resnet50-fpn-02/fine_tuned_model/' + 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/content/ssd-resnet50-fpn-02/', 'label_map.pbtxt')

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

sess = tf.Session(graph=detection_graph)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
###################################################
image_path = "/content/ssd-resnet50-fpn-02/data_v1/train_data/images/001.jpg"

image = Image.open(image_path)
image_np = load_image_into_numpy_array(image)
image_np_expanded = np.expand_dims(image_np, axis=0)
output_dict = run_inference_for_single_image(image, sess)
visualize(image_np, output_dict, category_index, skip_labels=True, min_score_thresh=0.5)
plt.imshow(image_np)
#####################################################
