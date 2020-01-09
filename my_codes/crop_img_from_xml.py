from lxml import etree
import tensorflow as tf
import PIL
import numpy as np
import os
from datetime import datetime

def recursive_parse_xml_to_dict(xml):
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def crop_img_from_xml(img_path, xml_path):
    img = np.array(PIL.Image.open(img_path))
    with tf.io.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = recursive_parse_xml_to_dict(xml)['annotation']['object']
    for i in data:
        name = i['name']
        xmin, ymin, xmax, ymax = map(int, list(i['bndbox'].values()))
        crop = img[ymin:ymax, xmin:xmax]
        os.makedirs(name, exist_ok=True)
        PIL.Image.fromarray(crop).save(f'{name}/{str(datetime.now())}.jpg')
