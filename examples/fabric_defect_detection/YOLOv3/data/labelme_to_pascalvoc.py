import os, json
from lxml.etree import Element, SubElement, tostring, ElementTree


class Labelme_to_PascalVOC(object):
    def __init__(self): pass
    
    
if __name__ == "__main__": 
    node_root = Element('annotation')
     
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'GTSDB'
     
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = '000001.jpg'
     
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '500'
     
    node_height = SubElement(node_size, 'height')
    node_height.text = '375'
     
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
     
    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = 'mouse'
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = '99'
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = '358'
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = '135'
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = '375'
    
    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = 'mouse'
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = '98'
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = '357'
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = '136'
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = '376'

    tree = ElementTree(node_root)
    tree.write('test.xml', pretty_print=True, xml_declaration=False, encoding='utf-8')
    print("Done")

