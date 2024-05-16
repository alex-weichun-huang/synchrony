# source: https://github.com/benkoger/overhead-video-worked-examples/blob/main/functions/labelbox_processing.py

import os
import cv2
import json
import shutil
import requests
import numpy as np


def save_frame(cap, frame_num, outfile, crop_size=None, top_left=None):
    """ Save frame from cv2 VideoCapture
    
    Args: 
        cap: cv2.VideoCapture object
        frame_num: the frame number to save
        outfile: where to save frame
        crop_size: pixels, size of crop (square). If None, no crop.
        top_left: (i, j) coordinate of top left corner of crop (if not None)
            if None and crop_size is not None, then choose random values
            
    Return crop_top_left
        
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if frame is not None:
        if crop_size:
            if not top_left:
                raise ValueError(f"If cropping, must provide top_left: {top_left}")
            top, left = top_left
            frame = frame[top:top+crop_size, left:left+crop_size]

        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        cv2.imwrite(outfile, frame)
    else:
        print(f"with frame to be saved at outfile {outfile}.")
        

def random_top_left(im_shape, crop_size, gaussian=False):
    """ Get a random top left coordinate for a crop of size (crop_size * crop_size).
    
    Args:
        im_shape: (h, w, ...)
        crop_size: size of ultimate crop
        gaussian: If True, then pull coordinates from gaussian with mean
            at the center of possible range of top left values and 1 standard 
            deviation to the min and max top left values
    
    Returns [top, left]
    """
    
    height, width = im_shape[:2]
    if gaussian:
        mean_top = (width-crop_size) / 2
        mean_left = (width-crop_size) / 2
        top = -1
        left = -1
        while ((top >= (height-crop_size)) or (top < 0)):
            top = int(np.random.normal(mean_top, mean_top))
        while ((left >= (width-crop_size)) or (left < 0)):
            left = int(np.random.normal(mean_left, mean_left))
    else:
        top = np.random.randint(0, height-crop_size)
        left = np.random.randint(0, width-crop_size)
    top_left = [top, left]
    return top_left


def get_bbox(contour):
    """ Get bounding box of contour.
    
    Args:
        contour: opencv contour object
        
    Returns:
        Upper left corner x, y and bbox width and height as list
    """
    
    upper_left = np.min(contour, 0)
    bottom_right = np.max(contour, 0)
    bbox_xy = (bottom_right - upper_left)
    
    return [int(upper_left[0]), int(upper_left[1]), int(bbox_xy[0]), int(bbox_xy[1])]


class COCOJson:
    
    def __init__(self, description, date_created):
        """ Info to create coco.json annotation file. 
    
        Args:
            description: string, description of dataset
            date_created: string, date dataset created

        """
        self.coco_dict = {}
        self.create_boilerplate(description, date_created)
        
        
    def create_boilerplate(self, description='', date_created=''):
        """ Creates the generic peices of coco annotation and returns dict.
        
        Args:
            description: string, general description of the dataset
            date_created: string, date the dataset is created
        """
        

        self.coco_dict['info'] = []
        self.coco_dict['info'].append({
            'description': description,
            'url': '',
            'version': '1.0',
            'year': 2020,
            'contributor': "Alex Huang",
            'date_created': date_created    
        })

        self.coco_dict['licenses'] = []
        self.coco_dict['licenses'].append({
            'url': '',
            'id': 0,
            'name': '',  
        })
        
        self.coco_dict['images'] = []
        self.coco_dict['annotations'] = []
        self.coco_dict['categories'] = []
        
        
    def add_image(self, file_name, image_shape, image_id):
        """ Add image to coco dict.
        
        Args:
            file_name: generally basename of full path to image
            image_shape: shape of the image (height, width)
            image_id: int, should be unique for each image in coco dataset (starting at 1)
            
        """

        self.coco_dict['images'].append({
            'license': 0,
            'file_name': file_name,
            'coco_url': '',
            'height': image_shape[0],
            'width': image_shape[1],
            'date_captured': '',
            'flickr_url': '',
            'id': image_id
            })
    
    def add_category(self, name, category_id, supercategory=''):
        """ Add new category to dataset.
        
        Args:
            name: string, name of new category
            category_id: category id, should be unique for each 
                category in dataset (starting at 1)
            supercategory: string, supercategory, if any
        
        """
        
        self.coco_dict['categories'].append({
            'supercategory': supercategory,
            'id': category_id,
            'name': name
            })
        
    def add_annotation_from_bounding_box(self, bbox, image_id, annotation_id, category_id):
        """ Add new annotation from bounding box to dataset.
        
        Args:
            bbox: [upper_left_x, upper_left_y, bbox_height, bbox_width]
            image_id: id of the image the annotation came from
            annotation_id: id number of this annotation, should 
                be unique for each annotation in dataset (from 1)
            category_id: category id, should be unique for each 
                category in dataset (starting at 1)
        """
        
        segmentation = [bbox[0], bbox[1],
                        bbox[0] + bbox[2], bbox[1],
                        bbox[0] + bbox[2], bbox[1] + bbox[3],
                        bbox[0], bbox[1] + bbox[3]]
        
        self.coco_dict['annotations'].append({
                'segmentation': [segmentation],
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': bbox,
                'category_id': category_id,
                'id': annotation_id
            })
        
        
    
        
    def add_annotation_from_contour(self, contour, image_id, annotation_id, category_id=1):
        """ Add annotation from openCV contour object.
        
        Args:
            contour: openCV contour object
            image_id: id of the image the contour came from
            annotation_id: id number of this annotation, should 
                be unique for each annotation in dataset (from 1)
        
        """
        segmentation = list(contour.reshape(-1))
        segmentation = [int(val) for val in segmentation]
        
        if len(segmentation) % 2 == 0 and len(segmentation) >= 6:
            # must have at least three points each with an x and a y
            self.coco_dict['annotations'].append({
                'segmentation': [segmentation],
                'area': cv2.contourArea(contour),
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': get_bbox(contour),
                'category_id': category_id,
                'id': annotation_id
            })
        else:
            print("removing invalid segmentation...")
        
    def _get_contours_from_binary(self, binary_image):
        """ Get all contours from binary image.
        
        Args:
            binary_image: 2d array of zeros and ones"""
        
        contours, hierarchy = cv2.findContours(binary_image.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [np.squeeze(contour) for contour in contours]
        
        return contours, hierarchy    
    
        
    def add_annotations_from_binary_mask(self, binary_image, image_id, current_annotation_id, category_id):
        """ Add every contour in binary image as seperate annotation.
        
        Args:
            binary_image: 2d array of ones and zeros
            image_id: id of the image that this binary image is the label for
            current_annotation_id: the next annotation id value to use in the dataset
            category_id: the annotation category of the binary mask
        """
        
        contours, hierarchy = self._get_contours_from_binary(binary_image)
        
        if not contours:
            return current_annotation_id
        
        for contour, h in zip(contours, hierarchy[0]):
            if len(contour.shape) < 2:
                continue
            self.add_annotation_from_contour(contour, image_id, current_annotation_id, category_id)
            current_annotation_id += 1
        
        return current_annotation_id
        
        
        
        
    def copy_image_to_image_folder(self, outfolder, file):
        """ Copy image to an annotated images folder.
        
        Args:
            outfolder: path of folder to copy images to
            file: existing file of image that should be copied"""
        
        shutil.copy(file, outfolder)
        
        
    def write_json_file(self, outfile):
        """ Write the information in coco_dict to json file.
        
        Args:
            outfile: string, file that the json file will be written to
            
        """
        
        with open(outfile, 'w') as outfile:
            json.dump(self.coco_dict, outfile, indent=4)
            
    def add_keypoint(self, x, y, image_id, annotation_id, category_id=1):
        """ Add keypoint annotation.
        
        Args:
            x: keypoint x value
            y: keypoint y value
            image_id: id of the image the contour came from
            annotation_id: id number of this annotation, should 
                be unique for each annotation in dataset (from 1)
            category_id: id of the annotation category
        
        """

        self.coco_dict['annotations'].append({
            'segmentation': [[]],
            'area': [],
            'iscrowd': 0,
            'keypoints': [x, y, 2],
            'num_keypoints': 1,
            'image_id': image_id,
            'bbox': [],
            'category_id': category_id,
            'id': annotation_id
        })


def save_labelbox_image(im_info, save_folder, overwrite=False):
    """ Save image located at adress linked to key 'Labeled Data' 
    
    Args:
        im_info: information for single annotated image in labelbox json
        save_folder: full path to folder where images should be saved
        overwrite: if True, save new image even if file already exists
        
    Returns:
        path to where image is saved
    """
    
    im_name = os.path.splitext(im_info["External ID"])[0] + ".jpg"
    image_outfile = os.path.join(save_folder, im_name)
    if os.path.exists(image_outfile):
        if not overwrite:
            return im_name
    im_bytes = requests.get(im_info['Labeled Data']).content
    im_raw = np.frombuffer(im_bytes, np.uint8)
    im = cv2.imdecode(im_raw, cv2.IMREAD_COLOR)

    cv2.imwrite(image_outfile, im)

    return im_name


def get_classes_in_json(labelbox_json, custom_class_reader=None):
    """ Get list of all unique values in labelbox json annotation.
    
    Here, classes are defined as each objects 'value.' 
    
    Args:
        labelbox_json: json file exported from labelbox
        custom_class_reader: if want to use a function to get object class
            other than annotation['value']
        
    Return list of classes
    """
    classes = set()
    for im_id, im_info in enumerate(labelbox_json):
        if im_info['Skipped']:
            continue
        for object_ann in im_info['Label']['objects']:
            if custom_class_reader:
                classes.add(custom_class_reader(object_ann))
            else:
                classes.add(object_ann['value'])
            
    return sorted(list(classes))


def labelbox_to_coco(labelbox_json_file, coco_json_file,
                     images_folder, description=None, date=None,
                     overwrite=False, verbose=False, custom_class_reader=None):
    
    """ Use labelbox json to create and save coco json and
        save corresponding annotated images.
        
        Currently just for images annotated with bounding boxes.
        
    Args:
        labelbox_json_file: path to json exported from labelbox
        coco_json_file: path to file where new coco json should
            be saved
        images_folder: path to folder were images used in labelbox
            annotations will be saved
        description: description of dataset that will be saved at
            coco_json['info']['description']
        date: date that will be saved at coco_json['info']['date']
        overwrite: if True overwrite existing image files
        verbose: if True print info like dataset classes present
        custom_class_reader: if want to use a function to get object class
            other than annotation['value']
        """
    
    coco = COCOJson(description, date)
    
    f = open(labelbox_json_file)
    labelbox_json = json.load(f)
    all_classes = get_classes_in_json(labelbox_json, custom_class_reader)
    
    # Maps class names to class ids
    label_dict = {} 
    for class_num, class_name in enumerate(all_classes):
        if verbose:
            print(class_num+1, class_name)
        coco.add_category(class_name, class_num+1)
        label_dict[class_name] = class_num + 1

    annotation_id = 1
    image_id = 1
    for im_info in labelbox_json:
        if im_info['Skipped']:
            continue
        image_name = save_labelbox_image(im_info, images_folder,
                                         overwrite=overwrite
                                        )
        image = cv2.imread(os.path.join(images_folder, image_name))
        coco.add_image(image_name, image.shape[:2], image_id)

        for annotation in im_info['Label']['objects']:
            bbox = annotation['bbox'] # ['top', 'left', 'height', 'width']
            coco_bbox = [bbox['left'], bbox['top'], bbox['width'], bbox['height']]
            if custom_class_reader:
                class_name = custom_class_reader(annotation)
            else:
                class_name = annotation['value']
            category_id = label_dict[class_name]
            coco.add_annotation_from_bounding_box(coco_bbox, image_id, 
                                                  annotation_id, category_id
                                                 )
            annotation_id += 1
        image_id += 1

    coco.write_json_file(coco_json_file)
    if verbose:
        print(f"saving at {coco_json_file}")
        print(f"{annotation_id-1} annotations from {image_id-1} images saved.")