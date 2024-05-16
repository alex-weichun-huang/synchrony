# Modify Dataset mapper for customed data augmentation

import copy
import torch
import logging
import numpy as np
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

class CustomDatasetMapper():
    """ Custom DatasetMapper.
    
    Args:
        cfg: Detectron2 config. file
        is_train: is the mapper being used for training
        calc_val_loss: same conditions as is_train=False but keeps annotations
    """
    
    def __init__(self, cfg, is_train=True, calc_val_loss=False):
        
        self.is_train = is_train
        self.calc_val_loss = calc_val_loss
        self.aug_on_test = cfg.INPUT.AUG_ON_TEST
        
        # Adding random crop augmentation
        if cfg.INPUT.CROP.ENABLED and self.is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger("detectron2").info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None
        
        # Adding random resize augmentation
        if cfg.INPUT.AUG_RESIZE and self.is_train:
            self.resize_gen = [T.Resize(cfg.INPUT.AUG_RESIZE_SHAPE)]
        else:
            self.resize_gen = None
        
        # Adding random data augmentation       
        if self.is_train or self.aug_on_test:
            self.tfm_gens = []
            if self.is_train:
                if cfg.INPUT.AUG_VER_FLIP:
                    self.tfm_gens.append(T.RandomFlip(vertical=True, horizontal=False))
                if cfg.INPUT.AUG_HOR_FLIP:
                    self.tfm_gens.append(T.RandomFlip(vertical=False, horizontal=True))
                          
            # On test, can only modify things that don't require changed annotations
            if cfg.INPUT.AUG_CONTRAST:
                self.tfm_gens.append(T.RandomContrast(*cfg.INPUT.AUG_CONTRAST_RANGE))
            if cfg.INPUT.AUG_BRIGHTNESS:
                self.tfm_gens.append(T.RandomBrightness(*cfg.INPUT.AUG_BRIGHTNESS_RANGE))
            if cfg.INPUT.AUG_SATURATION:
                self.tfm_gens.append(T.RandomSaturation(*cfg.INPUT.AUG_SATURATION_RANGE))
        else:
            self.tfm_gens = None
            
        logger = logging.getLogger(__name__)
        logger.info("Transforms used in training: " + str(self.tfm_gens))

        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

    def _apply_image_augmentations(self, image, transforms, dataset_dict):
        """ Apply augementations to image.
        
        Args:
            image: image array
            transforms: None or already applied transforms
            dataset_dict: detectron2 dataset_dict
            
        Returns modified image, used transforms
        """
        # Apply cropping
        if self.crop_gen:
            crop_tfm = utils.gen_crop_transform_with_instance(
                self.crop_gen.get_crop_size(image.shape[:2]),
                image.shape[:2],
                np.random.choice(dataset_dict["annotations"]),
            )
            image = crop_tfm.apply_image(image)
            if transforms:
                transforms += crop_tfm
            else:
                transforms = crop_tfm
        
        # Apply resizing      
        if self.resize_gen:
            image, resize_tfm = T.apply_transform_gens(self.resize_gen, image)
            if transforms:
                transforms += resize_tfm
            else:
                transforms = resize_tfm
                
        image = np.array(image) # to make writable array     
        if self.tfm_gens:
            image, gen_transforms = T.apply_transform_gens(self.tfm_gens, image)
            if transforms:
                transforms += gen_transforms
            else:
                transforms = gen_transforms
                
        return image, transforms
    
    def _apply_annotation_augmentations(self, image, transforms, dataset_dict):
        """ Remove unneeded annotations and augment/clean those that remain.
        
        Args:
            image: image array
            transforms: None or already applied transforms
            dataset_dict: detectron2 dataset_dict
        Return dataset_dict
        """
        
        if not self.is_train and not self.calc_val_loss:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            if not self.keypoint_on:
                anno.pop("keypoints", None)

        if transforms:
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image.shape[:2], 
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
        else:
            annos = [obj for obj in dataset_dict.pop("annotations")]

        instances = utils.annotations_to_instances(
            annos, image.shape[:2], mask_format=self.mask_format
        )

        # Create a tight bounding box from masks, useful when image is cropped
        if self.crop_gen and instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
        return dataset_dict
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict) 
        image = utils.read_image(dataset_dict["file_name"], format = self.img_format)
        utils.check_image_size(dataset_dict, image)

        transforms = None
        if self.is_train or self.aug_on_test:
            if "annotations" not in dataset_dict:
                assert False, "Must have annotations in dataset"
            image, transforms = self._apply_image_augmentations(image, transforms, dataset_dict)
        dataset_dict = self._apply_annotation_augmentations(image, transforms, dataset_dict)
        
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        ).contiguous()
        
        return dataset_dict
        