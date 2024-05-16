# suppress the warnings
import warnings
warnings.filterwarnings("ignore")


# import packages
import os
import sys
import wandb
import argparse
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


# our code
sys.path.append("..")
from src.detectron2_classes.CustomTrainer import CustomTrainer


def set_config(args, train_dicts, train_metadata):
    
    # get the paths
    root_folder = args.root_folder
    models_folder = args.models_folder
    device = args.device
    
    # define the metadata
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(root_folder, "detectron2", "configs", "COCO-Detection","faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(train_metadata.get("thing_classes"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.DEVICE=device
    cfg.SOLVER.CHECKPOINT_PERIOD = 0 # How often to save a checkpoint while training, 0 to save best model based on minimum val loss
    
    # define the config for testing
    iter_per_epoch = len(train_dicts) // cfg.SOLVER.IMS_PER_BATCH
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 20000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.TEST.DETECTIONS_PER_IMAGE = 512
    cfg.TEST.EVAL_PERIOD = iter_per_epoch 
    
    # define the lr scheduler
    max_epochs = args.epochs
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = max_epochs * iter_per_epoch
    cfg.SOLVER.STEPS = (int(max_epochs * iter_per_epoch * 0.8), int(max_epochs * iter_per_epoch * 0.9), )
    
    # define the model parameters
    cfg.MODEL.RPN.NMS_THRESH = .3
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    
    # define custom data augmentation
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = (1080.0, 1080.0)
    cfg.INPUT.AUG_RESIZE = False
    cfg.INPUT.AUG_VER_FLIP = False
    cfg.INPUT.AUG_HOR_FLIP = True
    cfg.INPUT.AUG_ON_TEST = False
    cfg.INPUT.AUG_CONTRAST = True
    cfg.INPUT.AUG_CONTRAST_RANGE = (.5, 1.5)
    cfg.INPUT.AUG_BRIGHTNESS = True
    cfg.INPUT.AUG_BRIGHTNESS_RANGE = (.3, 1.7)
    cfg.INPUT.AUG_SATURATION = True
    cfg.INPUT.AUG_SATURATION_RANGE = (.7, 1.4)
    
    # Save the config file for future reference
    cfg.OUTPUT_DIR = os.path.join(models_folder, args.run_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    yaml_file = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(yaml_file, 'w') as file:
        file.write(cfg.dump())
    return cfg
 

def set_wandb(args):
    if args.key is not None:
        wandb.init(id = wandb.util.generate_id(), resume = "allow", project=args.project, entity=args.entity, settings=wandb.Settings(code_dir="."))
        wandb.run.name = args.run_name
        wandb.config.update(args)
        return True
    else:
        return False
   
   
def train_model(args):

    # prepare the training dataset
    train_json = os.path.join(args.annotations_folder, "train.json")
    register_coco_instances("train", {}, train_json, args.images_folder)
    train_dicts = DatasetCatalog.get("train")
    train_metadata = MetadataCatalog.get("train")
    
    # prepare the validation dataset
    val_json = os.path.join(args.annotations_folder, "val.json") 
    register_coco_instances("val", {}, val_json, args.images_folder)
    val_dicts = DatasetCatalog.get("val")
    val_metadata = MetadataCatalog.get("val")
    
    # print the dataset information
    assert train_metadata.get('thing_classes') == val_metadata.get('thing_classes'), "The classes in the training and validation datasets do not match"
    print(f"Therer are {len(train_dicts)} training images and {len(val_dicts)} validation images")
    print(f"Annotated classes: {train_metadata.get('thing_classes')}")
    
    # set the configuration
    cfg = set_config(args, train_dicts, train_metadata)
    
    # train the model
    use_wandb = set_wandb(args)
    trainer = CustomTrainer(cfg, use_wandb)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print("All done!")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # wandb settings
    parser.add_argument("--entity", help="Name the wandb entity", default="whuang288", type=str)
    parser.add_argument("--project", help="Name the wandb project", default="drone_project_train", type=str)
    parser.add_argument("--run_name", help="Name the detectron2 model", default=None, type=str)
    parser.add_argument("--key", help="Wandb login key", default=None, type=str)
    
    # path settings
    parser.add_argument("--root_folder", help="Root directory of the project", default="../", type=str)
    parser.add_argument("--annotations_folder", help="Folder containing the annotations", default="/data/huanga/Synchrony/annotations", type=str)
    parser.add_argument("--images_folder", help="Folder containing the images", default="/data/huanga/Synchrony/annotations/annotated_images", type=str)
    parser.add_argument("--models_folder", help="Folder containing the models", default="/data/huanga/Synchrony/models", type=str)
    
    # training settings
    parser.add_argument("--device", help="Device to train the model on", default="cuda", type=str)
    parser.add_argument("--lr", help="Learning rate", default=0.007372, type=float)
    parser.add_argument("--epochs", help="Number of epochs to train the model", default=300, type=int)
    
    # parse the arguments
    args = parser.parse_args()
    
    # train the model
    train_model(args)