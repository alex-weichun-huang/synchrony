# Preparing

The code in this folder can be used to prepare the annotations for training an Object Detection model. Skip this part if you already have COCO format annotations available.

1. Extract Frames from drone footage with [videos_to_frames.ipynb](videos_to_frames.ipynb)

- This notebook can be used to extract frames from raw drone footage in a systematic way so that the annotations for the training data are rich and diverse enough to train a robust object detection model.

2. Annotate Frames using LabelBox

-  To learn more about how to annotate images through Labelbox, please visit their official <a href="https://labelbox.com/product/annotate/"> website </a>.  

3. Convert Labelbox exports to COCO format annotations with [exports_to_coco.ipynb](exports_to_coco.ipynb)

- This notebook can be used to convert Labelbox export files to COCO format annotations. To understand what COCO format is, please visit  <a href="https://cocodataset.org/#format-data">  this page </a>.

4. Split all the annotataions into train and validation datasets with [coco_to_annotations.ipynb](coco_to_annotations.ipynb)

- This notebook can be used to combine all the COCO format annotations you have into one file, and then split them into a train set and a test set for model training and evaluation.
