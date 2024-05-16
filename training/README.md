## Training

The code in this folder can be used to train a Faster R-CNN model with the annotations you generated from Step 1. Skip this part if you are only interested in using the <a href="https://drive.google.com/drive/folders/1-4e4OFroElRJWsfvat0vwKg6IGRk9BHP"> checkpoints </a> we provide.

1. Start training the model with [train_detection.py](train_detection.py)

```sh 
python train_detection.py --run_name name_your_model
```

2. Evaluate the trained model with [eval_detection.ipynb](evaluate_detection.ipynb)

- This notebook can be used to evaluate the model you trained. It reports the standard <a href="https://www.picsellia.com/post/coco-evaluation-metrics-explained"> COCO evaluation metrics </a> for object detection models. You can also play around with the IOU thresholds to get a desired precision and recall.

3. Visualize results with [visualize_detection.ipynb](visualize_detection.ipynb)

- This notebook can be used to do a visual sanity check on how good the model is doing in detecting marching band members.
