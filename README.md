
# Measuring Large Group Synchrony and Social Connection with Machine Learning and Computer Vision

## Introduction

This repo implements the code for measuring synchrony in a large naturalistic group for the paper "Measuring Large Group Synchrony and Social Connection with Machine Learning and Computer Vision". We applied <a href="https://arxiv.org/abs/1506.01497">Faster R-CNN</a> to drone footage to detect and track marching band members movements during a week of practice. We then identified frames of footage where the band was doing the same movements at the beginning of the week (time 1) and at the end of a week of practice (time 2) to measure changes in synchronous marching from the beginning and end of learning a new routine. We measured their marching steps and positions against ground truth drill videos from band directors to determine whether band members more closely match ground truth after a week of practice. This was our measure of synchrony. We found that band members became significantly more synchronous--that is, more aligned with ground truth drill videos of the whole band and their smaller rank in the band after a week of practice. In the paper, we describe our findings that synchronous marching is associated with increases in social connection within subgroups involved in a synchronous ritual (band and rank in the band) but not the highest-level group that the band represents (university). 

We provide this code in the case that other researchers would like to measure the movements of large naturalistic social groups from a birds-eye view recording such as in religious rituals, military behavior, protest etc. This repo provides code for how to use a machine learning and computer vision pipeline to identify humans in videos and extract their x and y coordinate positions. We do this by acquiring overhead footage, annotating the footage on <a href="https://labelbox.com">Labelbox</a>, creating a training and validation set to train a machine learning model to detect people in the footage rather than extraneous objects or features like trees, backpacks, animals etc. We then feed the video footage to the model and extract detections for all frames of the footage. In our use case, we also annotated ground truth drill videos and compared the coordinate data from the band members to the ground truth. 


## Installation

* Follow [INSTALL.md](INSTALL.md) for installing necessary dependencies.

## Steps:

1. Follow the steps in the [preparing folder](preparing/README.md) to prepare the annotations for training an Object Detection model. Skip this part if you already have COCO format annotations available or if you are only interested in using the <a href="https://drive.google.com/drive/folders/1-4e4OFroElRJWsfvat0vwKg6IGRk9BHP"> annotations </a> we provide.

2. Follow the steps in the [training folder](training/README.md) to train a Faster R-CNN model with the annotations you generated from Step 1. Skip this part if you are only interested in using the <a href="https://drive.google.com/drive/folders/1-4e4OFroElRJWsfvat0vwKg6IGRk9BHP"> checkpoints</a> we provide.

3. Follow the steps in the [inferencing folder](inferencing/README.md) to draw bounding boxes and calculate the coordinates of marching band members on the field using the checkpoints generated in Step 2.

4. Follow the steps in the [matching folder](matching/README.md) to match the "fomation coordinates" in the ground truth drill videos to the "detected coordinates" generated in Step 3.

## Acknowledgement

This directory is inspired by <a href = "https://github.com/benkoger/overhead-video-worked-examples"> Koger's  directory</a>.

Koger, B., Deshpande, A., Kerby, J.T., Graving, J.M., Costelloe, B.R., Couzin, I.D. Multi-animal behavioral tracking and environmental reconstruction using drones and computer vision in the wild.
