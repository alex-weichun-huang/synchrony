
# Measuring Large Group Synchrony and Social Connection with Machine Learning and Computer Vision

## Introduction

This repo implements the code for paper "Measuring Large Group Synchrony and Social Connection with Machine Learning and Computer Vision". We applied <a href="https://arxiv.org/abs/1506.01497">Faster R-CNN </a> to drone footage to detect and track marching band members during a week of practice. We then have marching band members identify moments where the band was doing the same movements at the beginning of the week (time 1) and at the end of the week (time 2) to compare how synchronized they were marching before and after practicing together. We also measured their synchrony against ground truth drill videos from band directors and showed that synchrony is associated with increases in social connection within subgroups involved in a synchronous ritual but not the highest-level group (university). 

## Installation

* Follow [INSTALL.md](INSTALL.md) for installing necessary dependencies.

## Steps:

1. Follow the steps in the [preparing folder](preparing/README.md) to prepare the annotations for training a Object Detection model. Skip this part if you already have COCO format annotations available or if you are only interested in using the <a href="https://drive.google.com/drive/folders/1-4e4OFroElRJWsfvat0vwKg6IGRk9BHP"> annotations </a> we provide.

2. Follow the steps in the [training folder](training/README.md) to train a Faster R-CNN model with the annotations you generated from Step 1. Skip this part if you are only interested in using the <a href="https://drive.google.com/drive/folders/1-4e4OFroElRJWsfvat0vwKg6IGRk9BHP"> checkpoints </a> we provide.

3. Follow the steps in the [inferencing folder](inferencing/README.md) to draw bounding boxes and calculate the coordinate of marching band members on the field using the checkpoint generated in Step 2.

4. Follow the steps in the [matching folder](matching/README.md) to match the "fomation coordiantes" in the ground truth drill videos to the "detected coordinates" generated in Step 3.

## Acknowledgement

This directory is built on top of <a href = "https://github.com/benkoger/overhead-video-worked-examples"> Koger's  directory </a>.

Koger, B., Deshpande, A., Kerby, J.T., Graving, J.M., Costelloe, B.R., Couzin, I.D. Multi-animal behavioral tracking and environmental reconstruction using drones and computer vision in the wild.