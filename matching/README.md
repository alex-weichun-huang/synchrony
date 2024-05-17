## Matching

The code in this folder can be used to adjust the ground truth coordinates to accommodate cropping and rotation (you can consider it as warping the drill videos in the same way we warped the detection video), and then create a one-to-one matching for each of the ground truth points.

1. Use the code in [preprocess_formation.ipynb](preprocess_formation.ipynb) to adjust the coordinates in the ground truth formation video for cropping and rotation. Skip this part if the coordinates of your ground truth videos and the detection videos are already comparable (in the same direction and  <b>ONLY</b> contains the area of interest).

2. Currently, we are using a simple matching algorithm to match detecion coordinates to formation coordinates with [match_formation_to_detection.ipynb](match_formation_to_detection.ipynb). 

- For each frame, we use either matching (which only consider the current frame and disregard any previous matching results) or tracking (use the previous frame's matching results to help match the current frame, based on the idea that the same person's location in two consecutive frames wouldn't change much).