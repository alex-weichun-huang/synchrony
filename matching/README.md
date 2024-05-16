## Matching

The code in this folder can be used to adjust the ground truth coordinates to the same scale as the detection videos (you can think of it as warping the drill videos just like how we warped the detection video), and then create a one-to-one matching for each of the ground truth points.

1. Use the code in [preprocess_ground_truth.ipynb](preprocess_ground_truth.ipynb) to adjust the coordinates in the ground truth formation video for cropping and rotation. Skip this part if the coordinates of your ground truth videos and the detection videos are already comparable (in the same direction and  <b>ONLY</b> contains the area of interest).

2. Currently, we are using a simple matching algorithm to match detecion coordinates to formation coordinates with [match_formation_to_detection.ipynb](match_formation_to_detection.ipynb). 
