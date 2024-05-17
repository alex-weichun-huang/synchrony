## Inferencing

The code in this folder can be used to detect marching band members and save their locations as coordinates into numpy files. 

1. Extract the frames from the videos that you are interested in analyzing with [videos_to_frames.ipynb](videos_to_frames.ipynb).

- In our case, we have marching band members identify moments where the band was doing the same movements at time 1 and time 2 and noted those time frames in a csv file. We also did some pre-processing to trim the raw drone footage into shorter video clips so that it takes less time to run through the rest of the pipeline. Some useful code can be found <a href="https://stackoverflow.com/questions/37317140/how-can-i-efficiently-cut-out-part-of-a-video"> here </a>.

2.  Use the code in the [frames_to_warped_frames.ipynb](frames_to_warped_frames.ipynb) notebook to get the area of interest for you. Skip this part if your input video is already pre-processed.

- In our case, we warped the frames to focus solely on the football field, making it easier to calculate coordinates and compare them with the ground truth drill videos.

3. Inference on the warped frames and save the box frames and the coordinates with [warped_frames_to_box_frames.ipynb](warped_frames_to_box_frames.ipynb).

4. (Optional) Generate box videos using box frames with [box_frames_to_box_videos.ipynb](box_frames_to_box_videos.ipynb) for sanity check and demo.
