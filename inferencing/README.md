## Inferencing

The code in this folder can be used to detect marching band members and save their locations as coordinates into numpy files. 

1. Extract the frames from the videos that you are interested in analyzing with [videos_to_frames.ipynb](inferencing/videos_to_frames.ipynb).

- In our case, we have marching band members identify moments where the band was doing the same movements at time 1 and time 2 and noted those time frames (we call it the <a href="https://docs.google.com/spreadsheets/d/1rupKVdS-l-eAkqmV4vzzu9JArVNMc1ydyCLtdegbooQ/edit#gid=0"> synchrony table </a>). We also did some pre-processing to trim the raw drone footage into shorter video clips so that it takes less time to run through the rest of the pipeline. Some useful code can be found <a href="https://stackoverflow.com/questions/37317140/how-can-i-efficiently-cut-out-part-of-a-video"> here </a>.

2.  Use the code in the [frames_to_warped_frames.ipynb](inferencing/frames_to_warped_frames.ipynb) notebook to get the area of interest for you. Skip this part if you input video is already pre-processed.

- In our case, We cropped the frames to only keep the football field so that it will be easier for us to calculate the coordinates and to compare it with the ground truth drill videos. 

3. Inference on the warped frames and save the box frames and the coordinates with [draw_boxes.ipynb](inferencing/draw_boxes.ipynb).

4. (Optional) Generate box videos using box frames with [frames_to_videos.ipynb](inferencing/frames_to_videos.ipynb)
