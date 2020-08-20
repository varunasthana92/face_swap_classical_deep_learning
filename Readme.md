# Face Swapping with Classical and Deep learning Approach

Objective of the project is to detect a face in the image (termed as target image) and replace it with another face from a source image. Project was part of the academic coursework CMSC733-Classical and Deep Learning Approaches for
Geometric Computer Vision at the Univeristy of Maryland-College Park.<br/>
<p align="center">
<img src="https://github.com/varunasthana92/Lane_Detection/blob/master/pics/final.gif">
</p>

For classical approach 2 methods were implemented-  
1) Thin Plate Spline (TPS)  
2) Delaunay Triangulation (DT)  
  
It was observed that TPS is faster than DT.  
  
For deep learning approach, we used a network from [this paper](https://arxiv.org/abs/1803.07835), which implements a supervised encoder-decoder model to obtain the full 3D mesh of the face. The code from the paper can be found [here](https://github.com/YadiraF/PRNet). 

__NOTE:__ Please make sure that shape_predictor_68_face_landmarks.dat file is available in main repo

Below command line arguments can be used as described
```
--method, default="TPS", help="Which method for warping"
--swap_vid, default=0, help="Set 1, when you want to input a target video. Also use arguemnt video_path"
--video_path, default="Data/Test1.mp4", help="Path to the target video file"
--swap_within_vid, default = 0, help="Set 1, When you want to swap two largest faces in one video"
--src_img_path, default="Data/1.jpeg", help="Path to the COLOR image whose face you want to use as source"
--img_path, default="Data/2.jpeg", help="Path to the COLOR image whose face you want to replace with source face"
--save_path, default="Results/", help="Path where you want to save output (provide / at the end)"
```



Please make sure that shape_predictor_68_face_landmarks.dat file is available in the Data folder

To run test 1 

with thin plate spline
```
python3 --swapOneFace=1 --Method="TPS" --imagePath="Data/Rambo.jpg" --videoPath="Data/Test1.mp4" 
```

with thin Delaunay Triangulation
```
python3 --swapOneFace=1 --Method="Delau" --imagePath="Data/Rambo.jpg" --videoPath="Data/Test1.mp4" 
```

To run test 2

with thin plate spline
```
python3 --swapTwoFace=1 --swapOneFace=0 --Method="TPS" --videoPath="Data/Test2.mp4" 
```

with thin Delaunay Triangulation
```
python3 --swapTwoFace=1 --swapOneFace=0 --Method="Delau" --videoPath="Data/Test2.mp4" 
```

To run test 3

with thin plate spline
```
python3 --swapOneFace=1 --Method="TPS" --imagePath="Data/Scarlett.jpg" --videoPath="Data/Test3.mp4" 
```

with thin Delaunay Triangulation
```
python3 --swapOneFace=1 --Method="Delau" --imagePath="Data/Scarlett.jpg" --videoPath="Data/Test3.mp4" 
```


To run the PRNet deep learning model, download the repository provided in the link on the instructions page of the project along with the data from google drive.

Copy the files provided in this submission in the directory where PRNet (after copying the repository)
1) demo_copy.py
2) tpsDeepLearning.py
3) api.py

To run the code use the below command with the same meaning of the arguments

__--swap_within_vid__  
__--src_img_path__  
__--img_path__  
__--save_path__  
__--video_path__  

This version only allows to either swap faces in 2 images, or swap 2 faces within a video.  

```
python2 demo_copy.py --isKpt=True --swap_within_vid=1 --video_path=<path>
```
