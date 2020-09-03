# Face Swapping with Classical and Deep learning Approach

Objective of the project is to detect a face in the image (termed as target image) and replace it with another face from a source image. Project was part of the academic coursework CMSC733-Classical and Deep Learning Approaches for
Geometric Computer Vision at the Univeristy of Maryland-College Park.<br/>
<p align="center">
<img src="https://github.com/varunasthana92/face_swap_classical_deep_learning/blob/master/Data/1.jpeg" width="300">
<img src="https://github.com/varunasthana92/face_swap_classical_deep_learning/blob/master/Data/2.jpeg" width="190">
</p>

<p align="center">
<img src="https://github.com/varunasthana92/face_swap_classical_deep_learning/blob/master/Results/tps_Swap_img.png"width="250">
</p>

<p align="center">
<img src="https://github.com/varunasthana92/face_swap_classical_deep_learning/blob/master/Data/rambo.png" width="270">
<img src="https://github.com/varunasthana92/face_swap_classical_deep_learning/blob/master/Data/3.png" width="300">
</p>

<p align="center">
<img src="https://github.com/varunasthana92/face_swap_classical_deep_learning/blob/master/Results/rambo_girl.gif">
</p>

For classical approach 2 methods were implemented-  
1) Thin Plate Spline (TPS)  
2) Delaunay Triangulation (DT)  
  
It was observed that TPS is faster than DT.  
  
For deep learning approach, we used a network from [this paper](https://arxiv.org/abs/1803.07835), which implements a supervised encoder-decoder model to obtain the full 3D mesh of the face. The code from the paper can be found [here](https://github.com/YadiraF/PRNet). 

## Dependencies
- opencv 3.3.1
- numpy 1.16.6
- matplotlib 2.2.5
- scipy 1.2.0
- dlib 19.19.99
- imutils 0.5.3
- python 2.7

## How to Run

### Classical Vision Methods
__NOTE:__ Please make sure that __shape_predictor_68_face_landmarks.dat__ file is available in main repo


```
python Wrapper.py --imagePath="Data/Rambo.jpg" --videoPath="Data/Test1.mp4"
```

Below command line arguments can be used as described:
```
--method, default="TPS", help="Which method for warping (any other input will trigger Delaunay method)"
--swap_vid, default=0, help="Set 1, when you want to input a target video. Also use arguemnt video_path"
--video_path, default="Data/Test1.mp4", help="Path to the target video file"
--swap_within_vid, default = 0, help="Set 1, When you want to swap two largest faces in one video"
--src_img_path, default="Data/1.jpeg", help="Path to the COLOR image whose face you want to use as source"
--img_path, default="Data/2.jpeg", help="Path to the COLOR image whose face you want to replace with source face"
--save_path, default="Results/", help="Path where you want to save output (provide / at the end)"
```

__Note:__ For this project, frontal face detector has been used, thus it does not work if the orientation of the face in the image is side-ways.


### Deep Learning Vision Method

To run the PRNet deep learning model, download the repository provided in the above link along with the data from google drive.

From the PRNet_files directory copy the below files to the downloaded PRNet repository:  
1) demo_copy.py  
2) tpsDeepLearning.py  
3) api.py  

To run the code use the below command with the same meaning of the arguments as above.  
__--swap_within_vid__  
__--src_img_path__  
__--img_path__  
__--save_path__  
__--video_path__  

This version only allows to either swap faces in 2 images, or swap 2 faces within a video.  

```
python demo_copy.py --isKpt=True --swap_within_vid=1 --video_path=<path>
```

## Contact Information
Name: Varun Asthana  
Email id: varunasthana92@gmail.com

## References

[Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network](https://arxiv.org/abs/1803.07835), ECCV, 2018.