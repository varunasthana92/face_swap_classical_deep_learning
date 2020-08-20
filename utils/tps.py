import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.remove(sys.path[2])
import cv2
import dlib
from imutils import face_utils
from scipy import interpolate
import math
import argparse


def giveFeaturePoints(gray):
	detector = dlib.get_frontal_face_detector()
	try:
		p = "shape_predictor_68_face_landmarks.dat"
		predictor = dlib.shape_predictor(p)
	except:
		p = "../shape_predictor_68_face_landmarks.dat"
		predictor = dlib.shape_predictor(p)
	faces = detector(gray, 0)
	detectedFace = False
	shape = None
	center = None
	markers = None
	if faces:
		faces = faces[0]
		markers = predictor(gray, faces)
		markers = face_utils.shape_to_np(markers)
		faceHull = cv2.convexHull(markers)
		(x,y,w,h) = cv2.boundingRect(faceHull)
		center = (int((x+x+w)/2), int((y+y+h)/2))
		detectedFace = True
	return markers, center, detectedFace

def giveTwoface(gray):
	detector = dlib.get_frontal_face_detector()
	try:
		p = "shape_predictor_68_face_landmarks.dat"
		predictor = dlib.shape_predictor(p)
	except:
		p = "../shape_predictor_68_face_landmarks.dat"
		predictor = dlib.shape_predictor(p)
	faces = detector(gray, 1)
	print("Number of faces - " + str(len(faces)))
	markers_1, markers_2 = None, None
	center_1, center_2 = None, None
	detectedFaces = False
	if len(faces)>1:
		face_1 = faces[0]
		markers_1 = predictor(gray, face_1)
		markers_1 = face_utils.shape_to_np(markers_1)
		faceHull = cv2.convexHull(markers_1)
		(x,y,w,h) = cv2.boundingRect(faceHull)
		center_1 = (int((x+x+w)/2), int((y+y+h)/2))

		face_2 = faces[1]
		markers_2 = predictor(gray, face_2)
		markers_2 = face_utils.shape_to_np(markers_2)
		faceHull = cv2.convexHull(markers_2)
		(x,y,w,h) = cv2.boundingRect(faceHull)
		center_2 = (int((x+x+w)/2), int((y+y+h)/2))

		detectedFaces = True
	return markers_1, markers_2, center_1, center_2, detectedFaces 

def compute_K_xy(x, y, source):    
	sx, sy 	= source[:,0], source[:,1] 
	kx_,ky_ = np.tile(x, (source.shape[0],1)).T, np.tile(y,(source.shape[0],1)).T
	tmp 	= (kx_ - sx)**2 + (ky_ - sy)**2 + sys.float_info.epsilon**2
	K_xy	= tmp*np.log(tmp)    
	return K_xy

def NewTps(target, source, lam):
	# for inverse warp from target to source
	# here calculations are done wrt detected facial landmarks
	# mapped from trg img to src image landmarks
	P 		= np.append(target,np.ones([target.shape[0],1]),axis=1)
	P_Trans = P.T
	Z 		= np.zeros([3,3])
	K 		= compute_K_xy(target[:,0], target[:,1], target)
	M 		= np.vstack([np.hstack([K,P]),np.hstack([P_Trans,Z])])
	I 		= np.identity(M.shape[0])
	L 		= M+lam*I
	L_inv 	= np.linalg.inv(L)

	# src points for mapping
	V 		= np.concatenate([source, np.zeros([3,2])])
	weights	= np.matmul(L_inv,V)
	return weights

def mask_from_points(size, points, erode_flag=0):
	mask 	= np.zeros(size, np.uint8)
	cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
	if erode_flag:
		kernel 	= np.ones((10, 10), np.uint8)
		mask 	= cv2.erode(mask, kernel, iterations=1)

	return mask


def NewWarp(img_target, img_source, trg_points, src_points, wt_trg_tps, mask_target):
	xy_trg_min = np.float32([min(trg_points[:,0]),min(trg_points[:,1])])
	xy_trg_max = np.float32([max(trg_points[:,0]),max(trg_points[:,1])])
	x 		= np.arange(xy_trg_min[0],xy_trg_max[0]).astype(int)
	y 		= np.arange(xy_trg_min[1],xy_trg_max[1]).astype(int)
	X,Y 	= np.mgrid[x[0]:x[-1]+1,y[0]:y[-1]+1]
	w,h 	= X.shape
	X,Y 	= X.ravel(), Y.ravel()
	pts_trg = np.hstack((X.reshape([-1,1]),Y.reshape([-1,1])))

	# here we use the TPS wts obtained by the processing the facial landmarks
	# and apply on the bounding box of the face in the trg img
	# thus applying the obtained TPS model on each pixel of the face 
	P 		= np.append(pts_trg, np.ones([pts_trg.shape[0],1]), axis=1)
	Z 		= np.zeros([3,3])
	K 		= compute_K_xy(X,Y, trg_points)
	M 		= np.hstack([K,P])
	vv 		= M.dot(wt_trg_tps).astype(np.int64)
	
	# mapped pixels points (int to float) from trg on src img
	map_x = (vv[:,0]).reshape([w,h]).astype(np.float32)
	map_y = (vv[:,1]).reshape([w,h]).astype(np.float32)
	# interpolating values for the mapped points in src image
	# these interpollated values will then be used in the corresponding trg img pixel (int) points
	dst = cv2.remap(img_source, map_x, map_y, cv2.INTER_LINEAR)
	dst = cv2.flip(np.rot90(dst, k=3), 1)
	h,w,_ = dst.shape

	swap_img = np.zeros_like(img_target)
	swap_img[y[0]:y[0]+h, x[0]:x[0]+w, :] = dst
	# swap_img = cv2.bitwise_and(swap_img, swap_img, mask=mask_target)
	return swap_img


def swap(sourceImage, targetImage, sourceImage_gray, targetImage_gray, 
	source_points, source_cent, target_points, target_cent, 
	debug = False, showW = 160, showH = 120):
	w, h 				= targetImage_gray.shape
	mask_target 		= mask_from_points((w, h), target_points)
	wt_trg_tps 			= NewTps(target_points, source_points, 1e-8)
	swap_img			= NewWarp(targetImage, sourceImage, target_points, source_points, wt_trg_tps, mask_target)
	swap_img_colorBlend	= cv2.seamlessClone(swap_img, targetImage, mask_target, tuple(target_cent) ,cv2.NORMAL_CLONE)
	return swap_img_colorBlend


def main(Args):
	sourceImage = cv2.imread(Args.src_img_path)

	if Args.swap_within_vid != 1:
		if type(sourceImage) is np.ndarray:
			sourceImage_gray 				= cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
			source_points, source_cent, res	= giveFeaturePoints(sourceImage_gray)
			if res == False:
				print('Face not detected in source image!!!')
				return
		else:
			print('src_img_path wrong!!!')
			return

	# swap face in 2 images
	if Args.swap_vid != 1:
		targetImage = cv2.imread(Args.img_path)
		if(type(targetImage) is not np.ndarray):
			print('img_path wrong!!!')
			return
		targetImage_gray 				= cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
		target_points, target_cent, res	= giveFeaturePoints(targetImage_gray)
		if res:
			swap_img_color_blend		= swap(sourceImage, targetImage, sourceImage_gray, targetImage_gray, 
										 	   source_points, source_cent, target_points, target_cent)
		else:
			print('Face not detected in target image!!!')
			return
		if(type(swap_img_color_blend) is not np.ndarray):
			print('Could not swap!!')
			return
		cv2.imwrite(Args.save_path + "Swap_img.png", swap_img_color_blend)
		print('Press ESC to exit')
		cv2.imshow(Args.save_path + "Swapped Img", swap_img_color_blend)
		cv2.waitKey(0)

	else:
		# swap face in a video with a src image
		if(Args.swap_within_vid != 1):
			cap = cv2.VideoCapture(Args.video_path)
			ret, targetImage = cap.read()
			if(ret == 0):
				print('video not read!!!')
				return;
			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			vw = cv2.VideoWriter(Args.save_path + "Swapped_video.avi", fourcc, 30, (targetImage.shape[1], targetImage.shape[0]))
			frameCount = 1
			while cap.isOpened():
				ret, targetImage = cap.read()
				frameCount += 1
				if(ret):
					print(frameCount)
					targetImage_gray 				= cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
					target_points, target_cent, res	= giveFeaturePoints(targetImage_gray)
					if res:
						swap_img_color_blend 		= swap(sourceImage, targetImage, sourceImage_gray, targetImage_gray, 
											 	   			   source_points, source_cent, target_points, target_cent)
						vw.write(swap_img_color_blend)
					else:
						vw.write(targetImage)
				else:
					print(frameCount,' Frame not read')
			vw.release()
			cap.release()
		# swap 2 largest face within the video
		else:
			cap = cv2.VideoCapture(Args.video_path)
			ret, image= cap.read()
			if(ret == 0):
				print('video not read!!!')
				return;
			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			vw = cv2.VideoWriter(Args.save_path + "Swapped_video.avi", fourcc, 30, (image.shape[1], image.shape[0]))
			frameCount = 1
			while (cap.isOpened()):
				ret, image = cap.read()
				frameCount += 1
				if(ret):
					print(frameCount)
					gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					face1_points, face2_points, face1_center, face2_center, detectedFaces = giveTwoface(gray)

					if detectedFaces:
						trg_img = swap(image, image, gray, gray, 
									face1_points, face1_center, face2_points, face2_center)

						trg_gray = cv2.cvtColor(trg_img, cv2.COLOR_BGR2GRAY)
						swap_img_color_blend = swap(image, trg_img, gray, trg_gray, face2_points,
													face2_center, face1_points, face1_center)
						vw.write(swap_img_color_blend)
					else:
						vw.write(image)
				else:
					print(frameCount,' Frame not read')
			vw.release()
			cap.release()
	return
	
if __name__ == '__main__':
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--swap_vid', type = int, default=0, help="Set 1, when you want to input a target video. Also use arguemnt video_path")
	Parser.add_argument('--video_path', default="../Data/Test1.mp4", help="Path to the target video file")
	Parser.add_argument('--swap_within_vid', type = int, default = 0, help="Set 1, When you want to swap two largest faces in one video")
	Parser.add_argument('--src_img_path', default="../Data/1.jpeg", help="Path to the COLOR image whose face you want to use as source")
	Parser.add_argument('--img_path', default="../Data/2.jpeg", help="Path to the COLOR image whose face you want to replace with source face")
	Parser.add_argument('--save_path', default="../Results/", help="Path where you want to save output (provide / at the end)")
	Args = Parser.parse_args()
	Args.save_path += 'tps_'
	main(Args)