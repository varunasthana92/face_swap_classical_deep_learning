import sys
# sys.path.remove(sys.path[1])
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import dlib
from imutils import face_utils
from scipy import interpolate
import math
from scipy import interpolate

p = "../Data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

names = ["../Data/harsh.jpeg", "../Data/photo.jpg", "../Data/H.jpeg", "../Data/U.jpeg"]

sourceImageName = names[3]
targetImageName = names[2]

sourceImage = cv2.imread(targetImageName)
sourceImage_gray = cv2.imread(targetImageName,0)

targetImage = cv2.imread(sourceImageName)
targetImage_gray = cv2.imread(sourceImageName,0)

def giveFeaturePoints(gray):
	rects = detector(gray, 0)
	detectedFaces = False
	shape = None
	cent = None
	if rects:
		sx, sy = rects[0].left(), rects[0].top()
		ex,ey = sx+rects[0].width(), sy+rects[0].height()
		patch = gray[sy:ey, sx:ex]
		cent = [sx + rects[0].width()//2, sy + rects[0].height()//2]
		rect = rects[0]
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		detectedFaces = True
	return shape, cent, detectedFaces

def giveTwoface(gray):
	rects = detector(gray, 1)
	print("Number of faces - " + str(len(rects)))
	shape1, shape2 = None, None
	cent1, cent2 = None, None
	detectedFaces = False
	if len(rects)==2:
		rect1 = rects[0]
		sx, sy = rects[0].left(), rects[0].top()
		cent1 = [sx + rects[0].width()//2, sy + rects[0].height()//2]
		shape1 = predictor(gray, rect1)
		shape1 = face_utils.shape_to_np(shape1)
		rect2 = rects[1]
		sx, sy = rects[1].left(), rects[1].top()
		cent2 = [sx + rects[1].width()//2, sy + rects[1].height()//2]
		shape2 = predictor(gray, rect2)
		shape2 = face_utils.shape_to_np(shape2)
		detectedFaces = True
	return shape1, shape2, cent1, cent2, detectedFaces 

def computeK(x, y, source):    
	sx, sy = source[:,0], source[:,1] 
	kx_,ky_ = np.tile(x, (source.shape[0],1)).T, np.tile(y,(source.shape[0],1)).T
	tmp = (kx_ - sx)**2 + (ky_ - sy)**2 + sys.float_info.epsilon**2
	K = tmp*np.log(tmp)    
	return K

def NewTps(trg, src, lam):
	P = np.append(trg,np.ones([trg.shape[0],1]),axis=1)
	P_Trans = P.T
	Z = np.zeros([3,3])
	K = computeK(trg[:,0], trg[:,1], trg)
	M = np.vstack([np.hstack([K,P]),np.hstack([P_Trans,Z])])
	I = np.identity(M.shape[0])
	L = M+lam*I
	L_inv = np.linalg.inv(L)
	V = np.concatenate([src,np.zeros([3,2])])
	weights = np.matmul(L_inv,V)
	return weights

def mask_from_points(size, points,erode_flag=0):
	radius = 10  # kernel size
	kernel = np.ones((radius, radius), np.uint8)
	mask = np.zeros(size, np.uint8)
	
	m = cv2.convexHull(points)
	# n = cv2.convexHull(points)
	cv2.fillConvexPoly(mask, m, 255)
	# if erode_flag:
	# 	mask = cv2.erode(mask, kernel,iterations=1)
	return mask

def U(r):
	return r*r*math.log(r*r)

def fxy(points1, points2, weights):
	K = np.zeros([points2.shape[0], 1])
	for i in range(points2.shape[0]):
		K[i] = U(np.linalg.norm((points2[i] - points1), ord =2)+sys.float_info.epsilon)
	f = weights[-1] + weights[-3] * points1[0] + weights[-2] * points1[1] + np.matmul(K.T, weights[0:-3])
	return f

def NewWarp1(img_target, img_src, trg_points, src_points, wt_trg_tps, mask_src, target_points):
	xyTrg_min = np.float32([min(trg_points[:,0]),min(trg_points[:,1])])
	xyTrg_max = np.float32([max(trg_points[:,0]),max(trg_points[:,1])])
	x = np.arange(xyTrg_min[0],xyTrg_max[0]).astype(int)
	y = np.arange(xyTrg_min[1],xyTrg_max[1]).astype(int)
	X,Y = np.mgrid[x[0]:x[-1]+1,y[0]:y[-1]+1]
	X,Y = X.ravel(), Y.ravel()
	xy = np.hstack((X.reshape([-1,1]),Y.reshape([-1,1]))) 

	u = np.zeros_like(xy[:,0])
	v = np.zeros_like(xy[:,0])

	# apply TPS on x and y axis
	for i in range(xy.shape[0]):
		u[i] = fxy(xy[i,:], target_points, wt_trg_tps[:,0])
		v[i] = fxy(xy[i,:], target_points, wt_trg_tps[:,1])

	outImg = img_target.copy()
	for i in range(u.shape[0]):
		try:
			# interpolation can be done, but for simplicity int values considered
			# i.e if mask is defined
			if mask_src[v[i], u[i]] > 0:
				outImg[xy[i, 1], xy[i, 0], :] = img_src[v[i], u[i], :]
		except:
			pass
	return outImg

def NewWarp(img_target, img_src, points1, points2, weights, mask2):
	xy1_min = np.float32([min(points1[:,0]),min(points1[:,1])])
	xy1_max = np.float32([max(points1[:,0]),max(points1[:,1])])
	xy2_min = np.float32([min(points2[:,0]),min(points2[:,1])])
	xy2_max = np.float32([max(points2[:,0]),max(points2[:,1])])
	x = np.arange(xy1_min[0],xy1_max[0]).astype(int)
	y = np.arange(xy1_min[1],xy1_max[1]).astype(int)
	# print(x[0],x[-1]+1)
	# print(y[0],y[-1]+1)
	X,Y = np.mgrid[x[0]:x[-1]+1,y[0]:y[-1]+1]
	w,h = X.shape
	print(X.shape)
	print(points1.shape)
	X,Y = X.ravel(), Y.ravel()

	pts_src = np.hstack((X.reshape([-1,1]),Y.reshape([-1,1]))) 
	P = np.append(pts_src,np.ones([pts_src.shape[0],1]),axis=1)
	Z = np.zeros([3,3])
	K = computeK(X,Y, points1)
	M = np.hstack([K,P])
	vv = M.dot(weights).astype(np.int64)
	
	# outImg = np.zeros([img_target.shape[1], img_target.shape[0], 3])
	outImg = img_target.copy()*0
	# outImg1 = img_target.copy()*0
	map_x = (vv[:,0]).reshape([w,h]).astype(np.float32)
	map_y = (vv[:,1]).reshape([w,h]).astype(np.float32)
	
	dst = cv2.remap(img_src, map_x, map_y, cv2.INTER_LINEAR)
	dst = cv2.flip(np.rot90(dst, k=3), 1)
	h,w,_ = dst.shape
	
	# cv2.ROTATE_90_COUNTERCLOCKWISE(dst)
	
	outImg[y[0]:y[0]+h, x[0]:x[0]+w , :] = dst
	
	return outImg

def ColorBlend(dst, src, src_mask):
    srcBox= cv2.boundingRect(src_mask)
    (x,y,w,h) = srcBox
    center = ((x+x+w)/2,(y+y+h)/2)
    imgNew = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
    return imgNew

def swap(targetImage, sourceImage, targetImage_gray, sourceImage_gray, 
	source_points, source_cent, target_points, target_cent, srcPtsall, trgPtsall,
	debug = False, showW = 160, showH = 120):
	w, h = targetImage_gray.shape
	mask_target_ = mask_from_points((w, h), trgPtsall)
	w, h = sourceImage_gray.shape
	mask_source_ = mask_from_points((w, h), srcPtsall)
	wt_trg_tps = NewTps(target_points, source_points, 0.001)
	
	# tt = NewWarp1(sourceImage, targetImage, trgPtsall, srcPtsall,wt_trg_tps, mask_source_, target_points)

	tt = NewWarp1(targetImage, sourceImage, trgPtsall, srcPtsall, wt_trg_tps, mask_source_, target_points)
	srcBox = cv2.boundingRect(mask_target_)
	(x,y,w,h) = srcBox
	center = ((x+x+w)/2,(y+y+h)/2)

	m = np.where(mask_target_!=0,targetImage_gray, 0)
	n = np.where(mask_source_!=0,sourceImage_gray, 0)

	newWarped = cv2.seamlessClone(tt, targetImage, mask_target_, center ,cv2.NORMAL_CLONE)
	g = cv2.cvtColor(tt, cv2.COLOR_BGR2GRAY)
	p = np.where(mask_target_!=0,g, 0)
	# cv2.imwrite("res/FM.jpg",p)
	if debug:
		mask_target = np.where(mask_target_!=0, targetImage_gray, 0)
		mask_source = np.where(mask_source_!=0, sourceImage_gray, 0)
		cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Source", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)		
		cv2.namedWindow("Final", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Target mask", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Source mask", cv2.WINDOW_NORMAL)

		cv2.resizeWindow("Warped", showW, showH)
		cv2.resizeWindow("Source", showW, showH)
		cv2.resizeWindow("Mask", showW, showH)
		cv2.resizeWindow("Final", showW, showH)
		cv2.resizeWindow("Target mask", showW, showH)
		cv2.resizeWindow("Source mask", showW, showH)
		
		cv2.imshow("Warped", tt)
		cv2.imshow("Source", sourceImage)
		cv2.imshow("Mask", mask_source_)
		cv2.imshow("Final", newWarped)
		cv2.imshow("Target mask",mask_target)
		cv2.imshow("Source mask",mask_source)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return newWarped

if __name__ == '__main__':
	start = time.time()
	source_points, source_cent = giveFeaturePoints(targetImage_gray)
	target_points, target_cent = giveFeaturePoints(sourceImage_gray)
	newImage = swap(targetImage, sourceImage, targetImage_gray, sourceImage_gray, 
		source_points, source_cent, target_points, target_cent, 
		debug = False, showW = 240, showH = 220)
	print(time.time()-start)