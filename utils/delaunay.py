import numpy as np
import matplotlib.pyplot as plt
try:
	import cv2
except:
	import sys
	sys.path.remove(sys.path[2])
	import cv2
import dlib
from imutils import face_utils
from scipy import interpolate
import argparse

def detectedFaces(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	detector = dlib.get_frontal_face_detector()
	try:
		p = "shape_predictor_68_face_landmarks.dat"
		predictor = dlib.shape_predictor(p)
	except:
		p = "../shape_predictor_68_face_landmarks.dat"
		predictor = dlib.shape_predictor(p)
	faces = detector(gray, 0)
	if(len(faces) == 0):
		faces = detector(gray, 1)
	allFaceMarkers = []
	allFaceImgs = []
	count = 0
	for face in faces:
		count += 1
		markers = predictor(gray, face)
		markers = face_utils.shape_to_np(markers)
		convexHull= cv2.convexHull(markers)
		mask = np.zeros_like(gray)
		cv2.fillConvexPoly(mask, convexHull, 255)
		faceImg = cv2.bitwise_and(img, img, mask=mask)
		allFaceMarkers.append(markers)
		allFaceImgs.append(faceImg)	
	return count, np.array(allFaceMarkers), np.array(allFaceImgs)

def delaunay(src, src_pts, trg, trg_pts):
	trg_swap = trg.copy()
	srcGray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	trgGray = cv2.cvtColor(trg, cv2.COLOR_BGR2GRAY)
	th, tw = srcGray.shape[0], srcGray.shape[1]
	srcHull = cv2.convexHull(src_pts)
	srcBox = cv2.boundingRect(srcHull)
	(x,y,w,h) = srcBox  
	subdiv = cv2.Subdiv2D(srcBox)
	for p in src_pts:
		subdiv.insert(tuple(p))
	triangles = subdiv.getTriangleList()
	triangles = np.array(triangles, dtype=np.int32)
	c = 0
	x = x - 10
	y = y - 10
	w = w + 20
	h = h + 20
	for t in triangles:
		'''
			Format of 't' is used as below to get 3 vertices of the triangle
			pt1 = (t[0], t[1])
			pt2 = (t[2], t[3])
			pt3 = (t[4], t[5])
			CHeck that detected delaunay triangle have vertices within the rectangular box of convex hull
			of detected facial landmarks
		'''
		if(t[0]>=x and t[0]<=x+w and t[1]>=y and t[1]<=y+h and t[2]>=x and t[2]<=x+w and t[3]>=y and t[3]<=y+h and t[4]>=x and t[4]<=x+w and t[5]>=y and t[5]<=y+h):
			c+=1
			pt1 = (t[0], t[1])
			pt2 = (t[2], t[3])
			pt3 = (t[4], t[5])
			srcTri = np.array([pt1, pt2, pt3])
			rect1 = cv2.boundingRect(srcTri)
			(x1,y1,w1,h1) = rect1                   
			src_tri_bund_box_arr = []
			
			for yy in range(y1,y1+h1+1):
				# generate the coordiantes of all pixel points inside the rect1 
				# as (x,y,1) and append to src_tri_bund_box_arr
				temp = np.linspace((x1,yy,1),(x1+w1,yy,1), w1+1)
				src_tri_bund_box_arr.append(temp)
			
			src_tri_bund_box_arr = np.array(src_tri_bund_box_arr, np.int32)
			src_tri_bund_box_arr = (src_tri_bund_box_arr.flatten()).reshape(-1,3)
			src_tri_bund_box_arr = np.transpose(src_tri_bund_box_arr)

			idx1 = np.argwhere((src_pts == pt1).all(axis=1))[0][0]
			idx2 = np.argwhere((src_pts == pt2).all(axis=1))[0][0]
			idx3 = np.argwhere((src_pts == pt3).all(axis=1))[0][0]
			
			# generate corresponding triangles in target image
			# src_pts and trg_pts have index correspondeence 
			trg_pt1 = tuple(trg_pts[idx1])
			trg_pt2 = tuple(trg_pts[idx2])
			trg_pt3 = tuple(trg_pts[idx3])
			trgTri = np.array([trg_pt1, trg_pt2, trg_pt3], dtype= np.int32)
			rect2 = cv2.boundingRect(trgTri)
			(x2,y2,w2,h2) = rect2
			trg_tri_bund_box_arr = []
			for yy in range(y2,y2+h2+1):
				# generate the coordiantes of all pixel points inside the rect12
				# as (x,y,1) and append to trg_tri_bund_box_arr
				temp = np.linspace((x2,yy,1),(x2+w2,yy,1), w2+1)
				trg_tri_bund_box_arr.append(temp)
			trg_tri_bund_box_arr = np.array(trg_tri_bund_box_arr, np.int32)
			trg_tri_bund_box_arr = (trg_tri_bund_box_arr.flatten()).reshape(-1,3)
			trg_tri_bund_box_arr = np.transpose(trg_tri_bund_box_arr)
			
            # for each corresponding delaunay triangle in src and trg image
			src_inv_match, trg_cord_idx, status = bary(src_tri_bund_box_arr, trg_tri_bund_box_arr, srcTri, trgTri, trg, src)
			
			if(status):
				# src image interpolation for the points between int pixels
				X  = np.arange(0, src.shape[1])
				Y  = np.arange(0, src.shape[0])
				ZB = (src[:,:,0])
				ZG = (src[:,:,1])
				ZR = (src[:,:,2])

				# define the interpolation function for each color channel
				# of the src img with the use of scipy.interpolate
				fb = interpolate.interp2d(X, Y, ZB, kind='cubic', fill_value=0)
				fg = interpolate.interp2d(X, Y, ZG, kind='cubic', fill_value=0)
				fr = interpolate.interp2d(X, Y, ZR, kind='cubic', fill_value=0)

				for i in range(len(trg_cord_idx)):
					blue   				= fb(src_inv_match[0,i], src_inv_match[1,i])
					red    				= fr(src_inv_match[0,i], src_inv_match[1,i])
					green  				= fg(src_inv_match[0,i], src_inv_match[1,i])
					w_, h_ 				= trg_tri_bund_box_arr[0, trg_cord_idx[i] ], trg_tri_bund_box_arr[1, trg_cord_idx[i] ]
					# replace the value at the (h,w) with the interpolated value from src img
					# for each channel, thus affine transform the src triangle to trg triangle
					trg_swap[h_,w_,0] 	= blue
					trg_swap[h_,w_,1] 	= green
					trg_swap[h_,w_,2] 	= red
	return trg_swap

def bary(src_box, trg_box, srcTri, trgTri, trgImg, srcImg):
	mask = np.zeros_like(trgImg)
	mask2 = np.zeros_like(srcImg)
	bary_trg = np.array([[trgTri[0][0], trgTri[1][0], trgTri[2][0]],
						 [trgTri[0][1], trgTri[1][1], trgTri[2][1]],
						 [1, 1, 1]])
	bary_src = np.array([[srcTri[0][0], srcTri[1][0], srcTri[2][0]],
						 [srcTri[0][1], srcTri[1][1], srcTri[2][1]],
						 [1, 1, 1]])
	try:
		# for inverse warping, if matrix inv exist
		bary_inv_trg = np.linalg.inv(bary_trg)
		trg_bcord = np.matmul(bary_inv_trg, trg_box)
		inliers = []
		coord = []
		for i in range (trg_bcord.shape[1]):
			a, b, y = trg_bcord[0,i], trg_bcord[1,i], trg_bcord[2,i]
			# of all the points in the bonding box, we only need the points
			# inside the triangle along with their barrycentric values
			if(a>=0 and a<=1 and b>=0 and b<=1 and y>=0 and y<=1 and (a+b+y)>0):
				inliers.append([a,b,y])
				coord.append(int(i))
				mask[trg_box[1,i], trg_box[0,i]] = trgImg[trg_box[1,i], trg_box[0,i]]
		inliers = np.array(inliers)
		inliers = (inliers.flatten()).reshape(-1,3)
		inliers = np.transpose(inliers)
		# convert the barycentric cordiantes to the src img points i.e. inverse warping
		src_match = np.matmul(bary_src, inliers, dtype= np.float64)
        # if(src_match[2] ==0):
        #     src_match[2] = 0.0000001

        # normalize the points to homogenous 2D points
        # note here src_match are not int, and thus we will interpolate their values
		src_match = src_match/src_match[2]
		return src_match, coord, True
	
	except:
		return [],[], False


def triangles(src, src_pts, name):
	srcCpy = src.copy()
	srcGray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	th, tw = srcGray.shape[0], srcGray.shape[1]
	srcHull = cv2.convexHull(src_pts)
	srcBox = cv2.boundingRect(srcHull)
	(x,y,w,h) = srcBox
	cv2.rectangle(srcCpy, (x, y), (x + w, y + h), (0, 255, 0), 2)  
	subdiv = cv2.Subdiv2D(srcBox)
	for p in src_pts:
		subdiv.insert(tuple(p))
	triangles = subdiv.getTriangleList()
	triangles = np.array(triangles, dtype = np.int32)
	x = x - 10
	y = y - 10
	w = w + 20
	h = h + 20
	for t in triangles:
		if(t[0]>=x and t[0]<=x+w and t[1]>=y and t[1]<=y+h and t[2]>=x and t[2]<=x+w and t[3]>=y and t[3]<=y+h and t[4]>=x and t[4]<=x+w and t[5]>=y and t[5]<=y+h):
			pt1 = (t[0], t[1])
			pt2 = (t[2], t[3])
			pt3 = (t[4], t[5])
			srcTri = np.array([pt1, pt2, pt3])
			#cv2.rectangle(srcCpy, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
			cv2.line(srcCpy, pt1, pt2, (0, 255, 0), 1)
			cv2.line(srcCpy, pt2, pt3, (0, 255, 0), 1)
			cv2.line(srcCpy, pt1, pt3, (0, 255, 0), 1)
	cv2.imwrite(name +"_Tri.jpg", srcCpy)
	return

def ColorBlend(src, dst, srcPts, erode_flag=1):
	srcHull = cv2.convexHull(srcPts)
	srcBox = cv2.boundingRect(srcHull)
	(x,y,w,h) = srcBox
	src_mask = np.zeros_like(dst)
	cv2.fillConvexPoly(src_mask, srcHull, (255, 255, 255))
	center = (int((x+x+w)/2), int((y+y+h)/2))

 	if erode_flag:
		kernel 		= np.ones((10, 10), np.uint8)
		src_mask 	= cv2.erode(src_mask, kernel, iterations=1)

	# Clone seamlessly.
	imgNew = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
	return imgNew

def singleFaceSwap(src, trg, src_face_count, srcAllMarkers, srcFaceImgs):
	trg_face_count, trgAllMarkers, trgFaceImgs = detectedFaces(trg)
	if(len(trgAllMarkers)):
		swap_img = delaunay(src, srcAllMarkers[0], trg, trgAllMarkers[0])
		swap_img_color_blend = ColorBlend(swap_img, trg, trgAllMarkers)
		return swap_img_color_blend
	return trg

def swapTwoFaces(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	trg_face_count, trgAllMarkers, trgFaceImgs = detectedFaces(image)
	if trg_face_count>=2:
		swap_img_1 = delaunay(image.copy(), trgAllMarkers[0], image.copy(), trgAllMarkers[1])
		swap_img_color_blend_1 = ColorBlend(swap_img_1, image, trgAllMarkers[1])
		swap_img_2 = delaunay(image.copy(), trgAllMarkers[1], image.copy(), trgAllMarkers[0])
		swap_img_color_blend_2 = ColorBlend(swap_img_2, swap_img_color_blend_1, trgAllMarkers[0])
		# cv2.imwrite("Swap1.jpg",swapClr1)
		# cv2.imwrite("Swap2.jpg",swapClr2)
		return swap_img_color_blend_2
	return image

def main(Args):
	src = cv2.imread(Args.src_img_path)
	if Args.swap_within_vid != 1:
		if type(src) is np.ndarray:
			src_face_count, srcAllMarkers, srcFaceImgs = detectedFaces(src)
		else:
			print('src_img_path wrong!!!')
			return

	# swap face in 2 images
	if Args.swap_vid != 1:
		trg = cv2.imread(Args.img_path)
		if(type(trg) is not np.ndarray):
			print('img_path wrong!!!')
			return
		swap_img_color_blend = singleFaceSwap(src, trg, src_face_count, srcAllMarkers, srcFaceImgs)
		if(type(swap_img_color_blend) is not np.ndarray):
			print('Could not swap!!')
			return
		cv2.imwrite(Args.save_path + "Swap_img.png", swap_img_color_blend)
		print('Press ESC to exit')
		cv2.imshow("Swapped Img", swap_img_color_blend)
		cv2.waitKey(0)

	else:
		# swap face in a video with a src image
		if(Args.swap_within_vid != 1):
			cap = cv2.VideoCapture(Args.video_path)
			ret, trg= cap.read()
			if(ret == 0):
				print('video not read!!!')
				return;
			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			vw = cv2.VideoWriter(Args.save_path + "Swapped_video.avi", fourcc, 30, (trg.shape[1], trg.shape[0]))
			frameCount = 1
			while (cap.isOpened()):
				ret, trg = cap.read()
				frameCount += 1
				if(ret):
					print(frameCount)
					swap_img_color_blend = singleFaceSwap(src, trg, src_face_count, srcAllMarkers, srcFaceImgs)
					vw.write(swap_img_color_blend)
				else:
					print(frameCount,' Frame not read')
			vw.release()
			cap.release()
		# swap 2 largest face within the video
		else:
			cap = cv2.VideoCapture(Args.video_path)
			ret, trg= cap.read()
			if(ret == 0):
				print('video not read!!!')
				return;
			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			vw = cv2.VideoWriter(Args.save_path + "Swapped_video.avi", fourcc, 30, (trg.shape[1], trg.shape[0]))
			frameCount = 1
			while (cap.isOpened()):
				ret, trg = cap.read()
				frameCount += 1
				if(ret):
					print(frameCount)
					swap_img_color_blend = swapTwoFaces(trg)
					vw.write(swap_img_color_blend)
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
	Args.save_path += 'delaunay_'
	main(Args)
