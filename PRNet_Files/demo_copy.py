import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
from tpsDeepLearning import *
import sys
# sys.path.remove(sys.path[1])
import cv2
from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture
from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

prn = PRN(is_dlib = True)

def getPos(args, frame, numberFace, name):
	# if args.isShow or args.isTexture:
	#     from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

	# ---- init PRN
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
	# prn = PRN(is_dlib = args.isDlib)

	# ------------- load data
	# image_folder = args.inputDir
	save_folder = args.outputDir
	if not os.path.exists(save_folder):
	    os.mkdir(save_folder)

	# types = ('*.jpg', '*.png', '*.jpeg')
	# image_path_list= []
	# for files in types:
	#     image_path_list.extend(glob(os.path.join(image_folder, files)))
	# total_num = len(image_path_list)

	# srcImg = cv2.imread(image_path_list[0])
	# srcImg_gray = cv2.imread(image_path_list[0],0)
	# trgImg = cv2.imread(image_path_list[1])
	# trgImg_gray = cv2.imread(image_path_list[1],0)
	frameVert= []
	frameKpt= []

	# for i, image_path in enumerate(image_path_list):
	for i in range(numberFace):
		name = name+'_'+str(i)
		# name = image_path.strip().split('/')[-1][:-4]

		# read image
		# image = imread(image_path)
		image = frame.copy()

		[h, w, c] = image.shape
		if c>3:
			image = image[:,:,:3]

		# the core: regress position map
		if args.isDlib:
			max_size = max(image.shape[0], image.shape[1])
			if max_size> 1000:
				image = rescale(image, 1000./max_size)
				image = (image*255).astype(np.uint8)

			pos = prn.process(image, i) # use dlib to detect face	
		else:
			if image.shape[0] == image.shape[1]:
				image = resize(image, (256,256))
				pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
			else:
				box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
				pos = prn.process(image, box)
		
		image = image/255.
		if pos is None:
			continue

		if args.is3d or args.isMat or args.isPose or args.isShow:
			# 3D vertices
			vertices = prn.get_vertices(pos)
			if args.isFront:
				save_vertices = frontalize(vertices)
			else:
				save_vertices = vertices.copy()
			save_vertices[:,1] = h - 1 - save_vertices[:,1]

		if args.isImage:
			imsave(os.path.join(save_folder, name + '.jpg'), image)

		if args.is3d:
			# corresponding colors
			colors = prn.get_colors(image, vertices)

			if args.isTexture:
				if args.texture_size != 256:
					pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range = True)
				else:
					pos_interpolated = pos.copy()
				texture = cv2.remap(image, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
				if args.isMask:
					vertices_vis = get_visibility(vertices, prn.triangles, h, w)
					uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
					uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range = True)
					texture = texture*uv_mask[:,:,np.newaxis]
				write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture, prn.uv_coords/prn.resolution_op)#save 3d face with texture(can open with meshlab)
			else:
				write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

		if args.isDepth:
			depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
			depth = get_depth_image(vertices, prn.triangles, h, w)
			imsave(os.path.join(save_folder, name + '_depth.jpg'), depth_image)
			sio.savemat(os.path.join(save_folder, name + '_depth.mat'), {'depth':depth})

		if args.isMat:
			sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})

		if args.isKpt or args.isShow:
			# get landmarks
			kpt = prn.get_landmarks(pos)
			# np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)
			# np.savetxt(os.path.join(save_folder, name + '_vertices.txt'), vertices)
			# print('number of points on face = ', vertices.shape)
			

		if args.isPose or args.isShow:
			# estimate pose
			camera_matrix, pose = estimate_pose(vertices)
			np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose) 
			np.savetxt(os.path.join(save_folder, name + '_camera_matrix.txt'), camera_matrix) 

			np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose)

		if args.isShow:
			# ---------- Plot
			image_pose = plot_pose_box(image, camera_matrix, kpt)
			cv2.imshow('sparse alignment', plot_kpt(image, kpt))
			cv2.imshow('dense alignment', plot_vertices(image, vertices))
			cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
			cv2.waitKey(0)

		finalKpt = np.round(kpt).astype(np.int64)
		finalVert = np.round(vertices).astype(np.int64)
		print(finalKpt.shape)
		print(finalVert.shape)
		frameKpt.append(finalKpt[:,:2])
		frameVert.append(finalVert[:,:2])
	return frameKpt, frameVert

def deepSwap(args, srcImg, srcImg_gray, trgImg, trgImg_gray, name='DL_trg_'):
	# if args.isShow or args.isTexture:
	#     from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

	# ---- init PRN
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
	prn = PRN(is_dlib = args.isDlib)

	# ------------- load data
	# image_folder = args.inputDir
	# save_folder = args.outputDir
	# if not os.path.exists(save_folder):
	#     os.mkdir(save_folder)

	# types = ('*.jpg', '*.png', '*.jpeg')
	# image_path_list= []
	# for files in types:
	#     image_path_list.extend(glob(os.path.join(image_folder, files)))
	# total_num = len(image_path_list)

	# srcImg = cv2.imread(image_path_list[0])
	# srcImg_gray = cv2.imread(image_path_list[0],0)
	# trgImg = cv2.imread(image_path_list[1])
	# trgImg_gray = cv2.imread(image_path_list[1],0)
	srcVert= None
	trgVert= None

	# for i, image_path in enumerate(image_path_list):
	for i in range(2):
		# name = image_path.strip().split('/')[-1][:-4]

		# read image
		# image = imread(image_path)
		if(i==0):
			image = srcImg.copy()
		else:
			image = trgImg.copy()

		[h, w, c] = image.shape
		if c>3:
			image = image[:,:,:3]

		# the core: regress position map
		if args.isDlib:
			max_size = max(image.shape[0], image.shape[1])
			if max_size> 1000:
				image = rescale(image, 1000./max_size)
				image = (image*255).astype(np.uint8)
			pos = prn.process(image) # use dlib to detect face
		else:
			if image.shape[0] == image.shape[1]:
				image = resize(image, (256,256))
				pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
			else:
				box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
				pos = prn.process(image, box)
		
		image = image/255.
		if pos is None:
			continue

		if args.is3d or args.isMat or args.isPose or args.isShow:
			# 3D vertices
			vertices = prn.get_vertices(pos)
			if args.isFront:
				save_vertices = frontalize(vertices)
			else:
				save_vertices = vertices.copy()
			save_vertices[:,1] = h - 1 - save_vertices[:,1]

		if args.isImage:
			imsave(os.path.join(save_folder, name + '.jpg'), image)

		if args.is3d:
			# corresponding colors
			colors = prn.get_colors(image, vertices)

			if args.isTexture:
				if args.texture_size != 256:
					pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range = True)
				else:
					pos_interpolated = pos.copy()
				texture = cv2.remap(image, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
				if args.isMask:
					vertices_vis = get_visibility(vertices, prn.triangles, h, w)
					uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
					uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range = True)
					texture = texture*uv_mask[:,:,np.newaxis]
				write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture, prn.uv_coords/prn.resolution_op)#save 3d face with texture(can open with meshlab)
			else:
				write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

		if args.isDepth:
			depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
			depth = get_depth_image(vertices, prn.triangles, h, w)
			imsave(os.path.join(save_folder, name + '_depth.jpg'), depth_image)
			sio.savemat(os.path.join(save_folder, name + '_depth.mat'), {'depth':depth})

		if args.isMat:
			sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})

		if args.isKpt or args.isShow:
			# get landmarks
			kpt = prn.get_landmarks(pos)
			np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)
			np.savetxt(os.path.join(save_folder, name + '_vertices.txt'), vertices)
			# print('number of points on face = ', vertices.shape)
			

		if args.isPose or args.isShow:
			# estimate pose
			camera_matrix, pose = estimate_pose(vertices)
			np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose) 
			np.savetxt(os.path.join(save_folder, name + '_camera_matrix.txt'), camera_matrix) 

			np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose)

		if args.isShow:
			# ---------- Plot
			image_pose = plot_pose_box(image, camera_matrix, kpt)
			cv2.imshow('sparse alignment', plot_kpt(image, kpt))
			cv2.imshow('dense alignment', plot_vertices(image, vertices))
			cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
			cv2.waitKey(0)
		if(i==0):
			srcKpt = np.round(kpt).astype(np.int64)
			srcVert = np.round(vertices).astype(np.int64)
		else:
			trgVert = np.round(vertices).astype(np.int64)
			trgKpt = np.round(kpt).astype(np.int64)

	newSwap = swap(trgImg, srcImg, trgImg_gray, srcImg_gray, 
		srcKpt[:,0:2], None, trgKpt[:,0:2], None, srcVert[:,0:2], trgVert[:,0:2],
		debug = False, showW = 240, showH = 220)
	return newSwap



parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

parser.add_argument('-i', '--inputDir', default='TestImages/', type=str,
					help='path to the input directory, where input images are stored.')
parser.add_argument('-o', '--outputDir', default='TestImages/results', type=str,
					help='path to the output directory, where results(obj,txt files) will be stored.')
parser.add_argument('--gpu', default='0', type=str,
					help='set gpu id, -1 for CPU')
parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
					help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
parser.add_argument('--is3d', default=True, type=ast.literal_eval,
					help='whether to output 3D face(.obj). default save colors.')
parser.add_argument('--isMat', default=False, type=ast.literal_eval,
					help='whether to save vertices,color,triangles as mat for matlab showing')
parser.add_argument('--isKpt', default=False, type=ast.literal_eval,
					help='whether to output key points(.txt)')
parser.add_argument('--isPose', default=False, type=ast.literal_eval,
					help='whether to output estimated pose(.txt)')
parser.add_argument('--isShow', default=False, type=ast.literal_eval,
					help='whether to show the results with opencv(need opencv)')
parser.add_argument('--isImage', default=False, type=ast.literal_eval,
					help='whether to save input image')
# update in 2017/4/10
parser.add_argument('--isFront', default=False, type=ast.literal_eval,
					help='whether to frontalize vertices(mesh)')
# update in 2017/4/25
parser.add_argument('--isDepth', default=False, type=ast.literal_eval,
					help='whether to output depth image')
# update in 2017/4/27
parser.add_argument('--isTexture', default=False, type=ast.literal_eval,
					help='whether to save texture in obj file')
parser.add_argument('--isMask', default=False, type=ast.literal_eval,
					help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
# update in 2017/7/19
parser.add_argument('--texture_size', default=256, type=int,
					help='size of texture map, default is 256. need isTexture is True')

parser.add_argument('--swap_within_vid', type = int, default = 0, help="Set 1, When you want to swap two largest faces in one video")
parser.add_argument('--src_img_path', default="../Data/1.jpeg", help="Path to the COLOR image whose face you want to use as source")
parser.add_argument('--img_path', default="../Data/2.jpeg", help="Path to the COLOR image whose face you want to replace with source face")
parser.add_argument('--save_path', default="../Results/", help="Path where you want to save output (provide / at the end)")
parser.add_argument('--video_path', default="../Data/Test1.mp4", help="Path to the target video file")
args = parser.parse_args()

src = cv2.imread(args.src_img_path)
if type(src) is not np.ndarray:
	print('src_img_path wrong!!!')
	return
srcGray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


if(args.swap_within_vid !=1):
	trg = cv2.imread(args.img_path)
	if type(trg) is not np.ndarray:
		print('img_path wrong!!!')
		return
	trgGray = cv2.cvtColor(trg, cv2.COLOR_BGR2GRAY)
	swapImg = deepSwap(args, src, srcGray, trg, trgGray)
	cv2.imwrite(args.save_path +"Deep_Learn_face1.jpg",swapImg)
	return

cap = cv2.VideoCapture(args.video_path)
ret, trg= cap.read()
if ret == False:
	print('video_path wrong!!!')
	return
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vw = cv2.VideoWriter(args.save_path + "Deep_learning_PRNet_TPS.avi", fourcc, 30, (trg.shape[1], trg.shape[0]))
frameCount= 1
srcSwap =True

while (cap.isOpened()):
	ret, trg= cap.read()
	trgGray = cv2.cvtColor(trg, cv2.COLOR_BGR2GRAY)
	name = str(frameCount)
	if(args.swap_within_vid):
		kpt, vert = getPos(args, trg.copy(), 2, name) #getPos(args, frame, numberFace=1, name)
		if len(kpt)>1 and len(vert)>1:
			srcNew = trg.copy()
			srcNewGray = cv2.cvtColor(srcNew, cv2.COLOR_BGR2GRAY)

			newSwap0 = swap(trg, srcNew, trgGray, srcNewGray.copy(), 
					kpt[0], None, kpt[1], None, vert[0], vert[1],
					debug = False, showW = 240, showH = 220)

			newSwap0Gray = cv2.cvtColor(newSwap0, cv2.COLOR_BGR2GRAY)
			newSwap = swap(newSwap0, trg.copy(), newSwap0Gray, srcNewGray.copy(), 
					kpt[1], None, kpt[0], None, vert[1], vert[0],
					debug = False, showW = 240, showH = 220)
			cv2.imwrite(args.save_path +"Deep_Learn_face1.jpg",newSwap0)
			cv2.imwrite(args.save_path +"Deep_Learn_face2.jpg",newSwap)
			vw.write(newSwap)
		else:
			vw.write(trg)
	else:
		if(srcSwap):
			srcKpt, srcVert = getPos(args, src, 1, name)
			srcSwap = False
		else:
			pass
		trgKpt, trgVert = getPos(args, trg, 1, name)
		if len(trgKpt)>0 and len(trgVert)>0:
			newSwap = swap(trg, src, trgGray, srcGray, 
					srcKpt[0], None, trgKpt[0], None, srcVert[0], trgVert[0],
					debug = False, showW = 240, showH = 220)
			vw.write(newSwap)
		else:
			vw.write(trg)
	frameCount+=1
	print(frameCount)

vw.release()
cap.release()
