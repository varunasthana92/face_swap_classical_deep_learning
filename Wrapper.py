from utils import tps, delaunay
import argparse

Parser = argparse.ArgumentParser()
Parser.add_argument('--method', default="TPS", help="Which method for warping")
Parser.add_argument('--swap_vid', type = int, default=0, help="Set 1, when you want to input a target video. Also use arguemnt video_path")
Parser.add_argument('--video_path', default="Data/Test1.mp4", help="Path to the target video file")
Parser.add_argument('--swap_within_vid', type = int, default = 0, help="Set 1, When you want to swap two largest faces in one video")
Parser.add_argument('--src_img_path', default="Data/1.jpeg", help="Path to the COLOR image whose face you want to use as source")
Parser.add_argument('--img_path', default="Data/2.jpeg", help="Path to the COLOR image whose face you want to replace with source face")
Parser.add_argument('--save_path', default="Results/", help="Path where you want to save output (provide / at the end)")
Args = Parser.parse_args()

if Args.method=="TPS":
	Args.save_path += 'tps_'
	print('Method = TPS')
	tps.main(Args)
else:
	Args.save_path += 'delaunay_'
	print('Method = Delaunay')
	delaunay.main(Args)

