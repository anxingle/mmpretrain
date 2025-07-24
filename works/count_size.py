import torch
from pathlib import Path
from PIL import Image
import copy
import shutil
import numpy as np
import cv2
import random
import torch.nn.functional as F
import time
from _logger import logger, add_file_handler_to_logger


def find_connected_components(img: np.ndarray, min_size: int = 100, shift: int = 25) -> np.ndarray:
	# 生成前景掩码：非黑区域设为1
	foreground_mask = np.any(img != [0, 0, 0], axis=2).astype(np.uint8)
	# 连通域分析
	# stats: [label, x, y, width, height, area]
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground_mask, connectivity=8)
	# 排除背景（label=0），找最大面积的连通域
	# 第一个是背景，不考虑
	largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 注意偏移+1
	# 生成最大连通区域的mask
	tongue_mask = (labels == largest_label).astype(np.uint8) * 255  # 0或255图像
	# 获取 bounding box 并裁剪
	x, y, w, h, area = stats[largest_label]
	cropped_tongue = cv2.bitwise_and(img, img, mask=tongue_mask)
	return cropped_tongue
	# # 保存结果
	# cv2.imwrite("tongue_largest_region.png", cropped_tongue)



def bbox_img_func(img: np.ndarray, shift_v: int = 25):
	assert img is not None, "图片读取失败"
	# 先找出最大连通域
	max_img = find_connected_components(img, min_size=100, shift=shift)
	# 创建一个mask：非黑色区域为1，黑色区域为0
	# 条件是BGR三个通道都不为0
	mask = np.any(max_img != [0, 0, 0], axis=2).astype(np.uint8)  # shape: (H, W), 0/1

	# 找出非黑区域的边界（bounding box）
	coords = cv2.findNonZero(mask)  # 返回所有非零点的坐标
	x, y, w, h = cv2.boundingRect(coords)  # 获取包含舌头的最小矩形
	shift = random.randint(0, shift_v)  # 裁剪时的额外偏移量
	shift_y = random.randint(0, shift_v)
	# 裁剪原图
	try:
		if y < shift:
			shift = y
		if x < shift:
			shift = x
		tongue = max_img[y-shift_y: y+h+shift_y, x-shift: x+w+shift]
	except Exception as e:
		tongue = max_img[y-2: y+h+2, x-2: x+w+2]
		logger.error(f"Error cropping image: {e}, using fallback crop")
	# max_tongue = find_connected_components(tongue, min_size=100, shift=shift)

	# cv2.imwrite("../tongue_cropped_5.png", tongue)  # 保存裁剪后的图片
	# 保存或者显示结果
	return tongue


def rsync_func():
	img_base_dir = Path("/home/an/mmpretrain/works/datasets/color6000_0717_cropped_error/")
	all_lines = []
	with open("logs/error.log", "r") as f:
		lines = f.readlines()
		for line in lines:
			_content = line.strip().split("69 - Image ")[-1]
			_cc = _content.split(" is too small: ")[0]
			source_img = _cc.replace("cropped_bbox", "cropped")
			category = Path(_cc).parent.name
			Path(img_base_dir / category).mkdir(parents=True, exist_ok=True)
			new_file = img_base_dir / category / Path(_cc).name
			shutil.copy(str(source_img), str(new_file))


if __name__ == "__main__":
	add_file_handler_to_logger(name="bbox_pre", dir_path="./logs", level="INFO")
	rsync_func()
	exit()
	tiny_datasets_dir = Path(__file__).parent / "datasets/color6000_0717_cropped"
	save_datasets_dir = Path(__file__).parent / "datasets/color6000_0717_cropped_bbox"
	all_images = [image_file for _dir in Path(save_datasets_dir).iterdir() if _dir.is_dir() for image_file in Path(_dir).iterdir() if ".png" in image_file.suffix]
	print(all_images)
	print(tiny_datasets_dir)
	error_img_list = ["5470", "5485", "5486", "5923", "6002", "6003", "3839", "3875", "5450", "5881", "5886", "5889", "5982", "5890", "6005", "6020", "6086", "6103", "3669", "3677", "3687", "5967"]

	for img_file in all_images:
		# _temp_name = img_file.name.split(".png")[0]
		# if _temp_name not in error_img_list:
		# 	# logger.error(f"Skipping error image: {img_file}")
		# 	continue
		# _img = cv2.imread(str(img_file))
		# bbox_img = bbox_img_func(img=_img, shift=35)
		# parent_dir = img_file.parent
		# save_dir = save_datasets_dir / parent_dir.name
		# save_dir.mkdir(parents=True, exist_ok=True)
		# save_path = save_dir / img_file.name
		# try:
		# 	cv2.imwrite(str(save_path), bbox_img)
		# 	print(f"Processed {img_file} and saved to {save_path}")
		# except Exception as e:
		# 	logger.error(f"Error saving image {img_file}: {e}")
		# 	shape = bbox_img.shape if bbox_img is not None else "Unknown shape"
		# 	logger.error(f"{img_file} shape: {shape}")

		input_img = Image.open(img_file)
		# get image size
		width, height = input_img.size
		if width < 200 or height < 200:
			logger.error(f"Image {img_file} is too small: {width}x{height}")
			continue
		print(f"Processing image: {img_file}, size: {width}x{height}")
