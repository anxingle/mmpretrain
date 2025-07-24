import torch
from pathlib import Path
from PIL import Image
import copy
import numpy as np
import cv2
import torch.nn.functional as F
import time

from nets.deeplabv3_plus import DeepLab


class model():
	def __init__(self):
		# 加载模型
		# self.net=DeepLab(num_classes=2, backbone="mobilenet", pretrained=False, downsample_factor=8)
		# self.net.load_state_dict(torch.load("./tongue_seg.pth", map_location="cpu"))
		self.net=torch.load("tongue_seg.pth", map_location="cpu", weights_only=False)
		# self.net=torch.load("models/tongue_seg.pth", map_location="cpu")
		# 确定是否有GPU运行环境
		self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.net=self.net.to(self.device)
		# 输入模型的图片尺寸
		self.input_shape=(512,512)

	def predict(self, image):
		# 模型进入eval模式
		self.net.eval()

		#   对输入图像进行一个备份，方便后续对原图进行更改
		old_img     = copy.deepcopy(image)
		#   获取原图的宽和高
		print(f"")
		orininal_h  = np.array(image).shape[0]
		orininal_w  = np.array(image).shape[1]

		#   并将图片缩放到512*512
		image_data=image.resize((self.input_shape[1],self.input_shape[0]))
		
		# 将图片归一化，交换维度
		temp=np.array(image_data, np.float32)/255.0
		image_data  = np.expand_dims(np.transpose(temp, (2, 0, 1)), 0)
		with torch.no_grad():
			a=time.time()
			# 转化为张量，并在指定环境(cpu或GPU)下运算
			images=torch.from_numpy(image_data).to(self.device)
			pr = self.net(images)[0]
			# 将结果进行softmax运算，获得各个类别的概率
			pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
			# 将结果缩放回原图大小
			pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)

			pr = pr.argmax(axis=-1)
			seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8') # 在原图上进行更改
			# 将新图片转换成Image的形式
			image = Image.fromarray(np.uint8(seg_img))

		return image


if __name__ == "__main__":
	mdl=model()
	tiny_datasets_dir = Path(__file__).parent / "datasets/color6000_0717"
	save_datasets_dir = Path(__file__).parent / "datasets/color6000_0717_cropped"
	all_images = [image_file for _dir in Path(tiny_datasets_dir).iterdir() if _dir.is_dir() for image_file in Path(_dir).iterdir() if ".png" in image_file.suffix]

	try:
		for img_file in all_images:
			input_img = Image.open(img_file)
			image = mdl.predict(input_img)
			img_np = np.array(image)
			img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
			parent_dir = img_file.parent
			save_dir = save_datasets_dir / parent_dir.name
			save_dir.mkdir(parents=True, exist_ok=True)
			save_path = save_dir / img_file.name
			cv2.imwrite(str(save_path), img_cv)
			print(f"Processed {img_file} and saved to {save_path}")
			# cv2.imshow("Result", img_cv)
			# cv2.waitKey(1500)  # 显示2秒
			# cv2.destroyAllWindows()
	# 捕捉键盘 Ctrl+c 键
	except KeyboardInterrupt:
		print("检测到 Ctrl+C,程序终止")
		break