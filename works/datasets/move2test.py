from pathlib import Path
import shutil
import random


def move_to_test_dataset(source_dir: Path, target_dir: Path, move_num: int):
	"""
	Randomly move a specified number of files from the source directory to the test directory.

	:param source_dir: Path to the source directory containing the files.
	:param target_dir: Path to the test directory where files will be moved.
	:param move_num: Number of files to move from source to test directory.
	"""
	if not source_dir.exists() or not source_dir.is_dir():
		raise ValueError(f"Source directory {source_dir} does not exist or is not a directory.")

	files = [f for f in source_dir.iterdir() if f.is_file() and ".png" in f.name]
	
	if move_num > len(files):
		raise ValueError(f"Not enough files in {source_dir} to move. Requested: {move_num}, Available: {len(files)}")

	# Randomly select files to move
	random.shuffle(files)
	# Move the specified number of files
	for file in files[:move_num]:
		shutil.move(str(file), str(target_dir / file.name))
		print(f"Moved file: {file.name} to {target_dir / file.name}")


if __name__ == "__main__":
	source_directory = Path("crack_teeth_swift_normal_bbox/normal")
	target_directory = Path("crack_teeth_swift_normal_bbox_test/normal")
	move_num = 150

	move_to_test_dataset(source_directory, target_directory, move_num)
	print(f"Moved {move_num} files from {source_directory} to {target_directory}.")

	# all_train_images = [f.name for _dir in Path("crack_teeth_swift_normal_bbox").iterdir() if _dir.is_dir() and _dir.name != "normal"  for f in _dir.iterdir() if f.is_file() and f.suffix in [".png", ".jpg"]]
	# all_test_images = [f.name for _dir in Path("crack_teeth_swift_normal_bbox_test").iterdir() if _dir.is_dir() and _dir.name != "normal"  for f in _dir.iterdir() if f.is_file() and f.suffix in [".png", ".jpg"]]

	# all_test_images_normal = [f.name for f in Path("crack_teeth_swift_normal_bbox_test/normal").iterdir() if f.is_file() and f.suffix in [".png", ".jpg"]]
	# for normal_file in all_test_images_normal:
	# 	if normal_file in all_train_images or normal_file in all_test_images:
	# 		print(f"................{normal_file}")
	# 		continue
	# 	else:
	# 		print(normal_file)