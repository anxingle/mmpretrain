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
	source_directory = Path("color6000_0717_cropped_bbox/white")
	target_directory = Path("test_color6000_0717_cropped_bbox/white")
	move_num = 15

	move_to_test_dataset(source_directory, target_directory, move_num)
	print(f"Moved {move_num} files from {source_directory} to {target_directory}.")