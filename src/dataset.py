import os
import sys

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io.image import read_image


IMG_DIR = 'images/geometric_dataset'
IMG_FILE_ENDING = '.png'

# lehl@2021-12-31: Based on the documentation for custom PyTorch datasets at
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
#
class CustomImageDataset(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
		self.df = pd.read_csv(annotations_file)
		self.labels = sorted(self.df.label.unique())
		
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		image = read_image(self.df.loc[idx, 'file_path'])
		label = self.labels.index(self.df.loc[idx, 'label'])

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		
		return image, label

def main():
	images_dict = list()
	for subdir, dirs, files in os.walk(IMG_DIR):
		for file in files:
			file_path = os.path.join(subdir, file)

			if file_path.endswith(IMG_FILE_ENDING):
				label, object_nr, xrot, yrot, zrot = file.split(IMG_FILE_ENDING)[0].split('_')

				images_dict.append({
					'label': label,
					'file_path': file_path,
					'image_object_nr': object_nr,
					'rotation': { 'x': xrot[1:], 'y': yrot[1:], 'z': zrot[1:] },
					'scale': { 'x': 1.0, 'y': 1.0, 'z': 1.0 },
					'translation': { 'x': 0.0, 'y': 0.0, 'z': 0.0 }
				})

	df = pd.DataFrame(images_dict)
	df = df.sort_values(['label','image_object_nr'])
	df.reset_index(inplace=True, drop=True)
	# df.to_csv(os.path.join(IMG_DIR, 'annotations.csv'), index=False)

	d = CustomImageDataset(os.path.join(IMG_DIR, 'annotations.csv'), IMG_DIR)


if __name__ == '__main__':
	main()
