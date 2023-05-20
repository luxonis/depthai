from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

class ImageScaler:
	scale_a: float = 1
	scale_b: float = 1
	translate_y_a: float = 0
	translate_y_b: float = 0

	def __init__(self, size_a: Tuple[int, int], size_b: Tuple[int, int]):
		"""
		Given the sizes of two images, calculate the scale and translation values to make them the same size.

		The size of the images must be given as (width, height).
		"""

		self.size_a = size_a
		self.size_b = size_b

		w_a, h_a = size_a
		w_b, h_b = size_b

		# calculate the scale factors
		if w_a > w_b:
			self.scale_a = w_b / w_a
			self.scale_size_a = (round(w_a * self.scale_a), round(h_a * self.scale_a))
			self.scale_b = 1
			self.scale_size_b = (w_b, h_b)
			self.scale_a = self.scale_size_a[0] / w_a # recalculate the scale factor to avoid rounding errors
		else:
			self.scale_b = w_a / w_b
			self.scale_size_b = (round(w_b * self.scale_b), round(h_b * self.scale_b))
			self.scale_a = 1
			self.scale_size_a = (w_a, h_a)
			self.scale_b = self.scale_size_b[0] / w_b # recalculate the scale factor to avoid rounding errors


		assert self.scale_size_a[0] == self.scale_size_b[0], "The width of the images should be the same after scaling."

		# calculate the translation values
		if self.scale_size_a[1] > self.scale_size_b[1]:
			self.target_size = self.scale_size_b
			self.translate_y_a = -((self.scale_size_a[1] - self.scale_size_b[1]) // 2)
			self.translate_y_b = 0
		else:
			self.target_size = self.scale_size_a
			self.translate_y_b = -((self.scale_size_b[1] - self.scale_size_a[1]) // 2)
			self.translate_y_a = 0

	def transform_img_a(self, img_a: np.ndarray):
		new_img_a = cv2.resize(img_a.copy(), self.scale_size_a, interpolation=cv2.INTER_CUBIC)
		crop_a = -self.translate_y_a
		if crop_a != 0:
			new_img_a = new_img_a[crop_a:self.target_size[1]+crop_a]

		return new_img_a
	

	def transform_img_b(self, img_b: np.ndarray):
		new_img_b = cv2.resize(img_b.copy(), self.scale_size_b, interpolation=cv2.INTER_CUBIC)
		crop_b = -self.translate_y_b
		if crop_b != 0:
			new_img_b = new_img_b[crop_b:self.target_size[1]+crop_b]

		return new_img_b
		
	def transform_img(self, img_a: np.ndarray, img_b: np.ndarray):
		"""
		Scale and translate the images to make them the same size.
		"""
		new_img_a = self.transform_img_a(img_a)
		new_img_b = self.transform_img_b(img_b)

		return new_img_a, new_img_b
	
	def transform_points(self, points_a: np.ndarray, points_b: np.ndarray):
		"""
		Scale and translate the points to align them with the scaled and cropped images. The points are expected to be in the format [[x,y], [x,y], ...].
		"""
		points_a = points_a.copy()
		points_b = points_b.copy()

		points_a[:, 0] *= self.scale_a
		points_a[:, 1] *= self.scale_a
		points_a[:, 1] += self.translate_y_a

		points_b[:, 0] *= self.scale_b
		points_b[:, 1] *= self.scale_b
		points_b[:, 1] += self.translate_y_b

		return points_a, points_b
	
	def transform_intrinsics(self, intrinsics_a: np.ndarray, intrinsics_b: np.ndarray):
		"""
		Scale the intrinsics to align them with the scaled images.
		"""
		intrinsics_a = intrinsics_a.copy()
		intrinsics_b = intrinsics_b.copy()

		intrinsics_a[0, :] *= self.scale_a
		intrinsics_a[1, :] *= self.scale_a
		intrinsics_a[1, 2] += self.translate_y_a

		intrinsics_b[0, :] *= self.scale_b
		intrinsics_b[1, :] *= self.scale_b
		intrinsics_b[1, 2] += self.translate_y_b

		return intrinsics_a, intrinsics_b