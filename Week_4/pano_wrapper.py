#!/usr/bin/env python3

import math
import numpy as np
import cv2
import argparse
import os

class PanoramaStitching:
	def __init__(self):
		self.MatchingThreshold = 0.75
		self.RANSACThreshold = 0.99
		self.BlendingWidth = 3
		self.scale = 0.5
		self.vis = True
		self.finalblend = False
		self.loop = False

	def readInput(self):
		"""
		Read input images from the user
		"""
		parser = argparse.ArgumentParser(description='Panorama Stitching')
		parser.add_argument('--InputPath', type=str, help='Path to the input images', required=True)
		parser.add_argument('--Scale', type=float, help='Scale', required=True)
		args = parser.parse_args()
		self.InputPath = args.InputPath
		self.scale = args.Scale

	def readImages(self):
		"""
		Read images from a folder

		Args:
			InputImagePath: Path to the input images
			prefix: Prefix of the images to perform stichting

		Returns:
			Images: List of images
		"""
		Images = []
		files = os.listdir(self.InputPath)
		# Read only .jpg files
		files = [file for file in files if file.endswith('.jpg')]

		# Read images from the folder in the order of the file names
		files.sort()
		print(files)

		for file in files:
			img = cv2.imread(self.InputPath+'/'+file)

			# Resize the image by scaling factor inter cubic according to the img size
			img = cv2.resize(img, (int(img.shape[1] *self.scale) ,int(img.shape[0] * self.scale )), interpolation = cv2.INTER_CUBIC)
						
			if self.vis:
				cv2.imshow('Image', img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

			Images.append(img)

		return Images
	
	def displayImages(self, Images):
		"""
		Display images

		Args:
			Images: List of images
		"""
		for i in range(len(Images)):
			# Display the image in a new window with different window name with index of image
			cv2.imshow('Image_{0}'.format(i), Images[i])
		
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	def featureExtraction(self, image):
		"""
		Extract features from the images

		Args:
			Image: List of images

		Returns:
			kp: Keypoints,
			des: Descriptors
		"""		
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		sift = cv2.SIFT_create()
		kp, des = sift.detectAndCompute(gray, None)

		return kp, des

	def featureMatching(self, features1, descriptors1, features2, descriptors2):
		"""
		Match features from two images

		Args:
			Features1: List of features from image 1
			Features2: List of features from image 2

		Returns:
			Matches: List of matches
		"""
		Matches = []
		bf = cv2.BFMatcher()

		matches = bf.knnMatch(descriptors1, descriptors2, k=2)
		
		for m,n in matches:
			if m.distance < self.MatchingThreshold*n.distance:
				Matches.append(m)
		
		return Matches	
	
	def estimateHomography(self, matches, kp1, kp2):
		"""
		Estimate Homography between two images

		Args:
			matches: List of matches
			kp1: Keypoints of image 1
			kp2: Keypoints of image 2

		Returns:
			Homography: Homography matrix
			Mask: Mask
		"""
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
		H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7, self.RANSACThreshold)

		return H, mask
		
	def match_features_homography(self):
		"""
		Match features between all the images
		"""
		H_matrix = []

		for i in range(len(self.image_original)-1):

			# Feature Extraction for all the images
			kp1, des1 = self.featureExtraction(self.image_original[i])
			kp2, des2 = self.featureExtraction(self.image_original[i+1])

			img_key_1 = cv2.drawKeypoints(self.image_original[i], kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			img_key_2 = cv2.drawKeypoints(self.image_original[i+1], kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

			# Feature Matching
			matches = self.featureMatching(kp1, des1, kp2, des2)

			img_matches = cv2.drawMatches(self.image_original[i], kp1, self.image_original[i+1], kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

			if self.vis:
				cv2.imshow('Matches', img_matches)
				cv2.waitKey(0)	
				cv2.destroyAllWindows()
			
			if not self.finalblend:
				cv2.imwrite(self.InputPath+'/Matches/match_{0}.jpg'.format(i),img_matches)

			# Estimate Homography
			H, _ = self.estimateHomography(matches, kp1, kp2)

			H_matrix.append(H)
		
		return H_matrix
	
	def create_weighted_mask(self,mask):
		# Ensure the mask is single-channel and of type uint8
		if len(mask.shape) == 3:
			mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		if mask.dtype != np.uint8:
			mask = mask.astype(np.uint8)

		# Compute distance transform
		dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

		# Normalize the distance transform to [0, 1]
		max_dist = np.max(dist_transform)
		if max_dist > 0:
			weighted_mask = dist_transform / max_dist
		else:
			weighted_mask = dist_transform

		return weighted_mask
	
	def Pano(self,H_matrix):
		"""
		Stitch images to form a panorama
		"""

		H_matrix = np.array(H_matrix, dtype=np.float32)
		if H_matrix.dtype != np.float32 and H_matrix.dtype != np.float64:
			H_matrix = H_matrix.astype(np.float32)
		
		for i in range(len(self.image_original)-1):
			n = int(math.ceil(len(self.image_original)/2))
			
			if i+1 < n:
				H = H_matrix[i]
				img1 = self.image_original[i+1]
				img2 = self.image_original[i]

			else:
				H = np.linalg.inv(H_matrix[i])

				img1 = self.image_original[i]
				img2 = self.image_original[i+1]


			height1, width1 = img1.shape[:2]
			corner_points1 = np.array([[0,0], [0, height1], [width1, height1], [width1, 0]], dtype=np.float32).reshape(-1, 1, 2)

			height2, width2 = img2.shape[:2]
			corner_points2 = np.array([[0,0], [0, height2], [width2, height2], [width2, 0]], dtype=np.float32).reshape(-1, 1, 2)

			# Warp the image2 
			corner_points2_ = cv2.perspectiveTransform(corner_points2, H)
			result_corner_points = np.concatenate((corner_points1, corner_points2_), axis=0)

			[x_min, y_min] = np.int32(result_corner_points.min(axis=0).ravel() - 0.5)
			[x_max, y_max] = np.int32(result_corner_points.max(axis=0).ravel() + 0.5)
			
			t = [-x_min, -y_min]
			Ht = np.array([[1, 0, t[0]], 
				  		   [0, 1, t[1]], 
						   [0, 0, 1]],dtype=np.float32)

			# Warp the images
			img1_warp = cv2.warpPerspective(img1, Ht, (x_max-x_min, y_max-y_min))
			img2_warp = cv2.warpPerspective(img2, Ht.dot(H), (x_max-x_min, y_max-y_min))
			
			# Create a mask on warped images replacing pixel with non zero pixel values set it to 255
			mask1 = (img1_warp > 0).astype(np.uint8) * 255
			mask2 = (img2_warp > 0).astype(np.uint8) * 255

			weighted_mask1 = self.create_weighted_mask(mask1)
			weighted_mask2 = self.create_weighted_mask(mask2)
			
			result = (img1_warp * weighted_mask1[..., None] + img2_warp * weighted_mask2[..., None]) /\
                (weighted_mask1[..., None] + weighted_mask2[..., None])

			if self.finalblend:
				cv2.imwrite(self.InputPath+'/Final_Panorama.jpg',result)
			
			elif self.finalblend==False:
				if self.loop:
					cv2.imwrite(self.InputPath+'/{0}.jpg'.format(i),result)
			
				elif self.loop==False:
					cv2.imwrite(self.InputPath+'/Pano/{0}.jpg'.format(i),result)

	def recursive(self):
		# Read the images from the Pano folder
		# Remove all the images from the Pano folder
		# Stitch the images to form a panorama
		# Repeat the process until all the images are stitched together to form a panorama image

		self.InputPath = self.InputPath+'/Pano/'
		# self.scale = 0.7

		while True:
			# Read the images from the Pano folder
			self.image_original = self.readImages()
			self.loop = True

			# Remove all the images from the folder .jpg
			for f in os.listdir(self.InputPath):
				if f.endswith('.jpg'):
					os.remove(self.InputPath + f)

			# Repeat the process until all the images are stitched together to form a panorama image
			if len(self.image_original) > 2:
				self.finalblend = False
				H_matrix = self.match_features_homography()
				self.Pano(H_matrix)

			elif len(self.image_original) == 2:
				self.loop = True
				self.finalblend = True
				H_matrix = self.match_features_homography()
				self.Pano(H_matrix)
				break

	def main(self):
		# Read input variables from command line
		self.readInput()

		# Read a set of images for Panorama stitching
		self.image_original = self.readImages()
		# self.displayImages(self.image_original)

		# Loop pairwise until all images are stitched to form a panorama image
		H_matrix = self.match_features_homography()
		self.Pano(H_matrix)

		# Write a loop which adjusts depth of iteration based on the number of images
		self.recursive()

if __name__ == "__main__":
	PS = PanoramaStitching()
	PS.main()