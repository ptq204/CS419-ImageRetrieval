# import the necessary packages
import numpy as np
import csv
import cv2
from image_processing.features_extractor import FeaturesExtractor
from scipy import spatial

class Searcher:
	def __init__(self, weightPath):
		# store our index path
		self.weightPath = weightPath

	def search(self, queryFeatures, limit = 10):
		# initialize our dictionary of results
		results = {}

		with open(self.weightPath) as f:
			
			reader = csv.reader(f)

			for row in reader:
				features = [float(x) for x in row[1:]]
				# d = self.chi2_distance(features, queryFeatures)
				d = self.cosine_distance(features, queryFeatures)
				results[row[0]] = d

			f.close()

		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		results = sorted([(v, k) for (k, v) in results.items()])

		# return our (limited) results
		return results[:limit]


	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d
	
	def cosine_distance(self, vectA, vectB):
		return spatial.distance.cosine(vectA, vectB)