# USAGE
# python search.py --index index.csv --query queries/103100.png --result-path dataset

# import the necessary packages
from image_search.searcher import Searcher
from image_processing.features_extractor import FeaturesExtractor

import argparse
import cv2
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required = True,
	help = "Path to where the tf-idf vectors is stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-c", "--clusters", required = True,
	help = "Path to where the clusters file is stored")
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the dataset")

args = vars(ap.parse_args())

# initialize the image descriptor
cd = FeaturesExtractor()

# load the query image and describe it
query = cv2.imread(args["query"])
features = cd.describe(query)

# Build BoVW for query image
with open(args["clusters"], "rb") as fread:
	cluster = pickle.load(fread)
fread.close()

bovw = cluster.predict(features)

# perform the search
searcher = Searcher(args["weights"])
results = searcher.search(bovw)

# display the query
cv2.imshow("Query", query)

# loop over the results
for (score, resultID) in results:
	# load the result image and display it
	print("{} {}".format(resultID, score))
	result = cv2.imread(args["dataset"] + "/" + resultID)
	cv2.imshow("Result", result)
	cv2.waitKey(0)