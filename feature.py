from image_processing.features_extractor import FeaturesExtractor
import argparse
import glob
import cv2
import numpy as np
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-f", "--features_db", required = True,
	help = "Path to the directory that stores images features")
ap.add_argument("-i", "--index", required = False,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

# initialize the color descriptor
#cd = FeaturesExtractor((8, 12, 3))

cd = FeaturesExtractor()

# open the output index file for writing
output = open(args["features_db"], "w")

# use glob to grab the image paths and loop over them
data = []
for imagePath in glob.glob(args["dataset"] + "/*.png"):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)

	# describe the image
	features = cd.describe(image)

	# tmp = [str([x for x in f]) for f in features]

	features = np.asarray(features)
	features = features.flatten()
	features = [str(x) for x in features]

	output.write("%s,%s\n" % (imageID, ",".join(features)))

# close the features_db file
output.close()