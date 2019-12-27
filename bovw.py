import argparse
import glob
import cv2
import numpy as np
import pickle
import csv

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features_db", required = True,
	help = "Path to the directory that stores images features")
ap.add_argument("-c", "--clusters", required = False,
	help = "Path to where the clusters will be stored")
ap.add_argument("-b", "--bovw", required = False,
	help = "Path to where the bag of visual words will be stored")
args = vars(ap.parse_args())


# Load centers after clustering
with open(args["clusters"], "rb") as f_read:
    cluster = pickle.load(f_read)
f_read.close()

output = open(args["bovw"], "w")

bovw = []
with open(args["features_db"]) as f:
    reader = csv.reader(f)

    for row in reader:
        features = [int(x) for x in row[1:]]
        features = np.asarray(features)

        # each feature produced by ORB method is a vector of length 32
        features = features.reshape(len(features)//32, 32)

        label = cluster.predict(features)
        label = [str(x) for x in label]
        
        output.write("%s,%s\n" % (row[0], ",".join(label)))

    f.close()

output.close()
