from image_processing.clustering import Cluster
import argparse
import glob
import cv2
import numpy as np
import csv
from collections import defaultdict
import pickle, pprint

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bovw", required = True,
	help = "Path to the bag of visual words directory")
ap.add_argument("-c", "--clusters", required = True,
    help = "Path to where the clusters will be stored")
ap.add_argument("-i", "--index", required = True,
    help = "Path to where the index will be stored")

args = vars(ap.parse_args())

# Build inverted index
with open(args["clusters"], "rb") as f_read:
    cluster = pickle.load(f_read)
f_read.close()

bovw = {}
with open(args["bovw"]) as f:
    reader = csv.reader(f)

    for row in reader:
        features = [int(x) for x in row[1:]]
        bovw[row[0]] = features

    f.close()

centers = cluster.cluster_centers_
centers_size = len(centers)

index = {}
for label in range(centers_size):
    for imageID in bovw:
        if label in bovw[imageID]:
            if label not in index:
                index[label] = [(imageID, bovw[imageID].count(label))]
            else:
                index[label].append((imageID, bovw[imageID].count(label)))

with open(args["index"], "wb") as f_out:
    pickle.dump(index, f_out, pickle.HIGHEST_PROTOCOL)
f_out.close()