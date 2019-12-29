from image_processing.clustering import Cluster
import argparse
import glob
import cv2
import numpy as np
import csv
import pickle, pprint

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features_db", required = True,
	help = "Path to the directory that stores images features")
ap.add_argument("-c", "--clusters", required = True,
    help = "Path to where the clusters will be stored")
args = vars(ap.parse_args())

print('Getting images features...')
data = []
with open(args["features_db"]) as f:
    reader = csv.reader(f)

    for row in reader:
        features = [float(x) for x in row[1:]]
        features = np.asarray(features)

        # each feature produced by ORB method is a vector of length 32
        features = features.reshape(len(features)//128, 128)
        for p in features:
            data.append(p)

    f.close()

# Perform clustering
print('Clustering...')
data = np.asarray(data)

# numCluster = len(data) // 100
numCluster = 150
clt = Cluster(numCluster)
cluster = clt.cluster(data)


print('Saving clusters...')
with open(args["clusters"], "wb") as fout:
    pickle.dump(cluster, fout, pickle.HIGHEST_PROTOCOL)
fout.close()

