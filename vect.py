from image_retrieval.tfidf import TFIDF
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
ap.add_argument("-i", "--index", required = True,
    help = "Path to where the index will be stored")
ap.add_argument("-v", "--vector", required = True,
    help = "Path to where vectors presentation of images will be stored")

args = vars(ap.parse_args())

vctBuilder = TFIDF(args["index"])

vwDict = vctBuilder.getWordDict()

tfIdfDict = defaultdict(list)

output = open(args["vector"], "w")

numDocs = 0
tfList = []
with open(args["bovw"]) as fbovw:

    reader = csv.reader(fbovw)

    for row in reader:
        tf = vctBuilder.calTF(row[0], len(row) - 1)

        # Store (imageID, tfDict) to temp list
        tfList.append((row[0], tf))
        numDocs+=1

    fbovw.close()


# Calculate TF-IDF
idf = vctBuilder.calIDF(numDocs)

for tf in tfList:
    # Calculate tf-idf for each image
    vect = []
    for vword in vwDict:
        print("{} {}".format(tf[1][vword], idf[vword]))
        vect.append(tf[1][vword] * idf[vword])

    vect = [str(x) for x in vect]
    output.write("%s,%s\n" % (tf[0], ",".join(vect)))

output.close()


