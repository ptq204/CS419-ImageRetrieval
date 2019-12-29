# import the necessary packages
import numpy as np
import cv2
import imutils

class FeaturesExtractor:

    def describe_ORB(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        orb = cv2.ORB_create()
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)
        return des

    def describe_SIFT(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(image, None)
        kp, des = sift.compute(image, kp)
        return des
		