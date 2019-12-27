# import the necessary packages
import numpy as np
import cv2
import imutils

class FeaturesExtractor:

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        orb = cv2.ORB_create()

        kp = orb.detect(image, None)

        kp, des = orb.compute(image, kp)

        return des
		