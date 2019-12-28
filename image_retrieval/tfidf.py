import pickle
import math
import csv
import numpy as np

class TFIDF():
    def __init__(self, indexPath):
        with open(indexPath, "rb") as f_index:
            self.index = pickle.load(f_index)
        f_index.close()
        self.vwordDict = [i for i in range(len(self.index))]

    def getWordDict(self):
        return self.vwordDict

    def calTF(self, imageID, nFeatures):
        tfDict = {}
        for vword in self.vwordDict:
            check = 0
            for imgDoc in self.index[vword]:
                if imgDoc[0] == imageID:
                    tfDict[vword] = imgDoc[1] / float(nFeatures)
                    check = 1
                    break
            if check == 0:
                tfDict[vword] = 0

        return tfDict
    
    def calIDF(self, numDocs):
        idfDict = {}
        for vword in self.vwordDict:
            idfDict[vword] = 1 + math.log(numDocs / float(len(self.index[vword])))
        
        # Save idf values because we just need to cal it once
        file = open("output/idf.csv", "w")
        for vword in idfDict:
            file.write("%s,%s\n" % (vword, str(idfDict[vword])))
        file.close()
        return idfDict

    def calTFQuery(self, features):
        tfDict = {}
        size = len(features)
        for vwrod in self.vwordDict:
            tfDict[vwrod] = (features == vwrod).sum() / size
        print(tfDict)
        return tfDict

    def calTF_IDFQuery(self, features):
        tf = self.calTFQuery(features)
        tfIdf = []
        with open("output/idf.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                tfIdf.append(tf[int(row[0])] * float(row[1]))
        return np.asarray(tfIdf)
        