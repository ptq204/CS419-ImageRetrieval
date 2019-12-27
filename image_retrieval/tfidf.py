import pickle
import math

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
        return idfDict