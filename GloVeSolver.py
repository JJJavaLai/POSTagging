# @Time : 2021/12/6 下午9:09
# @Author : Patrick.Lai
# @File : ExtractData.py
# @Software: PyCharm
import numpy as np
import os
import pickle


class GloveSolver:
    def __init__(self):
        self.file_route = "glove/glove-wiki-gigaword-300.txt"
        self.vectors = {}
        print("Open GloVe file, file route:", self.file_route)
        print("Mapping GloVe vector into matrix")
        with open(self.file_route, encoding="utf-8") as f:
            for line in f:
                data = line.split()
                word = data[0]
                vector = np.asarray(data[1:], dtype = "float32")
                self.vectors[word] = vector
        print("Map finished, matrix length: ", len(self.vectors))
        if not os.path.exists("PickledData/"):
            print("Making directory PickledData/ to save pickled GloVe file")
            os.mkdir("PickledData/")
        with open("PickledData/GloVe.pkl", "wb") as glove_file:
            print("Pickling and save file")
            pickle.dump(self.vectors, glove_file)
            print("Pickled file saved")


g = GloveSolver()

print(1)
