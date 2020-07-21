# This program is for testing the model
# It can directly input the image to know the outcome where the model have trained

import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import glob


bg = None
global loaded_model

def count(thresholded):
    # print(np.shape(thresholded))

    thresholded = cv2.resize(thresholded, (50, 50))
    # print(np.shape(thresholded))
    # 降維度
    thresholded = thresholded[:, :, 0]
    thresholded = thresholded.reshape(-1, 50, 50, 1)

    thresholded = thresholded / 255
    prob = loaded_model.predict_classes(thresholded)
    return prob


# Main function
if __name__ == "__main__":
    # feed the image
    image = cv2.imread("D:/user/Desktop/Hand/test/one/hand (160).jpg")
    # load the structure of the model
    json_file = open('D:/user/Desktop/finaleditv/trainedModel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # 從 JSON 重建模型
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("D:/user/Desktop/finaleditv/modelWeights.h5")
    print("\n\n\n\nLoaded model from disk\n\n\n\n")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    fingers = count(image)
    print(fingers)
