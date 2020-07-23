import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from PIL import Image
import os
import glob
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD,RMSprop
import cv2
from random import shuffle

images_all=[]
labels_all=[]

#Read one npy file
images_one = np.load('E:/user/Desktop/Hand/train/Handone_images.npy')
images_all = images_one
labels_one = np.load('E:/user/Desktop/Hand/train/Handone_labels_hot.npy')
labels_all = labels_one
# print(images_all[2].shape)
# gray= cv2.cvtColor(images_all[2], cv2.COLOR_BGR2GRAY)
# print(gray.shape)


#Read two npy file
#axis 0 - row
images_two = np.load('E:/user/Desktop/Hand/train/Handtwo_images.npy')
images_all = np.append(images_all, images_two, axis=0)
labels_two = np.load('E:/user/Desktop/Hand/train/Handtwo_labels_hot.npy')
labels_all = np.append(labels_all, labels_two, axis=0)

#Read three npy file
images_three = np.load('E:/user/Desktop/Hand/train/Handthree_images.npy')
images_all = np.append(images_all, images_three, axis=0)
labels_three = np.load('E:/user/Desktop/Hand/train/Handthree_labels_hot.npy')
labels_all = np.append(labels_all, labels_three, axis=0)

#Read four npy file
images_four = np.load('E:/user/Desktop/Hand/train/Handfour_images.npy')
images_all = np.append(images_all, images_four, axis=0)
labels_four = np.load('E:/user/Desktop/Hand/train/Handfour_labels_hot.npy')
labels_all = np.append(labels_all, labels_four, axis=0)

#Read five npy file
images_five = np.load('E:/user/Desktop/Hand/train/Handfive_images.npy')
# print(images_all.shape)
images_all = np.append(images_all, images_five, axis=0)
# print(images_five.shape)
labels_five = np.load('E:/user/Desktop/Hand/train/Handfive_labels_hot.npy')
# print(labels_five)
# #[5 5 5 ... 5 5 5]
labels_all = np.append(labels_all, labels_five, axis=0)
# print(labels_all.shape)
#[1 1 1 ... 5 5 5]

# new added
# print("images_all.shape before reshape")
# print(images_all.shape) -> (18000, 50, 50)
# Expend dimension for 1 channel image
images_all = images_all.reshape(-1,50,50,1)
# print("After / 255")
# print(images_all)

images_all = images_all / 255
# print("Before / 255")
# print(images_all)
#print(images_all.shape) -> (18000, 50, 50, 1)


# num = images_all.shape[0]
# for i in range(num-1):
#     # print(images_all[i].shape)
#     image = cv2.cvtColor(images_all[i], cv2.COLOR_BGR2GRAY)
#     print(image.shape)
#     if i ==0:
#         images_all_gray = image
#     else:
#         images_all_gray = np.append(images_all_gray,image,axis=0)
#
# print(images_all_gray.shape)

print("labels_all.shape before reshape")
print(labels_all.shape)

# one-hot-encoding
labels_all = np_utils.to_categorical(labels_all)
print(labels_all)
# [[0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0.]
#  ...
#  [0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 1.]]

# define baseline model
def baseline_model():
    # create model
    # Generate model
    model = Sequential()
    # input: 50x50 images with 3 channels -> (50, 50, 1) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 1), padding='same', name='block1_conv2_1'))
    # print(images_all.shape)
    # print("labels_all.shape")
    # filter : 128 , kernel_size: 3 * 3 , padding='same'
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block1_MaxPooling'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2_1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block2_MaxPooling'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='final_output_1'))
    model.add(Dropout(0.5))
    # model.add(Dense(128, activation='relu', name='final_output_2'))
    # model.add(Dropout(0.5))
    model.add(Dense(6, activation='sigmoid', name='class_output'))
    optimizer = RMSprop(lr=1e-4)
    objective = 'categorical_crossentropy'
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    # EStop = EarlyStopping(monitor='val_acc', min_delta=0,patience=10, verbose=1, mode='auto')
    return model

# build the model
model = baseline_model()
# Fit the model
# Shuffle coefficient for all data 手動打亂 (兩個打亂次序相同)
allran_num = np.arange(0,18000)
np.random.shuffle(allran_num)
allInt = allran_num.astype(int)
Index= allInt.tolist()
images_all = images_all[Index]
labels_all = labels_all[Index]


# model.fit(images_all, labels_all, batch_size=64, epochs=10, shuffle= True)
model.fit(images_all, labels_all, batch_size=64, epochs=10, validation_split=0.3)


# Save the model
model_json = model.to_json()
with open("trainedmodel.json", "w") as jsonFile:
    jsonFile.write(model_json)
model.save_weights("modelWeights.h5")
print("Saved model to disk")
