import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from PIL import Image
import os
import glob
from keras.preprocessing.image import img_to_array, load_img
# all the library use to read image

# set the label
labels_dir = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
# set the picture size
size = (50, 50)
# set the number of picture capture from the folder
# However, it will end until reading the whole folder
# just set the number
read_data_num = 3600
for folders in glob.glob("D:/user/Desktop/Hand/train/*"):

    print(folders)
    images = []
    labels_hot = []
    labels = []
    read_data_num_i = 1 # control number of data we can read
# read all the  whole sub folder

    for filename in os.listdir(folders):

        if  read_data_num_i <= read_data_num:
            label = os.path.basename(folders)
            # print(label)
            className = np.asarray(label)
            img = load_img(os.path.join(folders, filename))
            img = img.resize(size, Image.BILINEAR)
            # print(np.shape(img))

            if img is not None:
                if label is not None:
                    labels.append(className)
                    labels_hot.append(labels_dir[label])
                    print(label)
                    print(labels_dir[label])
                x = img_to_array(img)
                images.append(x)
            read_data_num_i += 1

    images = np.array(images)
    # 降維度
    images = images[:, :, :, 0]
    print(images.shape)
    # images.resize(images,(50,50))
    labels_hot = np.array(labels_hot)
    print(labels_hot)
    print("images.shape={}, labels_hot.shape=={}".format(images.shape, labels_hot.shape))
    imagesavepath = " D:/user/Desktop/Hand/train/"
    if not os.path.exists(imagesavepath):
        os.makedirs(imagesavepath)
    np.save(imagesavepath + '{}_images.npy'.format(label), images)
    # np.save(imagesavepath + '{}_label.npy'.format(label), labels)
    np.save(imagesavepath + '{}_labels_hot.npy'.format(label), labels_hot)
    print('{} files has been saved.'.format(label))

