# This code is for python and flsk
from flask import Flask, render_template, Response
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
# from tensorflow.python.keras import layers
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from keras import models
import tensorflow as tf
import os
import time
# from camera import VideoCamera

bg = None
global loa_model


# windows環境下,tf不分配所有可用的記憶體所以要手動設定
# 允許動態記憶體分配增長。
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# Function - To segment the region of hand in the image
def segment(image, threshold=30):
        global bg
        # find the absolute difference between background and current frame
        # 前景背景分離-將連續兩個frame的差取絕對值
        diff = cv2.absdiff(bg.astype("uint8"), image)
        # diff畫面
        # cv2.imshow("diff = grey - bg", diff)
        # 灰度圖畫面
        # cv2.imshow("grey", image)
        '''
        # 對變化的影象進行閾值化，膨脹閾值影象來填補 thresholded 只能用於灰階
        # ret,mask = cv2.threshold(img2gray,175,255,cv2.THRESH_BINARY)
        # ret:暫時就認爲是設定的thresh閾值，mask：二值化的圖像
        # 灰度圖diff中灰度值小於threshold = 30的點置0 (BLACK)，灰度值大於30的點設255 (WHITE)。
        '''
        # 將圖片轉二值化
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

        # 輪廓找尋 : 輪廓數 cnts / _輪廓之間關係 - ( 原圖像, 輪廓的檢索模式 (cv2.RETR_EXTERNAL 只檢測外輪廓) ,輪廓的近似方法(只保留該方向的終點坐標) )
        (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            return
        else:
            # 取得Contours面積的指令：contourArea(Contours物件) 單位pixel。
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)

# Function - here's where the main recognition work happens
def count(thresholded, segmented):
        # resize 給定一個數組和特定維度會返回一個給定維度形式的新陣列
        # reshape 改變陣列形狀 如果給定的陣列資料和需要reshape的形狀不符合時會報錯
        thresholded = cv2.resize(thresholded, (50, 50))
        thresholded = thresholded.reshape(-1, 50, 50, 1)
        # print(np.shape(thresholded))

        thresholded = thresholded / 255
        # 因為圖像經過one hot 所以預測的是類別
        prob = loaded_model.predict_classes(thresholded).astype('int')
        # time.sleep(5)
        return prob

# background subtraction: 使用 accumulateWeighted 提取前景容易受到背景擾動
def run_avg(image, accumWeight):
        global bg
        if bg is None:
            bg = image.copy().astype("float")
            return
        # accumulateWeighted (img , accumulator image, Weight of the input image)
        cv2.accumulateWeighted(image, bg, accumWeight)

# FLASK
app = Flask(__name__)

class VideoCamera(object):
    def __init__(self):
        # 停五秒再開鏡頭
        time.sleep(5)
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # free up memory
        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        # success, image = self.video.read()
        # 設置拍攝框大小
        top, right, bottom, left = 10, 350, 225, 590
        # initialize num of frames
        num_frames = 0

        # calibration indicator
        calibrated = False

        while (True):
            # 從鏡頭拍攝圖像 grabbed:代表是否成功 /  frame :攝影機單張畫面
            (grabbed, frame) = self.video.read()

            # 圖像處理 resize圖像大小
            frame = imutils.resize(frame, width=700)

            # 因為拍照後圖像會左右顛倒,所以需要利用flip進行圖像翻轉 1:水平橫向翻轉
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

            # shape[0]: row / shape[1]:column / shape[:2]取圖像的長寬
            (height, width) = frame.shape[:2]

            # get the ROI 圖像陣列
            roi = frame[top:bottom, right:left]

            # 透過轉換函式轉成灰階圖像
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 高斯模糊 - 高斯矩陣[7*7] 標準差0 (圖像平滑,去雜訊)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our weighted average model gets calibrated
            accumWeight = 0.5
            if num_frames < 30:
                run_avg(gray, accumWeight)
                if num_frames == 1:
                    print(">>>Please wait! Program is calibrating the background...")
                elif num_frames == 29:
                    print(">>>Calibration successfull. ...")
            else:
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    (thresholded, segmented) = hand
                    # draw the segmented region and display the frame
                    # cv2.drawContours (在哪張圖上, 輪廓本身 ,畫第幾個輪廓 -1表示畫出全部輪廓(塗滿) , 圖的顏色 )
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))  # -1 塗滿 #255白色

                    fingers = count(thresholded, segmented)
                    print(fingers)
                    # cv2.putText(圖像, 文字, 位置, 字體, 字體大小, 顏色, 文字粗細 )
                    cv2.putText(clone, str(fingers), (200, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2)
                    # show the thresholded image
                    # cv2.imshow("Thesholded", thresholded)

            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            # increment the number of frames
            num_frames += 1

            # # observe the keypress by the user
            # keypress = cv2.waitKey(1) & 0xFF
            #
            # # if the user has pressed "q", then stop looping
            # if keypress == ord("q"):
            #     break
        # 因为opencv讀的不是jpg格式，因此要用motion JPEG模式重jpg格式
        grabbed, jpeg = cv2.imencode('.jpg', clone)
        return jpeg.tobytes()


@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

def gen(camera):
    while True:
        #get camera frame
        show_video = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + show_video + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    # http://localhost:3000/
    # load the structure of the model
    print(('* Loading Keras model and Flask starting server...'
           'please wait until server has fully started'))

    # loading the model

    json_file = open('E:/user/Desktop/finaleditv/trainedModel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("E:/user/Desktop/finaleditv/modelWeights.h5")
    print("\n\n\n\nLoaded model from disk\n\n\n\n")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    app.run(host='0.0.0.0', debug=True, port=3000, threaded=False)
