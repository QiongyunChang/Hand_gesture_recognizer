# Hand_gesture_recognizer
This is the college graduation monograph.  


* 圖片生成: captureHand.py -> 自行拍照(resize、二值化圖像)

* 圖像的前處理: readimg.py -> label圖像 (產生afterencode.zip)

* 模型訓練(CNN): trainer.py -> 產生model與權重 (trainedmodel.json 、modelWeights.h5 )

* 實際辨識: recognizer_final.py -> Real-time實際辨識的檔案
 
* 直接餵入圖片做測試: testPho.py  -> 測試

----------------
* 與網站結合: web_flask.py、index.html、main.css


# Recognition
![image](https://github.com/Todoorno/Hand_gesture_recognizer/blob/master/tempsnip.png)

# Outcome 
![image](https://github.com/Todoorno/Hand_gesture_recognizer/blob/master/outcome.png)





