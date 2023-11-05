# -*- coding: UTF-8 -*-
from flask import Flask, request, jsonify
import torch
import numpy as np
import librosa
import os
from tqdm import tqdm
from flask_cors import CORS
import config
# 確保您的模型和函數可以從這裡導入
from TIM import TIMNet, TIM_Net, WeightLayer, Temporal_Aware_Block, SpatialDropout, Chomp1d 
from MAE import prediction
app = Flask(__name__)
CORS(app) 
# 檢查 "/Users/ro9air/VScode/Flask/Server_Recive_file" 目錄是否存在，如果不存在，創建它
if not os.path.exists(config.AUDIO_SAVE_DIR):
    os.makedirs(config.AUDIO_SAVE_DIR)
# check if the directory for saving video exists, if not, create it
if not os.path.exists(config.VIDEO_SAVE_DIR):
    os.makedirs(config.VIDEO_SAVE_DIR)


# 這個函式可以用來讀取單個音頻文件並計算它的MFCC
def get_mfcc(filename, sr=22050, duration=4, framelength=0.05):
    data, sr = librosa.load(filename, sr=sr)
    time = librosa.get_duration(y=data, sr=sr)
    if time > duration:
        data = data[0:int(sr * duration)]
    else:
        padding_len = int(sr * duration - len(data))
        data = np.hstack([data, np.zeros(padding_len)])
    framesize = int(framelength * sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13, n_fft=framesize)
    mfcc = mfcc.T
    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    mfcc_acc = librosa.feature.delta(mfcc_delta, width=3)
    mfcc = np.hstack([mfcc, mfcc_delta, mfcc_acc])
    return mfcc

# 這個函式可以用來讀取多個音頻文件並計算它們的MFCC
def get_mfccs(wav_files: list, sr=22050, duration=4, framelength=0.05):
    print("正在計算MFCC...")
    mfccs = get_mfcc(wav_files[0], sr=sr, duration=duration, framelength=framelength)
    size = mfccs.shape
    for it in tqdm(wav_files[1:]):
        mfcc = get_mfcc(it, sr=sr, duration=duration, framelength=framelength)
        mfccs = np.vstack((mfccs, mfcc))
    mfccs = mfccs.reshape(-1, size[0], size[1])
    return mfccs

# 函數用於從資料夾讀取音訊並返回其MFCC特徵
def load_and_preprocess_from_folder(folder_path):
    wav_files = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)]
    mfcc_features = get_mfccs(wav_files)  # 假設get_mfccs函數會返回一個NumPy數組
    return mfcc_features


def load_model(model_path):
    # 載入模型
    model_path = config.MODEL_PATH
    # 使用這個代碼行來加載整個模型
    model = torch.load(model_path, map_location='cpu') # weight of size [128, 39, 1]
    # 評估模式
    model.eval()
    model.cpu()  
    return model
def predict_emotion(model, audio_file_path):
    emotion_labels = ['anger', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']

    # 使用單一音訊檔案計算 MFCC
    x = get_mfcc(audio_file_path)
    
    # 添加一個新的軸以匹配模型輸入
    x = np.expand_dims(x, axis=0)
    
    # 轉換 x 的形狀（如果需要）
    x = np.transpose(x, (0, 2, 1))

    # 使用模型進行情緒分類
    with torch.no_grad():
        predictions = model(torch.tensor(x, dtype=torch.float32))
        
        # 找到預測張量中的最大值的索引
        _, predicted_label_index = torch.max(predictions, 1)
        
        # 使用索引找到對應的情緒標籤
        predicted_emotion = emotion_labels[predicted_label_index.item()]
        
        return predicted_emotion

# 全局變量來存儲模型
model = load_model(config.MODEL_PATH)
@app.route('/predict', methods=['POST'])
def predict():
    if ('file' or 'file_v') not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    files = request.files.getlist('file')
    files_v = request.files.getlist('file_v')
    emotions = []
    emotions_v = []
    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400
        audio_file_path = os.path.join(config.AUDIO_SAVE_DIR, file.filename) # type: ignore
        file.save(audio_file_path)
        predicted_emotion = predict_emotion(model, audio_file_path)
        #os.remove(audio_file_path)  # 選項：刪除音訊檔案
        emotions.append({'filename': file.filename, 'emotion': predicted_emotion})
    for file_v in files_v:
        if file_v.filename == '':
            return jsonify({'error': 'No file_v selected for uploading'}), 400
        vudio_file_path = os.path.join(config.VIDEO_SAVE_DIR, file_v.filename)
        file_v.save(vudio_file_path)
        predicted_emotion_v = prediction(vudio_file_path)
        emotions.append({'filename': file_v.filename, 'emotion': predicted_emotion_v})


    return jsonify({'results': emotions})
    


if __name__ == '__main__':
    app.run(debug=True)