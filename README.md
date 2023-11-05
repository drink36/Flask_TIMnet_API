# Emotion Recognition API 操作手冊

這個專案提供一個基於 Flask 的 API，能夠分析音頻檔案，並識別其中的情感。以下是如何設定和運行此專案的步驟。

## 1. 環境準備

首先，您需要安裝必要的依賴項。在專案的根目錄下運行以下命令：

```bash
pip install -r requirements.txt
```

## 2. 配置設定

所有的配置設定都在 `config.py` 文件中。您可以根據您的需求修改此文件。以下是一些重要的配置選項：

- `MODEL_PATH`: 模型檔案的路徑。
- `AUDIO_SAVE_DIR`: 上傳的音頻檔案保存的目錄。

## 3. 放置模型檔案

將您的模型檔案放在 `Model` 目錄下，並確保 `config.py` 中的 `MODEL_PATH` 設定正確。
Facial 模型請至 https://drive.google.com/drive/folders/1yQtXbi0xCUHRTkix63MEBsFjs3K_00OL?usp=sharing下載後放至 `Model` 目錄下
## 4. 放置音頻檔案

如果您有一些預先準備好的音頻檔案，可以將它們放在 `Wav` 目錄下。

## 5. 運行服務器

在專案的根目錄下，運行以下命令以啟動 Flask 服務器：

```bash
python flask_server.py
```

服務器將在 `http://127.0.0.1:5000/` 上運行。現在，您可以通過發送 POST 請求到 `http://127.0.0.1:5000/predict` 來分析音頻檔案。

## 6. 關於 Postman

[Postman](https://www.postman.com/) 是一個流行的 API 測試工具，它允許開發人員輕鬆地發送各種 HTTP 請求，並查看服務器的響應。它提供了一個直觀的界面，使得測試和調試 API 變得非常簡單。下載並安裝 Postman，以便您可以在本地機器上測試 API。


## 7. 使用 Postman 測試 API

在 Postman 中，創建一個新的 POST 請求，並設置 URL 為 `http://127.0.0.1:5000/predict`。

1. 在「Body」選項卡下，選擇「form-data」。
2. 添加一個名為 `file` 的 key，並上傳一個音頻檔案作為值。/ 影片請添加名為 `file_v` 的 key
3. 點擊「Send」按鈕發送請求。
![教學圖片](https://github.com/t109ab0014/Flask_TIMnet_API/blob/main/teach.png)
您應該會在 Postman 中看到情感分析的結果。

## 8. 檔案結構

下面是專案目錄的結構：

```plaintext
- Model/                  # 存放模型檔案的目錄
- Wav/                    # 存放音頻檔案的目錄
- __pycache__/            # Python 編譯檔案的目錄
- config.py               # 配置設定文件
- flask_server.py         # Flask 服務器檔案
- requirements.txt        # 依賴項列表
- TIM.py                  # 包含模型定義和其他相關函數的檔案
```

這就是全部必要的步驟和信息，以便您能夠設定和運行此專案。
