# 🧠 Age Detection AI Web App

Age Detection AI Web App is a deep learning-based application that detects human faces in images and predicts their age group. Built using OpenCV’s DNN module and deployed with Streamlit, this project provides a simple, fast, and interactive interface for real-time age prediction.

---

## 🚀 Features

- 📤 Upload image and detect faces instantly  
- 🧠 Deep Learning-based age prediction  
- 🎯 Accurate face detection using OpenCV DNN  
- 🌐 Interactive and user-friendly UI with Streamlit  
- ⚡ Fast and lightweight performance  

---

## 🧠 How It Works

- The user uploads an image through the web interface  
- OpenCV DNN model detects faces in the image  
- Each detected face is processed using a pre-trained Caffe model  
- The model predicts the **age group** (e.g., 0–2, 4–6, 8–12, etc.)  
- Bounding boxes and predicted age labels are displayed on the image  

---

## 🛠️ Tech Stack

- 🐍 Python  
- 🎥 OpenCV (DNN Module)  
- 🧮 NumPy  
- 🌐 Streamlit  
- 🤖 Deep Learning (Caffe Models)  

---

## 📂 Project Structure

```
project/
│
├── app.py
├── opencv_face_detector.pbtxt
├── opencv_face_detector_uint8.pb
├── age_deploy.prototxt
├── age_net.caffemodel
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---


---

## 🌐 Applications

- 👤 Facial analysis systems  
- 🛍️ Retail customer insights  
- 📊 Demographic analysis  
- 🤖 AI-based smart applications  
- 🔍 Computer vision learning projects  

---

## 📌 Note

This project uses pre-trained deep learning models for age prediction. While it provides good results, predictions may vary depending on image quality, lighting, and facial features.

---

## 🔮 Future Improvements

- 🎥 Real-time webcam support  
- 🚻 Gender detection integration  
- 📈 Improve model accuracy  
- ☁️ Deployment on Streamlit Cloud  

---

## 🙌 Author

**Shubham Dwivedi**

---

## ⭐ Support

If you like this project, please ⭐ star the repository!
