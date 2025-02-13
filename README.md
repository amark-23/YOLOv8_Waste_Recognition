# YOLOv8 Waste Recognition

## 📌 Project Overview
This project implements **YOLOv8** for **waste recognition** using object detection. The model is trained to classify and detect different types of waste items such as **plastic, glass, organic, metal, and paper**. The system includes dataset preparation, training, evaluation, and real-time webcam inference.

You can access an example dataset [in this link](https://drive.google.com/file/d/1SF7nb_AjUP72SkaFRd2ZiT_rU4ScqZk2/view?usp=sharing).
The runs/ folder contains the training results of that dataset.

---

## 📂 Directory Structure
```
yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── scripts/
│   ├── train_custom_yolo.py
│   ├── evaluate_model.py
│   ├── webcam_detection.py
│   └── custom_data.yaml
├── runs/ (YOLO training results)
├── yolo_env/ (Virtual environment)
└── README.md
```

---

## 🔧 Environment Setup
### 1️⃣ Install Python & Git
- **Windows**: Install from [python.org](https://www.python.org/)
- **Mac**: Run `brew install python git`
- **Ubuntu**: Run `sudo apt install python3 python3-pip git`

### 2️⃣ Set Up Virtual Environment
```sh
python -m venv yolo_env
```
- **Windows**: `yolo_env\Scripts\activate`
- **Mac/Linux**: `source yolo_env/bin/activate`

### 3️⃣ Install Dependencies
```sh
pip install ultralytics opencv-python numpy torch torchvision
```

### 4️⃣ Verify Installation
```sh
python -c "import ultralytics, cv2, torch; print(ultralytics.__version__, cv2.__version__, torch.__version__)"
```

---

## 📊 Dataset Preparation
### 1️⃣ Collect Images
- Use **at least 100 images per class** (total: 500+ images)
- Ensure **diverse backgrounds, lighting, and angles**

### 2️⃣ Label Images
Install **LabelImg**:
```sh
pip install labelImg
labelImg
```
Save annotations in **YOLO format (.txt)**.

### 3️⃣ Split Data (Train/Val/Test)
Run the following script to split data:
```sh
python split_dataset.py
```

### 4️⃣ Create `custom_data.yaml`
```yaml
path: /path/to/yolo_dataset
train: images/train
val: images/val
test: images/test
nc: 5  # Number of classes
names: ["plastic", "glass", "organic", "metal", "paper"]
```

---

## 🎯 Model Training
### 1️⃣ Download Pretrained YOLOv8 Weights
```sh
yolo download yolov8n.pt
```

### 2️⃣ Run Training Script
```sh
python train_custom_yolo.py
```
- **Custom training parameters:**
  - `epochs=100`
  - `batch=16`
  - `optimizer='Adam'`
  - `learning_rate=0.001`
  - `augment=True`

### 3️⃣ Monitor Training with TensorBoard
```sh
pip install tensorboard
tensorboard --logdir runs/detect/custom_model
```

---

## 📈 Model Evaluation
Run the evaluation script:
```sh
python evaluate_model.py
```
This prints:
- **mAP50**: Mean Average Precision at IoU 0.5
- **Precision**: Accuracy of detections
- **Recall**: Correctly detected objects

---

## 🎥 Real-Time Webcam Detection
Run the real-time object detection script:
```sh
python webcam_detection.py
```
Press **'q'** to exit.

**Troubleshooting:**
- If the **webcam doesn’t open**, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.
- Reduce `imgsz` if performance is slow on CPU.

---

