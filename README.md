# Crowd Density Regression

This project implements a **convolutional neural network (CNN)** in TensorFlow/Keras that predicts **crowd counts** directly from input images of the **ShanghaiTech Part B dataset**.  
Unlike density-map methods, this is a **regression baseline** that outputs the total number of people in each image.

---

## 🔍 Features
✅ CNN regression model for crowd counting  
✅ End-to-end training & evaluation scripts  
✅ Dataset download via Kaggle API  
✅ Easy inference on new images  
✅ Lightweight & reproducible structure  

---

## 📸 Architecture Overview
ShanghaiTech Part B Dataset → Preprocessing → CNN Regression → Crowd Count Prediction

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/crowd-density-regressor.git
cd crowd-density-regressor
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Download dataset
```python
python scripts/download_shanghaitech_partb.py --dest data
```
### 4. Train model
```python
python src/train.py
```
### 5. Evaluate
```python
python src/evaluate.py
```
### 6. Predict on a single image
```python
python src/predict.py --image examples/sample.jpg
```
### 7. Configuration

All paths & parameters are in src/config.py. Example::
```python
IMAGE_SIZE = (995, 421)
EPOCHS = 100
BATCH_SIZE = 4
DATASET_PATH = "data/ShanghaiTech/part_B"

```
Adjust these based on your screen and crowd threshold preference.

📦 Dataset
This project uses the ShanghaiTech Part B dataset, available on Kaggle:  
https://www.kaggle.com/datasets/tthien/shanghaitech  

Due to licensing, the dataset is not included in this repository.  
To run training, download Part_B manually and place it in:
data/ShanghaiTech/part_B/train_data/
data/ShanghaiTech/part_B/test_data/

### ✍️ Authors

- **Mohammed Afnan**  
  6th Sem, ETE, RVCE – [GitHub](https://github.com/MohammedAfnanet22RVCe) [LinkedIn](https://www.linkedin.com/in/mohammed-afnan-17b30122a/)

### 👥 Collaborators

- **Karunesh Kumar**  
  6th Sem, IEM, RVCE – [GitHub](https://github.com/Karunesh3770)
