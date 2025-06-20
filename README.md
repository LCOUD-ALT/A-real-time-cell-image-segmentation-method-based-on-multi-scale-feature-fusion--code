# A-real-time-cell-image-segmentation-method-based-on-multi-scale-feature-fusion--code

## Project Introduction

## Environment
python==3.8.20 <br>
pytorch>=2.4.1

## Installation
### Clone repo  
```python
git clone https://github.com/LCOUD-ALT/A-real-time-cell-image-segmentation-method-based-on-multi-scale-feature-fusion--code.git 
cd A-real-time-cell-image-segmentation-method-based-on-multi-scale-feature-fusion--code
```
 
### Install dependencies   
```python
pip install -r requirements.txt
```

## 🧠 Training & Inference
### Train with custom dataset
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8-BiFPN-AKconv')

    model.train(data='path/to/your/data.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                single_cls=False,  
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                amp=True,  
                project='runs/train',
                name='exp',
                )
```
### Inference
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('path/to/best.pt')
    results = model.predict("image.jpg",  conf=0.5)  
    results[0].show()  # Display results
```
