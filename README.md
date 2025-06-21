# A-real-time-cell-image-segmentation-method-based-on-multi-scale-feature-fusion--code

## 🚀 Project Introduction
Glial tumor stem cell bright-field microscopic image segmentation plays a critical role in disease mechanism research and anti-tumor drug development, yet existing methods face challenges including multi-scale heterogeneity, unmarked boundary blurring, and model efficiency-precision imbalance. This study proposes the AKB-Yolo model through multidimensional architectural innovations to achieve high-precision real-time segmentation. The model simultaneously optimizes cell boundary localization, confluence calculation, and population counting. First, we establish a pre-processing system based on Contrast Limited Adaptive Histogram Equalization (CLAHE) and adaptive edge filtering to balance noise suppression with local contrast enhancement requirements in biological imaging. Second, we design the Adaptive Kernel Parameterized Convolution module (AKConv), which captures heterogeneous spatial distribution features of glioma stem cells through dynamic kernel deformation mechanisms, enhancing boundary segmentation accuracy while reducing parameter quantity. Third, we construct a Bidirectional Feature Pyramid Network (BiFPN) employing cross-scale feature field calibration strategies to strengthen multi-size cell recognition capabilities. Finally, we propose probability density-guided non-maximum suppression algorithm to reduce cell underdetection. Experimental results demonstrate that the model achieves 95% mAP on our self-constructed glioma stem cell dataset with 38 fps inference speed, simultaneously supporting dual-modality output for cell confluence analysis and high-precision counting, providing reliable automated tools for tumor microenvironment research.
## 📥 Environment
```python
python==3.8.20 
pytorch>=2.4.1
```

## 🧩 Installation
### Clone repo  
```python
git clone “https://github.com/LCOUD-ALT/A-real-time-cell-image-segmentation-method-based-on-multi-scale-feature-fusion--code.git”  AKB-YOLO
cd AKB-YOLO
```
 
### Install dependencies   
```python
pip install -r requirements.txt
```

## 🧠 Training & Inference
1. ### Train 
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('AKB-YOLO.yaml')

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
2. ### Inference
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('path/to/best.pt')
    results = model.predict("image.jpg",  conf=0.5)  
    results[0].show()  # Display results
```
## 📜 License
Licensed under the MIT License; see LICENSE for details.
