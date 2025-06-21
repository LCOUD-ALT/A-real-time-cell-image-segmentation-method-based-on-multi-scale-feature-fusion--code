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
