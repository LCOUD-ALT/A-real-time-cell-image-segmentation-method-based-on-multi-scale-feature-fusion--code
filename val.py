import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('path/to/best.pt')
    model.val(data='path/to/your/data.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, 
              project='runs/val',
              name='exp',
              )
