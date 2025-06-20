import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp26/weights/best.pt')
    model.val(data=r'C:\Users\Administrator\Desktop\Snu77\ultralytics-main\New_GC-DET\data.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )