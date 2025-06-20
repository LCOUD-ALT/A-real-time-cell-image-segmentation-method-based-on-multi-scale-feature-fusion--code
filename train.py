import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8-BiFPN-AKconv.yaml')

    # model.load('yolov8n.pt') # 是否加载预训练权重

    model.train(data=r'C:\Users\Administrator\Desktop\Snu77\ultralytics-main\New_GC-DET\data.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=150,
                single_cls=False,  # 是否是单类别检测
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                )