import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')
    model.predict(source='inference/horse/',
              imgsz=640,
              batch=16,
              split='test',
              workers=10,
              device='',
              )

