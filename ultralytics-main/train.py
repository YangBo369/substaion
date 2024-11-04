from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.yaml') .load("weights/yolo11n.pt") # 此处以 m 为例，只需写yolov11m即可定位到m模型
    model.train(data='substaion.yaml',
                imgsz=640,
                epochs=300,
                single_cls=True,
                batch=8,
                workers=8,
                device='1',
                )
