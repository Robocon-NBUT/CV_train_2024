from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'E:\desktop\cv-study\train\runs\train\exp13\weights\best.pt')
    model.predict(source=0,
                  save=True,
                  show=True,
                  )