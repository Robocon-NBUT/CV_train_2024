from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'C:\Users\许奇峰\PycharmProjects\PythonProject\runs\train\exp\weights\best.pt')
    model.predict(source=0,
                  save=True,
                  show=True,
                  )
