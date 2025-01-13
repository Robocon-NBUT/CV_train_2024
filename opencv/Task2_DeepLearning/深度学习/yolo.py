import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os

# 图片路径和标签路径
image_path = r'C:\Users\ORIENTCG\PycharmProjects\PythonProject1\yolo\images\bamboo'
label_path = r'C:\Users\ORIENTCG\PycharmProjects\PythonProject1\yolo\labels\bamboo'

# 分类名称
classes = ['红瓶盖', '黄瓶盖', '橙瓶盖', '黄笔盖', '透明笔盖', '黑笔盖', '南孚电池', '绿电池', '紫电池']

# 写入数据集配置文件 (YOLO格式)
yaml_content = f"""
train: {image_path}
val: {image_path}
test: {image_path}

nc: {len(classes)}  # 类别数量
names: {classes}
"""
# 保存数据集配置为 .yaml 文件
yaml_file_path = r'C:\Users\ORIENTCG\PycharmProjects\PythonProject1\yolo\bamboo_dataset.yaml'
with open(yaml_file_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

# 使用 YOLO 训练模型
if __name__ == '__main__':
    # 加载模型
    model = YOLO('yolo11s.pt')

    # 配置 EarlyStopping
    early_stop = EarlyStopping(
        monitor='val/box_loss',
        patience=10,
        mode='min',
        min_delta=0.001,
        verbose=True
    )

    # 配置 Checkpoint 保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val/box_loss',
        dirpath=os.path.join(r'C:\Users\ORIENTCG\PycharmProjects\PythonProject1\yolo', 'runs', 'train', 'num1', 'checkpoints'),
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    # 配置 Trainer
    trainer = pl.Trainer(
        callbacks=[early_stop, checkpoint_callback],
        max_epochs=150,
        accelerator='gpu',
        devices=1,  # 使用 GPU，修改为 -1 使用所有可用 GPU
        logger=True,
        log_every_n_steps=10
    )

    # 训练模型
    model.train(
        data=yaml_file_path,
        cache=False,
        imgsz=640,
        epochs=150,
        single_cls=False,
        batch=32,
        close_mosaic=10,
        workers=4,
        device='0',
        optimizer='AdamW',  # 使用 AdamW 优化器
        amp=True,
        project=r'C:\Users\ORIENTCG\PycharmProjects\PythonProject1\yolo\runs\train',
        name='num1',
    )

    # 输出结果
    if early_stop.stopped_epoch > 0:
        print(f"训练提前结束在第 {early_stop.stopped_epoch} 轮")
    else:
        print("训练正常结束")

    # 保存最佳模型
    print("保存最佳模型至 ./best_model.pt")
    model.save(r'C:\Users\ORIENTCG\PycharmProjects\PythonProject1\yolo\best_model.pt')

