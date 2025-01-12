# <center> opencv_basic_operates心得
<font face ="楷体" size=6>1. 多层感知机实现 XOR 运算
多层感知机（MLP）任务的目的是解决经典的 XOR 问题。这个任务的挑战在于 XOR 逻辑操作的非线性特点，需要通过适当的隐藏层和激活函数来实现。通过这个任务，我理解了神经网络如何通过训练来逼近非线性函数，尤其是在数据维度较低时，模型的表现和训练速度较为直观。在实现时，我使用了两个布尔输入和一个输出，通过适当的网络结构和 ReLU 激活函数来训练网络。

2. 卷积神经网络（CNN）识别手写汉字
卷积神经网络任务是这个项目中最具挑战性的部分。为了实现对自己书写的汉字的识别，我需要首先建立一个包含多个卷积层和池化层的 CNN 模型。这个模型能够提取图像中的特征，并通过全连接层对汉字进行分类。训练数据集的准备是关键，尤其是在手写汉字的样本获取和标签处理上。通过数据增强技术，如旋转、平移和缩放等，能够增加训练数据的多样性，避免过拟合。

在训练过程中，我也遇到了一些困难，例如由于训练样本不足导致模型的准确性不高，但通过适当的优化器（如 Adam）和调整学习率，模型逐渐收敛并能够实现较高的识别精度。此外，这个任务让我深刻理解了卷积神经网络的工作原理，尤其是在处理图像数据时，卷积层在特征提取方面的优势是无可比拟的。

3. YOLOv11进行垃圾分类
YOLOv11 任务是项目中的重点和难点。在使用 YOLOv5/YOLOv11 模型进行垃圾分类时，我首先需要准备一个包含多个类别（如可回收垃圾、有害垃圾、厨余垃圾等）的数据集。通过数据标注和图片增强，确保数据集的多样性和准确性是至关重要的。训练过程中的非最大抑制（NMS）算法帮助去除冗余检测框，提高了检测精度。

YOLO模型的优势在于其优秀的实时检测能力，能够快速且准确地对图像中的多个目标进行分类和定位。在训练过程中，我经历了多次调整超参数（如学习率、批大小等）和模型结构，以提高模型的精度和推理速度。最终，通过充分的训练和优化，我成功实现了垃圾分类任务，能够识别多个类别的垃圾。

总结
这个项目不仅加深了我对深度学习的理解，还让我亲自经历了从数据准备、模型设计到训练调优的全过程。通过这三个任务，我深刻体会到了模型选择、数据质量和超参数调整对最终结果的重要性。每一个任务的成功实现都让我更加自信，并且激发了我深入研究深度学习领域的兴趣。今后，我希望能够在这些任务的基础上，探索更多的应用场景，如更复杂的物体检测、图像分类和自然语言处理等。


<img width="1280" alt="西" src="https://github.com/user-attachments/assets/9b535bf8-b84b-4881-bfdd-df1c84cb1aff" />
<img width="1280" alt="东" src="https://github.com/user-attachments/assets/1eb734ea-18b8-46d2-b592-9ef79cd00a5e" />
<img width="1271" alt="刀" src="https://github.com/user-attachments/assets/28ff7cd6-c45f-4e70-be94-6792f068e1a8" />
<img width="1280" alt="内" src="https://github.com/user-attachments/assets/5c13b955-fadf-4341-ba0f-16215eaeac1a" />
<img width="1280" alt="上" src="https://github.com/user-attachments/assets/04358005-8397-4267-b2bd-2951bcc05849" />
<img width="1280" alt="十" src="https://github.com/user-attachments/assets/21f4b114-ab58-4036-975f-4eeed63ffa49" />
<img width="1280" alt="九" src="https://github.com/user-attachments/assets/8f992cd2-1a5d-4216-ac51-e78526421a40" />
<img width="1280" alt="八" src="https://github.com/user-attachments/assets/0e78c019-9095-4353-a8be-df7cb199e6bb" />
<img width="1280" alt="七" src="https://github.com/user-attachments/assets/0eb369f9-54c8-4fde-b2a0-68b01599582d" />
<img width="1280" alt="六" src="https://github.com/user-attachments/assets/e07a1c33-badb-42fc-bac5-576c69c18c5f" />
<img width="1280" alt="五" src="https://github.com/user-attachments/assets/94750f83-49eb-4e3a-8fc1-6f9843e8b0d7" />
<img width="1280" alt="四" src="https://github.com/user-attachments/assets/ed6321a1-91b3-4da8-90bb-437cd8eb6a51" />
<img width="1280" alt="三" src="https://github.com/user-attachments/assets/3ea76144-aa81-43cb-89eb-13b0c11dd654" />
<img width="1280" alt="二" src="https://github.com/user-attachments/assets/9fa60461-dc1c-48d4-b293-e7719ba7bab8" />
<img width="1280" alt="一" src="https://github.com/user-attachments/assets/08623cd5-d868-4cc3-8b22-ff32a290cb8d" />
<img width="1280" alt="丁" src="https://github.com/user-attachments/assets/fec5497d-0d89-4816-9aee-43b18f76ac0f" />
<img width="1280" alt="万" src="https://github.com/user-attachments/assets/93a10b82-5f22-46e9-b79d-bef26a99128f" />
<img width="1277" alt="中" src="https://github.com/user-attachments/assets/74bbc9ca-bcf0-45e7-9d70-270e17ef3c02" />
<img width="1280" alt="北" src="https://github.com/user-attachments/assets/524355bf-adc5-4575-99ea-14d00a4926d6" />
<img width="1280" alt="南" src="https://github.com/user-attachments/assets/6e5386c2-5e95-4b5f-9c4e-181ca8ea7a2b" />




https://github.com/user-attachments/assets/247e5a16-5e2d-4c64-b693-7bb285834025


