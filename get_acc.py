import os
import torch
from ConvNext import convnext_tiny
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torchvision import transforms, models
import time
from sklearn.metrics import confusion_matrix
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import shufflenet
from Attention import Eca_Block

label = np.array(['A', 'B', 'C', 'D'])

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    # transforms.Grayscale(),  # 灰度图
    transforms.ToTensor()
])
device = torch.device("cuda")
model = models.densenet121(progress=True)
for param in model.parameters():
    param.requires_grad = True
model.features[0] = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
    Eca_Block(64)
)
# print(model)
# 构建新的全连接层
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(1024, 512),
    # torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(512, 4)
)
model.load_state_dict(torch.load('mode2/densnet101_att.pth'))
print(model)
model.eval().cuda()
predictions_labels = []


def predict(image_path):
    # 打开图片
    img = Image.open(image_path)
    # 数据处理，增加一个维度
    img = transform(img).unsqueeze(0)
    # 预测得到结果
    inputs = Variable(img.cuda())
    outputs = model(inputs)
    # 获得最大值所在位置
    _, predicted = torch.max(outputs, 1)
    # print(predicted.item())
    # print(outputs[0].data.cpu())
    if predicted.item() > len(label):
        return 0
        # 转化为类别名称
    else:
        predictions_labels.append(label[predicted.item()])
        # print("pred:", label[predicted.item()])
    # if (label[predicted.item()]) == name:
    #     right.append(1)


time1 = time.time()
X = []  # 定义图像名称
Y = []  # 定义图像分类类标
for i in range(65, 69):
    # 遍历文件夹，读取图片
    for f in os.listdir("ct2_dataset/x-new/" + chr(i)):
        # 获取图像名称
        X.append("ct2_dataset/x-new/" + chr(i) + "/" + str(f))
        predict("ct2_dataset/x-new/" + chr(i) + "/" + str(f))
        # 获取图像类标即为文件夹名称
        Y.append(chr(i))
print("耗时：", time.time() - time1)
print(X)
print(Y)

y_true = Y  # 正确标签
y_pred = predictions_labels  # 预测标签
m = sm.confusion_matrix(y_true, y_pred)
print('混淆矩阵为：', m, sep='\n')
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
r = sm.classification_report(y_true, y_pred)
print('分类报告为：', r, sep='\n')
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(label))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
tick_marks = np.array(range(len(label))) + 0.5


def plot_confusion_matrix(cm, title='last', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(label)))
    plt.xticks(xlocations, label, rotation=90)
    plt.yticks(xlocations, label)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='DenseNet-Att')
# show confusion matrix
plt.savefig('0824/densenet-att.png', format='png')
plt.show()
