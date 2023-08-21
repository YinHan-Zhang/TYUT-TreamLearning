### 1.完整代码

```python
import glob
import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import nibabel as nib


torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# 详解见CV方向baseline精读
train_path = glob.glob('../脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('../脑PET图像分析和疾病预测挑战赛公开数据/Test/*')

# 详解见CV方向baseline精读
np.random.shuffle(train_path)
np.random.shuffle(test_path)

DATA_CACHE = {}


class XunFeiDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = nib.load(self.img_path[index])
            img = img.dataobj[:, :, :, 0]
            DATA_CACHE[self.img_path[index]] = img

        # 随机选择一些通道
        idx = np.random.choice(range(img.shape[-1]), 50)
        img = img[:, :, idx]
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = img.transpose([2, 0, 1])
        return img, torch.from_numpy(np.array(int('NC' in self.img_path[index])))

    def __len__(self):
        return len(self.img_path)


train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[:-10],
                  A.Compose([
                      A.RandomRotate90(),
                      A.RandomCrop(120, 120),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                      A.RandomBrightnessContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=True, num_workers=False, pin_memory=False
)

val_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[-10:],
                  A.Compose([
                      A.RandomCrop(120, 120),
                  ])
                  ), batch_size=2, shuffle=False, num_workers=False, pin_memory=False
)

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_path,
                  A.Compose([
                      A.RandomCrop(128, 128),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=False, num_workers=False, pin_memory=False
)


class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()
        model = models.resnet18(True)
        model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 2)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


model = XunFeiNet()
# model = model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), 0.001)


def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for i, (input, target) in enumerate(train_loader):
        # input = input.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        output = model(input)
        target = torch.tensor(target, dtype=torch.float32)
        target = target.to(dtype=torch.long)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(loss.item())

        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate(val_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # input = input.cuda()
            # target = target.cuda()

            # compute output
            output = model(input)
            target = torch.tensor(target, dtype=torch.float32)
            target = target.to(dtype=torch.long)
            loss = criterion(output, target)

            val_acc += (output.argmax(1) == target).sum().item()

    return val_acc / len(val_loader.dataset)


for _ in range(3):
    train_loss = train(train_loader, model, criterion, optimizer)
    val_acc = validate(val_loader, model, criterion)
    train_acc = validate(train_loader, model, criterion)

    print(train_loss, train_acc, val_acc)


def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    test_pred = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            # input = input.cuda()
            # target = target.cuda()

            output = model(input)
            test_pred.append(output.data.cpu().numpy())

    return np.vstack(test_pred)


pred = None
for _ in range(10):
    if pred is None:
        pred = predict(test_loader, model, criterion)
    else:
        pred += predict(test_loader, model, criterion)

submit = pd.DataFrame(
    {
        'uuid': [int(x.split('\\')[-1][:-4]) for x in test_path],
        'label': pred.argmax(1)
    })
submit['label'] = submit['label'].map({1: 'NC', 0: 'MCI'})
submit = submit.sort_values(by='uuid')
submit.to_csv('submit5.csv', index=None)
```

### 2.代码精读与知识补充

**（1）pytorch**

pytorch是一个开源的Python深度学习框架，提供了许多灵活高效的方法来构建和训练深度神经网络

pytorch的核心特点包括：

1.动态计算图：pytorch使用动态计算图，使得在模型训练过程中可以实时调整和修改计算图。这种动态计算图的设计使得模型开发和调试更加灵活高效

2.张量操作：pytorch提供了丰富的张量操作和数学运算函数，可以方便地进行矩阵计算、张量操作和数学运算，支持GPU加速

3.自动求导：pytorch能够自动计算变量的梯度，无需手动编写反向传播算法。这种自动求导功能使得梯度计算和参数更新更加简单和高效

4.模块化设计：pytorch的模块化设计使得构建复杂的神经网络更加容易。模型的各个组件、层和损失函数都可以根据需要进行组合和定制。

5.高效的生态系统：pytorch具有庞大而活跃的社区，提供了丰富的扩展库和工具，如TorchVision（计算机视觉）、TorchText（自然语言处理）和TorchAudio（音频处理），进一步扩展了pytorch的功能

**（2）**

```python
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
```

`torch.manual_seed(0)` 将 PyTorch 的随机数种子设置为 0。通过设置这个种子，可以使得每次运行代码时生成的随机数序列是可再现的，即相同的种子会得到相同的随机数序列。这对于调试和复现实验结果是很重要的

`torch.backends.cudnn.deterministic = False` 将 CUDA 的随机数生成行为设置为非确定性。默认情况下，CUDA 使用非确定性算法来加速卷积和其他操作。将该选项设置为 False 可以提高性能

`torch.backends.cudnn.benchmark = True` 启用基于输入大小的自动寻找最快的卷积实现。当输入大小固定时，这一选项可以自动寻找最优的卷积算法以提高性能。然而，如果输入大小会变化，这可能会导致性能波动

**补充：**

CUDA：

CUDA（Compute Unified Device Architecture）是NVIDIA开发的一种并行计算平台和编程模型，用于利用GPU（图形处理器）进行通用目的的并行计算。CUDA使开发者能够利用GPU的大规模并行性，将任务分解为多个并发运行的线程块（thread block）。每个线程块中可以有多个线程，同时多个线程块可以并发地在GPU上执行。通过在GPU上同时执行大量线程，CUDA可以加速诸如图像处理、机器学习、科学计算等需要大量计算的任务。

并行计算方法详情请看[thorough-pytorch/source/第二章/2.3 并行计算简介.md at main · datawhalechina/thorough-pytorch (github.com)](https://github.com/datawhalechina/thorough-pytorch/blob/main/source/第二章/2.3 并行计算简介.md)

非确定性算法：

非确定性算法是指在每次运行时产生不同结果的算法。在深度学习中，通常使用的卷积操作是卷积神经网络（CNN）中最常用的操作之一。非确定性算法通常采用的是一些启发式的方法或近似计算，通过牺牲一定的精度来换取更快的计算速度。非确定性卷积算法采用一些优化策略，例如基于快速傅里叶变换（FFT）的算法或Winograd算法。这些算法通过将卷积计算转化为频域或其他形式的计算，减少运算量和复杂度，从而提高计算速度。

**（3）**

```python
DATA_CACHE = {}

if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            # 详解见CV方向baseline精读
            img = nib.load(self.img_path[index])
            img = img.dataobj[:, :, :, 0]
            DATA_CACHE[self.img_path[index]] = img
```

`DATA_CACHE = {}` 是一个空的字典对象，用于缓存从磁盘加载的图像数据。在代码中，每次从磁盘加载图像时，首先检查该图像是否已经存在于缓存中。如果存在，则直接从缓存中获取图像数据，而不需要再次从磁盘加载。这样可以减少重复的磁盘读取操作，提高数据加载的效率。

`DATA_CACHE` 字典的键是图像路径，值是相应图像的数据。

**（4）通道**

```python
# 随机选择一些通道
	idx = np.random.choice(range(img.shape[-1]), 50)
    img = img[:, :, idx]
```

`shape[-1]`取出最后一个维度的大小，这里指图像数据的通道数

`img[:, :, idx]` 则表示选择 `img` 中的所有高度和宽度，但只选择 `idx` 中指定的通道索引对应的通道数据

**补充：**

图片深度和通道的区别 ：

深度（depth）：在图像处理领域，深度通常指的是图像的三维空间中的切片数量或层数。在医学影像中，深度可以表示脑部扫描中的切片数量，每个切片代表一个不同的层级或深度

通道（channel）：通道则通常指的是图像中的颜色通道或特征通道。对于彩色图像，常见的通道是红色、绿色和蓝色（RGB通道）。每个通道代表一种颜色信息。对于黑白图像，也可以将其视为单通道图像。此外，在某些情况下，通道可以表示不同的特征或图像处理操作的结果

**（5）**

```python
img = img.transpose([2, 0, 1])
```

`img = img.transpose([2, 0, 1])` 用于对图像数据的维度进行转置操作。这里的 img 是一个 numpy 数组，其形状为 `(depth, height, width)`

`2` 表示原始图像数据的第三个维度（切片维度）将变为新图像数据的第一个维度

`0` 表示原始图像数据的第一个维度（高度维度）将变为新图像数据的第二个维度

`1` 表示原始图像数据的第二个维度（宽度维度）将变为新图像数据的第三个维度

这种维度转置的目的是为了适应后续模型的输入要求：`(channels, height, width)`

**（6）DataLoader**

```python
train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[:-10],
                  A.Compose([
                      A.RandomRotate90(),
                      A.RandomCrop(120, 120),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                      A.RandomBrightnessContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=True, num_workers=False, pin_memory=False
)
```

`torch.utils.data.Dataset`是代表这一数据的抽象类（也就是基类）。我们可以通过继承和重写这个抽象类实现自己的数据类，只需要定义`__len__`和`__getitem__`这个两个函数
DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，并在数据集上提供单线程或多线程(num_workers )的可迭代对象

主要参数有：

`dataset`: 一个自定义的数据集对象，可以是 `Dataset` 的子类

`batch_size`: 每个批次的数据样本数量

`shuffle`: 是否在每个 epoch 迭代或数据迭代中对数据进行随机重排

`num_workers`: 加载数据样本时使用的并行工作进程数

`pin_memory`: 是否将数据加载到显存的固定内存中，用于加速数据传输

如果是线程报错，有可能是`num_workers`出了问题

**（7）albumentations库**

```python
train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[:-10],
                  A.Compose([
                      A.RandomRotate90(),
                      A.RandomCrop(120, 120),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                      A.RandomBrightnessContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=True, num_workers=False, pin_memory=False
)
```

从 `train_path` 中选择了倒数第11个元素之前的所有元素，将剩余元素留作验证集

`A.RandomRotate90()`: 随机旋转图像90度。这个操作随机选择90度、180度或270度的旋转角度来对图像进行旋转

`A.RandomCrop(120, 120)`: 随机裁剪图像。这个操作将图像随机裁剪为给定的大小 (120, 120)

`A.HorizontalFlip(p=0.5)`: 水平翻转图像的操作。这个操作有50%的概率水平翻转输入图像

`A.RandomContrast(p=0.5)`: 随机调整图像对比度的操作。这个操作有50%的概率随机调整图像的对比度

`A.RandomBrightnessContrast(p=0.5)`: 随机调整图像亮度和对比度的操作。这个操作有50%的概率随机调整图像的亮度和对比度

**albumentations库与torchvision.transforms的比较：**

1.`albumentations` 库提供了更多种类和更复杂的图像增强操作，涵盖了旋转、裁剪、翻转、缩放、亮度调整、对比度调整、色彩变换、噪声添加、几何变换等多个方面。而 `torchvision.transforms` 属于 torchvision 库的一部分，提供了一些简单的变换操作，如图像翻转、裁剪和缩放等，相对于 `albumentations` 来说功能较为有限

2.`albumentations` 库与主流的深度学习框架（如 PyTorch、TensorFlow 等）兼容性很好，可以无缝地集成到这些框架中。而 `torchvision.transforms` 是PyTorch的专属库，原生支持 PyTorch 的数据加载和处理流程

3.`albumentations` 库采用了高度优化的图像处理算法和并行处理方式，以提高处理效率。相比之下，`torchvision.transforms` 的处理速度较慢，尤其在大规模数据集上会有较大的性能差异

**albumentations库常见的数据增强方法：**

1.Geometric Transformations（几何变换）:

随机旋转图像：`RandomRotate90()`, `Rotate()`, `RandomRotate()`

随机缩放和平移：`RandomScale()`, `RandomTranslate()`

随机翻转图像：`HorizontalFlip()`, `VerticalFlip()`

2.Image Enhancement（图像增强）:

随机调整对比度和亮度：`RandomContrast()`, `RandomBrightness()`

随机调整颜色通道：`HueSaturationValue()`

随机调整图像锐度：`Sharpen()`

3.Image Filtering（图像滤波）:

随机加噪声：`GaussNoise()`, `RandomBrightnessContrast()`

随机模糊：`Blur()`

随机交换颜色通道：`ChannelShuffle()`

4.Image Restoration（图像修复）:

随机模糊或改变清晰度：`Blur(), Sharpen()`

随机加噪声或去噪：`GaussNoise(), Denoise()`

5.Semantic and Instance Segmentation（语义和实例分割）:

随机剪裁或调整图像尺寸：`RandomCrop()`, `Resize()`

随机翻转和旋转图像：`HorizontalFlip()`, `VerticalFlip()`, `Rotate()`

**（8）训练集、验证集、测试集**

训练集：

训练集是用于模型训练的数据集。在训练过程中，模型会基于训练集的样本进行学习和参数调整，以使模型能够最大程度地拟合训练数据的特征和目标

验证集：

验证集是用于模型调优和选择合适模型超参数的数据集。在训练过程中，使用验证集评估模型在未见过的数据上的性能表现，如准确率、损失等。通过选择不同的超参数或模型，可以在验证集上进行评估，并选择性能最佳的模型或超参数

测试集：

测试集是用于模型最终评估和性能测试的数据集。在完成模型训练和调优后，使用测试集来评估模型的泛化能力，即模型对未知数据的预测性能。测试集通常是独立于训练集和验证集的，确保了对模型性能的客观评估

划分训练集、验证集和测试集的比例可以根据具体问题和数据量来确定。常用的划分方式有很多，如80%用于训练，10%用于验证，10%用于测试。但实际使用时，根据数据集大小和特定需求，可以调整划分比例，如使用交叉验证等更复杂的划分方法

**（9）神经网络**

```python
class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()
        model = models.resnet18(True)
        model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 2)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out
```

可以看视频先了解一下神经网络是什么[【神经网络】大白话直观理解！_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1YY411a7F7/?spm_id_from=333.788&vd_source=8c365458ad3da57c111e163a058ae355)

**神经网络**接收输入数据，并通过多个**神经元**的计算和组合得出输出结果。每个神经元接收上一层神经元的输出，通过**激活函数**计算出加权和，并将该结果传递给下一层的神经元。这一过程逐层进行，直到到达输出层，最终产生**预测结果或分类输出**

**（10）torch.nn**

`torch.nn` 是 PyTorch 中用于构建神经网络的模块化库

1.`nn.Module`：`nn.Module` 是所有神经网络模型的基类。通过继承 `nn.Module`，我们可以构建自己的神经网络模型，并定义一些必要的方法和属性

2.Layers（层）：`nn` 模块库提供了各种预定义的层类型，如全连接层（`nn.Linear`）、卷积层（`nn.Conv2d`）、循环神经网络层（`nn.RNN`）等。这些层可以按需组合起来构建深度神经网络结构

3.Activation Functions（激活函数）：`nn` 提供了一系列常用的激活函数，如`ReLU、sigmoid、tanh` 等。这些激活函数可用于增加网络的非线性特性

4.Loss Functions（损失函数）：`nn` 包含各种损失函数，用于在训练网络时计算预测结果与真实标签之间的误差。常用的损失函数包括交叉熵损失（`nn.CrossEntropyLoss`）、均方误差损失（`nn.MSELoss`）等

5.Optimizers（优化器）：`nn` 还提供了各种优化器，如随机梯度下降（SGD）、Adam 等，用于更新网络中的参数以降低损失函数的值

**（11）残差网络ResNet**

```python
model = models.resnet18(True)
        model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 2)
        self.resnet = model
```

网络越深训练效果越好，但是计算量与训练难度也越大，为解决这个问题，提出了通过在网络中引入残差块，使得信息能够直接跳过一部分层的连接的想法

在baseline中运用的是ResNet的一个变体ResNet-18，它是一个相对较小的深度网络结构，具有18个卷积层。ResNet-18在许多计算机视觉任务中表现出色，并成为许多深度学习应用的基础模型之一

`model = models.resnet18(True)`：这一行创建了一个预训练的 ResNet-18 模型，并将其赋值给变量 `model`。该模型是从 `torchvision.models` 中的 `resnet18` 函数返回的

`model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)`：这一行修改了 ResNet-18 模型的第一个卷积层 (`conv1`)。通过指定参数，将输入通道数从默认的 3 修改为 50，输出通道数为 64，卷积核大小为 7x7，步长为 2x2，填充为 3x3，且禁用偏置项 (`bias=False`)

`model.avgpool = nn.AdaptiveAvgPool2d(1)`：这一行修改了 ResNet-18 模型的平均池化层 (`avgpool`)。使用 `nn.AdaptiveAvgPool2d` 函数将平均池化层改为自适应平均池化，输出大小为 1x1

`model.fc = nn.Linear(512, 2)`：这一行修改了 ResNet-18 模型的全连接层 (`fc`)。通过 `nn.Linear` 函数将全连接层的输入特征数改为 512，输出特征数改为 2，用于适应特定的分类任务。

`self.resnet = model`：最后，将修改后的 ResNet-18 模型赋值给 `self.resnet` 变量。这样，变量 `self.resnet` 将包含已经进行了自定义修改的 ResNet-18 模型

**（12）forward**

```python
def forward(self, img):
        out = self.resnet(img)
        return out
```

在神经网络中，`forward` 方法是一个特殊的函数，用于定义模型的前向传播过程。它指定了输入如何通过网络的不同层和操作，最终产生输出

在 PyTorch 中，定义一个自定义模型时，必须实现 `forward` 方法作为模型类的一部分。因为在训练和推断的过程中，PyTorch 会调用模型的 `forward` 方法来进行数据传递和计算

**（13）损失函数**

```python
criterion = nn.CrossEntropyLoss()
```

损失函数是用来衡量模型预测结果与真实标签之间的差异程度的函数

损失函数的作用是帮助优化算法评估模型的性能，并指导模型参数的更新，以使模型能够更好地拟合训练数据和泛化到未见过的数据

一些常见的损失函数包括：

1.均方误差（Mean Squared Error, MSE）：用于回归任务，在预测连续值时常用

2.交叉熵损失（Cross-Entropy Loss）：用于分类问题中，特别是多类别分类情况

3.二元交叉熵损失（Binary Cross-Entropy Loss）：用于二分类问题

4.KL 散度（Kullback-Leibler Divergence）：用于度量概率分布的差异

5.Hinge 损失（Hinge Loss）：用于支持向量机（SVM）等模型的训练

**（14）优化器**

```python
# 0.001是学习率(lr)
optimizer = torch.optim.AdamW(model.parameters(), 0.001)
```

优化器是用来更新模型参数以最小化损失函数的算法

优化器的作用是根据损失函数的梯度信息来调整模型参数，以使模型能够逐步逼近最优解

优化器的主要作用如下：

1.参数更新：优化器根据损失函数的梯度信息，计算模型参数的更新量，并更新模型的参数。通过迭代地反复更新参数，可以使模型在训练数据中逐渐优化

2.学习率调整：优化器通常包含学习率参数，用来控制参数更新的步长。学习率决定了每次参数更新的幅度，过大的学习率可能导致无法收敛，而过小的学习率则可能导致收敛速度缓慢。优化器可以自动地在训练过程中适应性地调整学习率

3.参数加速：某些优化器（如动量优化器、Adam 等）引入动量来加速参数更新。动量可以使参数更新在梯度方向上具有惯性，从而在减小损失的同时加速收敛

4.正则化：一些优化器支持正则化方法，如 L1 正则化和 L2 正则化。正则化通过在损失函数中添加正则项，对模型参数进行约束，可以帮助防止过拟合（详情可看CV方向baseline精读）

常见的优化器算法包括梯度下降、随机梯度下降、动量优化器、Adam、Adagrad 等。每个优化器算法都有其独特的特点和适用场景。选择合适的优化器算法可以加快模型训练的速度、提高性能并改善收敛效果

**AdamW**

AdamW的变体在Adam算法的基础上加入了权重衰减的机制

AdamW的优点包括：

1.减小了权重衰减对自适应学习率机制的干扰：传统的Adam算法将权重衰减项应用于参数更新中，这可能导致自适应学习率机制向正则化策略偏移。而AdamW通过将权重衰减分开计算，保持了自适应学习率机制的独立性

2.改善了模型的泛化性能：通过在优化过程中对权重进行正则化，AdamW能够抑制过拟合，提高模型的泛化性能

3.更好地适应不同的任务和网络结构：AdamW一般比传统的Adam算法表现更好，特别是在处理大规模数据和复杂网络结构时

**（15）训练**

```python
def train(train_loader, model, criterion, optimizer):
    # 将模型设置为训练模式
    model.train()
    # 初始化训练损失为0，用于记录每个批次的损失并计算平均训练损失
    train_loss = 0.0
    # 遍历训练数据集
    for i, (input, target) in enumerate(train_loader):
        # 通过模型进行前向传播，得到模型的输出
        output = model(input)
        # 将目标值转换为浮点型的张量
        target = torch.tensor(target, dtype=torch.float32)
        # 将标签转换为长整型的张量
        target = target.to(dtype=torch.long)
        # 计算模型输出和目标值之间的损失
        loss = criterion(output, target)
        # 梯度清零，以防止梯度累积
        optimizer.zero_grad()
        # 反向传播，计算损失函数的梯度
        loss.backward()
        # 根据梯度更新模型参数
        optimizer.step()
        # 每训练20个批次，打印出当前批次的损失值
        if i % 20 == 0:
            print(loss.item())

        # 将当前批次的损失值累加到训练损失中
        train_loss += loss.item()

    # 返回平均的训练损失，即所有批次损失的平均值
    return train_loss / len(train_loader)
```

**补充：**

反向传播：

[【梯度下降】3D可视化讲解通俗易懂_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV18P4y1j7uH/?spm_id_from=333.788&vd_source=8c365458ad3da57c111e163a058ae355)

反向传播通过导数**链式法则**计算损失函数对各参数的**梯度**，并根据梯度进行参数的更新

加法节点：1

乘法节点：输入的值**交叉**相乘然后流入下一个节点

**（16）激活函数**

激活函数是神经网络中的一类函数，被用于在神经元中引入非线性变换

它的作用是对神经元的输入进行非线性映射，从而给予神经网络更强大的表达能力

在ResNet-18模型中，主要使用的激活函数是ReLU。ReLU函数将所有负值的输入置为0，并保持正值不变。这是一种简单而常用的非线性激活函数，可以有效地解决梯度消失问题和饱和问题

**（17）预测**

```python
def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    test_pred = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            # input = input.cuda()
            # target = target.cuda()

            output = model(input)
            test_pred.append(output.data.cpu().numpy())

    return np.vstack(test_pred)
```

`model.eval()` 是将模型切换到评估模式的函数调用。该调用会对模型及其各个组件（如批量归一化、Dropout等）进行相应的设置，以禁用与训练相关的操作，从而在评估阶段获得稳定的预测结果

`with torch.no_grad():` 可以用来包装需要禁用梯度计算的代码块，在该代码块中，所有的张量操作不会被记录用于自动求导，也不会产生计算图。这样可以减少内存的消耗，并提高代码执行的速度

`np.vstack` 可以将多个数组沿垂直方向堆叠在一起，形成一个新的数组