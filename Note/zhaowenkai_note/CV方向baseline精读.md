### 1.完整代码

```python
import glob  # 获取文件路径
import numpy as np
import pandas as pd
import nibabel as nib  # 处理医学图像数据
from collections import Counter  # 计数统计
from sklearn.linear_model import LogisticRegression

# 读取训练集文件路径
train_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Test/*')

# 打乱训练集和测试集的顺序
np.random.shuffle(train_path)
np.random.shuffle(test_path)


# 对PET文件提取特征
def extract_feature(path):
    # 加载PET图像数据
    img = nib.load(path)
    # 获取第一个通道的数据
    img = img.dataobj[:, :, :, 0]

    # 随机筛选其中的10个通道提取特征
    random_img = img[:, :, np.random.choice(range(img.shape[2]), 10)]

    # 对图片计算统计值
    feat = [
        (random_img != 0).sum(),  # 非零像素的数量
        (random_img == 0).sum(),  # 零像素的数量
        random_img.mean(),  # 平均值
        random_img.std(),  # 标准差
        len(np.where(random_img.mean(0))[0]),  # 在列方向上平均值不为零的数量
        len(np.where(random_img.mean(1))[0]),  # 在行方向上平均值不为零的数量
        random_img.mean(0).max(),  # 列方向上的最大平均值
        random_img.mean(1).max()  # 行方向上的最大平均值
    ]

    # 根据路径判断样本类别（'NC'表示正常，'MCI'表示异常）
    if 'NC' in path:
        return feat + ['NC']
    else:
        return feat + ['MCI']


# 对训练集进行30次特征提取，每次提取后的特征以及类别（'NC'表示正常，'MCI'表示异常）被添加到train_feat列表中。
train_feat = []
for _ in range(30):
    for path in train_path:
        train_feat.append(extract_feature(path))

# 对测试集进行30次特征提取
test_feat = []
for _ in range(30):
    for path in test_path:
        test_feat.append(extract_feature(path))

# 使用训练集的特征作为输入，训练集的类别作为输出，对逻辑回归模型进行训练。
m = LogisticRegression(max_iter=1000)
m.fit(
    np.array(train_feat)[:, :-1].astype(np.float32),  # 特征
    np.array(train_feat)[:, -1]  # 类别
)

# 对测试集进行预测并进行转置操作，使得每个样本有30次预测结果。
test_pred = m.predict(np.array(test_feat)[:, :-1].astype(np.float32))
test_pred = test_pred.reshape(30, -1).T

# 对每个样本的30次预测结果进行投票，选出最多的类别作为该样本的最终预测类别，存储在test_pred_label列表中。
test_pred_label = [Counter(x).most_common(1)[0][0] for x in test_pred]

# 生成提交结果的DataFrame，其中包括样本ID和预测类别。
submit = pd.DataFrame(
    {
        'uuid': [int(x.split('\\')[-1][:-4]) for x in test_path],  # 提取测试集文件名中的ID
        'label': test_pred_label  # 预测的类别
    }
)

# 按照ID对结果排序并保存为CSV文件
submit = submit.sort_values(by='uuid')
submit.to_csv('submit1.csv', index=None)
```

### 2.代码精读

**（1）glob模块**

```python
# 读取训练集文件路径
train_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Test/*')
```

glob模块和os模块都是python中常用的文件和路径操作模块。

glob模块支持的通配符：

| 通配符 | 功能                                                      |
| :----- | :-------------------------------------------------------- |
| *      | 匹配0或多个字符                                           |
| **     | 匹配所有文件、目录、子目录和子目录里的文件（3.5版本新增） |
| ?      | 匹配1个字符，与正则表达式里的?不同                        |
| [exp]  | 匹配指定范围内的字符，如：[1-9]匹配1至9范围内的字符       |
| [!exp] | 匹配不在指定范围内的字符                                  |

**区别：**

glob模块支持通配符匹配，其主要用途是返回符合匹配模式的文件和目录的列表；但不支持递归搜索

os模块提供了更多底层的文件和目录操作功能、支持递归搜索(`os.walk`)；但不支持通配符操作

**（2）numpy模块**

深度学习中需要对大量的参数进行运算，而numpy使用的是C语言实现的底层数据结构，通过矢量化操作和广播机制进行高效的数值计算；

numpy提供了一种称为`ndarray`的多维数组对象，可以高效地进行数组操作和数值计算； 

numpy的广播机制允许在形状不匹配的数组之间进行操作，而无需进行显式的循环迭代；

```python
import numpy as np


arr1 = np.array([[0, 0, 0],[1, 1, 1],[2, 2, 2], [3, 3, 3]])  # arr1.shape = (4,3)，也就是4行3列的二维数组
'''
[[0 0 0]
 [1 1 1]
 [2 2 2]
 [3 3 3]]
'''
arr2 = np.array([1, 2, 3])    #arr2.shape = (3,)，也就是有3个元素的一维数组
'''
[1 2 3]
'''
arr_sum = arr1 + arr2
print(arr_sum)

输入结果如下:
'''
[[1 2 3]
[2 3 4]
[3 4 5]
[4 5 6]]
'''
```

numpy提供了大量的数学函数和线性代数工具，包括傅里叶变换、随机数生成、矩阵计算等

**用处：**

`ndarray`对象可以表示图片、视频、文本、音频等多种类型的数据，并提供了丰富的数据处理和转换方法；

可以用于数据集的加载、预处理和分割，以及模型的训练和预测过程中对数据的处理。它可以进行张量运算、批量计算和损失函数的计算等操作；

numpy结合其他库(如matplotlib)，可以更容易地将数据可视化为图像、柱状图、散点图等，并以可视化的方式分析模型的性能和结果

**（3）pandas模块**

pandas提供了简单而强大的数据结构，如DataFrame和Series，支持数据的选择、过滤、排序、合并等操作，可对数据进行预处理和特征工程；

pandas提供了丰富的数据分析函数和工具，如统计描述、聚合计算、数据透视表、时间序列操作等，用于深入了解数据的特征和关系；

pandas可以结合matplotlib等绘图库，提供了简单的绘图接口，支持绘制折线图、柱状图、散点图等多种类型的图表，有助于数据可视化和结果展示；

pandas支持各种格式的数据导入和导出，包括csv、Excel、SQL数据库、JSON、HDF5等

```python
# 生成提交结果的DataFrame，其中包括样本ID和预测类别。
submit = pd.DataFrame(
    {
        'uuid': [int(x.split('/')[-1][:-4]) for x in test_path],  # 提取测试集文件名中的ID
        'label': test_pred_label  # 预测的类别
    }
)

# 按照ID对结果排序并保存为CSV文件
submit = submit.sort_values(by='uuid')
submit.to_csv('submit1.csv', index=None)
```

**用处：**

数据准备和预处理：处理缺失值、异常值、重复值等数据问题，并进行数据规范化和离散化；

特征工程：创建新的特征，处理分类变量、时间序列数据等，以供深度学习模型使用；

数据探索和可视化：进行数据分组、统计计算、数据透视和相关性分析，绘制直观的图表和可视化结果；

数据集处理：加载、处理和转换不同来源和类型的数据集

**（4）第三方库nibabel**

本次夏令营比赛的图片数据格式为NIfTI，所以用nibabel模块

nibabel图像由3D或4D图像数据数组组成

```python
# 获取第一个通道的数据
    img = img.dataobj[:, :, :, 0]
```

在NIfTI图像中，数据存储在`dataobj`属性中，其中四个冒号（`:, :, :, 0`）用于切片操作。第一个冒号表示在x轴上选择所有的切片，第二个冒号表示在y轴上选择所有的切片，第三个冒号表示在z轴上选择所有的切片，0代表选择图像数据中的第一个通道或者说是第一个时间点。

```python
# 随机筛选其中的10个通道提取特征
    random_img = img[:, :, np.random.choice(range(img.shape[2]), 10)]
```

这一行代码的作用是随机筛选出其中的10个通道，并将它们存储在`random_img`中

`np.random.choice(range(img.shape[2]), 10)`会在图像的第三个维度（z轴）上进行随机选择，选择出10个不同的索引

**（5）特征提取**

```python
# 对训练集进行30次特征提取，每次提取后的特征以及类别（'NC'表示正常，'MCI'表示异常）被添加到train_feat列表中。
train_feat = []
for _ in range(30):
    for path in train_path:
        train_feat.append(extract_feature(path))
```

对每一个path中的数据进行特征提取，将extract_feature中return回来的标签添加到train_feat列表中。我们通常需要训练模型来学习特征与标签之间的关系，然后使用这个模型来预测新样本的标签

**（6）sklearn模块**

最常用的机器学习库之一，内置了许多经典和先进的机器学习算法，包括分类、回归、聚类、降维和模型选择等。它还提供了用于数据预处理、特征选择、特征提取和模型评估等常用工具。本次用到的是逻辑回归模型

```python
# 使用训练集的特征作为输入，训练集的类别作为输出，对逻辑回归模型进行训练。
m = LogisticRegression(max_iter=1000)
m.fit(
    np.array(train_feat)[:, :-1].astype(np.float32),  # 特征
    np.array(train_feat)[:, -1]  # 类别
)

# 对测试集进行预测并进行转置操作，使得每个样本有30次预测结果。
test_pred = m.predict(np.array(test_feat)[:, :-1].astype(np.float32))
test_pred = test_pred.reshape(30, -1).T
```

先创建一个`LogisticRegression`对象，`max_iter`是一个可选参数，用于指定迭代的最大次数。迭代次数是指算法在训练过程中进行参数优化的迭代次数，通常用于控制算法的收敛速度

`m.fit()`方法接受训练数据的特征和类别作为参数，并使用这些数据来拟合训练模型。-1是因为之前将类别append在对应特征之后

`test_pred = m.predict(np.array(test_feat)[:, :-1].astype(np.float32))`是使用训练好的逻辑回归模型`m`对测试集进行预测，得到测试集的预测结果

`reshape(30, -1)`将一维数组重新排列为30行和未知列（根据数组的大小和形状）。然后，`.T`操作将结果进行转置，使得每一列对应于一个样本的30次预测结果

**（7）常见小操作**

```python
# 打乱训练集和测试集的顺序
np.random.shuffle(train_path)
np.random.shuffle(test_path)
```

如果训练集的样本按照特定的顺序排列（例如按照类别或其他特征进行排序），那么模型在学习过程中可能会受到数据顺序的影响；

打乱训练集，可以确保模型在每个训练批次中都接触到不同的样本，提供了更多的样本多样性，进而帮助模型更好地推广到未见过的数据上；

使得模型更难记住特定的样本分布和顺序，从而帮助减少过拟合的风险

**（8）略微深入讲解sklearn**

逻辑回归是一种广义线性回归模型，是`Sigmoid`函数归一化后的线性回归模型，它假设数据服从伯努利分布，通过梯度下降法对其损失函数（极大似然函数）求解，以达到数据二分类的目的

**基本原理是**其函数表达式为对数几率函数，通过`Sigmoid`函数将线性回归方程转化，将任何实数映射到（0，1）之间

![902c0e522439da64098eb049c6eba11](zhaowenkai_note\picture\902c0e522439da64098eb049c6eba11.png)

逻辑回归的损失函数

![fbd63b00b63f4c0b6a613ce3365c7a6](zhaowenkai_note\picture\fbd63b00b63f4c0b6a613ce3365c7a6.png)

```python
sklearn.linear_model.LogisticRegression (penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None)
```

**`penalty`：**

用于指定正则化项的类型，默认是`'l2'`。可选的值包括`'l1'`（L1正则化）和`'l2'`（L2正则化）

L1正则化和L2正则化虽然都可以控制过拟合，但它们的效果并不相同。当正则化强度逐渐增大(即C逐渐变小)， 参数w的取值会逐渐变小，但L1正则化会将参数压缩为0，L2正则化只会让参数尽量小，不会取到0

**`C`：**

正则化超参数的倒数，默认为1.0。较小的`C`值表示较强的正则化

**`solver`：**

用于优化问题的算法，默认为`'lbfgs'`。可选的值包括`'newton-cg'`（拟牛顿法）、`'lbfgs'`（L-BFGS拟牛顿法）、`'liblinear'`（坐标轴下降法）和`'sag'`（随机平均梯度法）。

L-BFGS通过存储有限的历史信息来近似牛顿方向，从而减少了存储和计算开销。由于L-BFGS仅需要存储有限的历史信息，而不是完整的Hessian矩阵，因此它在处理大规模问题时表现出色。并且L-BFGS是一种二阶优化方法，相比于一阶梯度下降法（如随机梯度下降法），具有更快的收敛速度和更好的全局收敛性

对于小数据集，`"liblinear"`是一个不错的选择，而`"sag"和"saga"`对于大数据集更快。

对于多类问题，只有`"newton-cg"、"sag"、"saga"和"lbfgs"`处理多项式损失；`"liblinear"`仅限于`"OvR"`方案。

**`multi_class`：**

输入"ovr", "multinomial", "auto"来告知模型，我们要处理的分类问题的类型

`'ovr'`：表示分类问题是二分类，或让模型使用`"OvR"`的形式来处理多分类问题

`'multinomial'`：表示处理多分类问题，这种输入在参数 `solver='liblinear'`时不可用

`"auto"`：表示会根据数据的分类情况和其他参数来确定模型要处理的分类问题的类型。比如说，如果数据是二分类，或者`solver`的取值为`"liblinear"，"auto"`会默认选择`"ovr"`。反之，则会选择`"nultinomial"`

**`class_weight`：**

使用参数`class_weight`对样本标签进行一定的均衡，给少量的标签更多的权重，让模型更偏向少数类

```python
m.fit(
    np.array(train_feat)[:, :-1].astype(np.float32),  # 特征
    np.array(train_feat)[:, -1]  # 类别
)
```

这里的fit方法就不多说了，我们将在进阶阶段的深度学习代码中展示训练过程