# Datawhale夏令营-NLP学习笔记-代码精读

## 1.赛题解析

**基于论文摘要的文本分类与关键词抽取挑战赛**

https://challenge.xfyun.cn/topic/info?type=abstract-of-the-paper&ch=ymfk4uU

医学领域的文献库中蕴含了丰富的疾病诊断和治疗信息，如何高效地从海量文献中提取关键信息，进行疾病诊断和治疗推荐，对于临床医生和研究人员具有重要意义。

我们需要做的：让机器通过对论文摘要等信息的理解，判断该论文是否属于医学领域的文献。

### **任务示例**

输入：

论文信息，格式如下：

Inflammatory Breast Cancer: What to Know About This Unique, Aggressive Breast Cancer.，

[Arjun Menta, Tamer M Fouad, Anthony Lucci, Huong Le-Petross, Michael C Stauder, Wendy A Woodward, Naoto T Ueno, Bora Lim]，

Inflammatory breast cancer (IBC) is a rare form of breast cancer that accounts for only 2% to 4% of all breast cancer cases. Despite its low incidence, IBC contributes to 7% to 10% of breast cancer caused mortality. Despite ongoing international efforts to formulate better diagnosis, treatment, and research, the survival of patients with IBC has not been significantly improved, and there are no therapeutic agents that specifically target IBC to date. The authors present a comprehensive overview that aims to assess the present and new management strategies of IBC.，

Breast changes; Clinical trials; Inflammatory breast cancer; Trimodality care.

输出：

是

### **比赛数据**

训练集与测试集数据为CSV格式文件，各字段分别是标题、作者、摘要、关键词。

### **评价指标**

本次竞赛的评价标准采用F1_score，分数越高，效果越好。

## 2.解题思路

理解题意后可知，本次赛题为很经典的文本的二分类任务，也就是通过论文标题、摘要、作者这三个信息来判断该文献是否属于医学领域的文献，是则标签为1，不是则标签为0。针对文本分类任务，可以提供两种实践思路：

* 一种是使用传统的特征提取方法（如TF-IDF/BOW）结合机器学习模型，
* 另一种是使用预训练的BERT模型进行建模。

### 2.1**传统的机器学习法**

使用**特征提取 + 机器学习**的思路步骤如下：

1. **数据预处理**：首先，对文本数据进行预处理，包括文本清洗（如去除特殊字符、标点符号）、分词等操作。可以使用常见的NLP工具包（如NLTK或spaCy）来辅助进行预处理。
2. **特征提取**：使用TF-IDF（词频-逆文档频率）或BOW（词袋模型）方法将文本转换为向量表示。TF-IDF可以计算文本中词语的重要性，而BOW则简单地统计每个词语在文本中的出现次数。可以使用**scikit-learn**库的TfidfVectorizer或CountVectorizer来实现特征提取。
3. **构建训练集和测试集**：将预处理后的文本数据分割为训练集和测试集，确保数据集的样本分布均匀。
4. **选择机器学习模型**：根据实际情况选择适合的机器学习模型，如朴素贝叶斯、支持向量机（SVM）、随机森林等。这些模型在文本分类任务中表现良好。可以使用scikit-learn库中相应的分类器进行模型训练和评估。
5. **模型训练和评估**：使用训练集对选定的机器学习模型进行训练，然后使用测试集进行评估。评估指标可以选择准确率、精确率、召回率、F1值等。
6. **调参优化**：如果模型效果不理想，可以尝试调整特征提取的参数（如词频阈值、词袋大小等）或机器学习模型的参数，以获得更好的性能。

官方给的baseline使用的是机器学习方法，在解决机器学习问题时，一般会遵循以下流程：

![](https://s2.loli.net/2023/08/14/KfXaHiG7pQmyO4d.png)

#### **代码精读**

**代码目录结构**：

![](https://s2.loli.net/2023/08/14/pEax9IBu8wdciUy.png)

**有很多同学刚开始接触项目对于目录结构不熟悉，这里我给大家解释一下**

* data文件夹存储用于训练和测试的数据文件，一般包括测试集和训练集
* main.ipynb就是我们jupyter-notebook的工程文件了，我们编辑的代码都在这里
* submit.csv是我们的运行结果，也就是最后提交作品时要交的文件
* work文件夹作为一个工作目录，存储代码执行期间的临时结果、日志、中间文件等
* model存储训练好的模型文件。在机器学习和深度学习中，训练好的模型通常会保存到文件中以备后续的预测和推理使用

**完整的baseline代码如下**：

```python
# 导入pandas用于读取表格数据
import pandas as pd

# 导入BOW（词袋模型），可以选择将CountVectorizer替换为TfidfVectorizer（TF-IDF（词频-逆文档频率）），注意上下文要同时修改，亲测后者效果更佳
from sklearn.feature_extraction.text import CountVectorizer

# 导入LogisticRegression回归模型
from sklearn.linear_model import LogisticRegression

# 过滤警告消息
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# 读取数据集
train = pd.read_csv('./data/train.csv')  # 读取训练集数据
train['title'] = train['title'].fillna('')  # 填充训练集中的'title'列的缺失值为空字符串
train['abstract'] = train['abstract'].fillna('')  # 填充训练集中的'abstract'列的缺失值为空字符串

test = pd.read_csv('./data/testB.csv')  # 读取测试集数据
test['title'] = test['title'].fillna('')  # 填充测试集中的'title'列的缺失值为空字符串
test['abstract'] = test['abstract'].fillna('')  # 填充测试集中的'abstract'列的缺失值为空字符串

# 提取文本特征，生成训练集与测试集
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')  # 将训练集的'title'、'author'、'abstract'和'Keywords'列的文本内容合并成一列，存储在'text'列中
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')  # 将测试集的'title'、'author'和'abstract'列的文本内容合并成一列，存储在'text'列中

vector = CountVectorizer().fit(train['text'])  # 使用CountVectorizer提取文本特征，拟合训练集中的文本数据
train_vector = vector.transform(train['text'])  # 将训练集的文本数据转换为特征矩阵表示
test_vector = vector.transform(test['text'])  # 将测试集的文本数据转换为特征矩阵表示

# 引入模型
model = LogisticRegression()  # 初始化一个LogisticRegression模型

# 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
model.fit(train_vector, train['label'])  # 使用训练集的特征矩阵和标签训练模型

# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)  # 使用训练好的模型对测试集的特征矩阵进行预测，并将预测结果存储在测试集的'label'列

test['Keywords'] = test['title'].fillna('')  # 填充测试集中的'title'列的缺失值为空字符串
test[['uuid', 'Keywords', 'label']].to_csv('submit_task1.csv', index=None)  # 将测试集的'uuid'、'Keywords'和'label'列保存到一个CSV文件中，不包含行索引
```

**其中涉及到的包、模型、方法在这里给大家详细解读**

* **导入所需的库：Pandas用于数据处理，CountVectorizer用于提取文本特征，LogisticRegression用于分类，warnings用于过滤警告。**

* **Pandas是一个在Python编程语言中使用的开源数据处理和分析库。**它提供了高效的数据结构和数据分析工具，使你能够快速处理和分析结构化数据。Pandas最常用的数据结构是DataFrame，它类似于电子表格或关系型数据库中的表格，可以处理包含不同类型数据的行和列。

  使用Pandas，你可以进行数据清洗、转换、筛选、聚合等操作，还可以进行数据可视化和统计分析。Pandas的灵活性和强大的功能使其成为数据科学和数据分析领域的重要工具之一。

* **Sklearn 是一个机器学习、深度学习中非常常用的 Python 第三方库，内部封装了多种机器学习算法与数据处理算法，提供了包括数据清洗、数据预处理、建模调参、数据验证、数据可视化的全流程功能，是入门机器学习的必备工具**。

  通过使用 sklearn，你可以便捷地完成机器学习的整体流程，尝试使用多种模型完成训练与预测任务，而不需要再手动实现各种机器学习算法。

  此处介绍几个常用包：

  - sklearn.linear_model：线性模型，如线性回归、逻辑回归、岭回归等
  - sklearn.tree：树模型，一般为决策树
  - sklearn.neighbors：最近邻模型，常见如 K 近邻算法
  - sklearn.svm：支持向量机
  - sklearn.ensemble：集成模型，如 AdaBoost、GBDT等

* **CountVectorizer是scikit-learn库中的一个文本特征提取工具，用于将文本转换为特征向量**。它根据文本中单词（或者根据其他的n-gram模式）的出现频率来构建特征向量。

  CountVectorizer的工作原理：

  1. 首先，它会将文本中的单词进行分词，并统计每个单词的出现频率。
  2. 接着，对于每个文本样本，CountVectorizer会根据文本中单词的出现频率生成该样本的特征向量。特征向量的每个维度对应不同的单词，而维度的值则代表该单词在文本中出现的次数。
  3. 最后，CountVectorizer会将所有文本样本的特征向量组合成一个特征矩阵，其中每一行代表一个文本样本的特征向量。

  CountVectorizer通常用于将文本数据转换为机器学习算法能够处理的数值型输入，例如在分类、聚类、文本挖掘等任务中。它是文本特征提取中最简单和常用的方法之一。

* **LogisticRegression（逻辑回归）是一种用于解决二分类问题的机器学习算法，常用于分类任务**。

  逻辑回归的基本原理：

  1. 将输入特征进行线性组合，得到一个关于输入特征的线性函数。
  2. 将线性函数的结果通过逻辑函数进行映射，得到样本属于某一类的概率。
  3. 使用优化算法（例如最大似然估计或梯度下降）来拟合模型参数，以最大化预测正确的概率或最小化分类误差。

  其中常用的两种逻辑函数Logistic函数和Sigmoid函数

  ![](https://s2.loli.net/2023/08/18/qi4Sfvod193hras.png)

  ![sigmoid](https://s2.loli.net/2023/08/18/oqMGAcSXD37CNLh.png)

  逻辑回归通常用于二分类任务，其中目标变量只有两个可能的取值。然而，逻辑回归可以通过修改或扩展来处理多类分类问题。在实践中，逻辑回归广泛应用于许多领域，如医学、金融、自然语言处理等。

* **读取训练集和测试集的数据**：使用`pd.read_csv()`方法读取一个CSV文件，并使用`fillna()`方法填充缺失值。

  例如baseline中的：

  ```python
  train = pd.read_csv('./data/train.csv')  # 读取训练集数据
  train['title'] = train['title'].fillna('')  # 填充训练集中的'title'列的缺失值为空字符串
  train['abstract'] = train['abstract'].fillna('')  # 填充训练集中的'abstract'列的缺失值为空字符串
  ```

* **提取文本特征**：将训练集的’title’、‘author’、'abstract’和’Keywords’列的文本内容合并成一列，并使用CountVectorizer提取文本特征。然后，使用`transform`方法将文本转换为稀疏矩阵表示(说白了就是把一堆文字变成了机器能处理的数字)

  例如baseline中的：

  ```python
  # 提取文本特征，生成训练集与测试集
  train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')  # 将训练集的'title'、'author'、'abstract'和'Keywords'列的文本内容合并成一列，存储在'text'列中
  test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')  # 将测试集的'title'、'author'和'abstract'列的文本内容合并成一列，存储在'text'列中
  vector = CountVectorizer().fit(train['text'])  # 使用CountVectorizer提取文本特征，拟合训练集中的文本数据
  train_vector = vector.transform(train['text'])  # 将训练集的文本数据转换为特征矩阵表示
  test_vector = vector.transform(test['text'])  # 将测试集的文本数据转换为特征矩阵表示
  ```

* **初始化模型**：使用LogisticRegression()初始化一个逻辑回归模型。

  例如baseline中的：

  ```python
  # 引入模型
  model = LogisticRegression()  # 初始化一个LogisticRegression模型
  ```

* **训练模型**：使用`fit`方法训练模型，其中`train_vector`是训练集的特征矩阵，`train['label']`是训练集的标签。

  例如baseline中的：

  ```python
  # 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
  model.fit(train_vector, train['label'])  # 使用训练集的特征矩阵和标签训练模型
  ```

* **预测测试集标签**：使用训练好的模型对测试集的文本特征进行预测，将预测结果存储在测试集的’label’列

  例如baseline中的：

  ```python
  # 利用模型对测试集label标签进行预测
  test['label'] = model.predict(test_vector)  # 使用训练好的模型对测试集的特征矩阵进行预测，并将预测结果存储在测试集的'label'列
  ```

* **保存结果**：将测试集的’uuid’、'Keywords’和’label’列保存到一个CSV文件中

  例如baseline中的：

  ```python
  test['Keywords'] = test['title'].fillna('')  # 填充测试集中的'title'列的缺失值为空字符串
  test[['uuid', 'Keywords', 'label']].to_csv('submit_task1.csv', index=None)  # 将测试集的'uuid'、'Keywords'和'label'列保存到一个CSV文件中，不包含行索引
  ```

  这时候小白就问了，为什么要这样保存呢？？？

  因为！！！

  ![](https://s2.loli.net/2023/08/14/LAnkMGH3BDVYsxT.png)

  因为平台的验证集需要这样的格式才能给你评分(就跟平时给老师交作业的格式一个道理)

### 2.2**基于BERT模型的深度学习法**

* 思路：用Bert将文本转换为表征向量，再输入两层全连接网络，使用BCE损失函数进行训练。

* 该模型需要用到torch框架，没有安装的可以在我的仓库找到最新教程

  **[torch安装](https://github.com/Yfcan/TYUT-TreamLearning/blob/main/Note/yangfeifan_note/%E6%9C%80%E6%96%B0%E7%89%88Pytorch%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B%EF%BC%88GPU%EF%BC%89.md)**

**bert微调具体思路步骤如下**

1. **数据预处理**：首先，对文本数据进行预处理，包括文本清洗（如去除特殊字符、标点符号）、分词等操作。可以使用常见的NLP工具包（如NLTK或spaCy）来辅助进行预处理。
2. **构建训练所需的dataloader与dataset**，构建Dataset类时，需要定义三个方法`__init__`，`__getitem__`，` __len__`，其中`__init__`方法完成类初始化，`__getitem__`要求返回返回内容和label，`__len__`方法返回数据长度
3. **构造Dataloader**，在其中完成对句子进行编码、填充、组装batch等动作：
4. **定义预测模型利用预训练的BERT模型来解决文本二分类任务**，我们将使用BERT模型编码中的[CLS]向量来完成二分类任务
5. **模型训练和评估**：使用训练集对选定的机器学习模型进行训练，然后使用测试集进行评估。评估指标可以选择准确率、精确率、召回率、F1值等。
6. **调参优化**：如果模型效果不理想，可以尝试调整特征提取的参数（如词频阈值、词袋大小等）或机器学习模型的参数，以获得更好的性能。

在这个进阶实践中，我们使用深度学习方法，一般会遵循以下流程：

![](https://s2.loli.net/2023/08/14/nBDGVUZKN7kdJfS.png)

#### **代码精读**

**代码目录结构**：

![](https://s2.loli.net/2023/08/14/vTqFgAwlny8U62u.png)

这时候我们就可以看到model目录的使用了

**完整的Bert微调分类代码如下**

```python
#Bert微调分类 + 余弦相似度选择关键词
#导入前置依赖
import os  # 导入操作系统相关的模块
import pandas as pd  # 导入用于数据处理的pandas库
import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch的nn模块
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据加载和处理模块
from transformers import AutoTokenizer  # 导入用于加载BERT模型的分词器
from transformers import BertModel  # 导入BERT模型
from pathlib import Path  # 导入用于处理文件路径的模块

batch_size = 16  # 批量大小
text_max_length = 128  # 文本的最大长度
epochs = 50  # 总训练的epochs数
lr = 3e-5  # 学习率
validation_ratio = 0.1  # 取多少训练集的数据作为验证集
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否可用GPU
log_per_step = 50  # 每多少步，打印一次loss

# 设置数据集存储路径
dataset_dir = Path("./data")
# 如果目录不存在，则创建目录
os.makedirs(dataset_dir) if not os.path.exists(dataset_dir) else ''

# 设置模型存储路径
model_dir = Path("./model/bert_checkpoints")
# 如果目录不存在，则创建目录
os.makedirs(model_dir) if not os.path.exists(model_dir) else ''

print("Device:", device)  # 打印当前使用的设备

# 读取训练集数据，并处理缺失值
pd_train_data = pd.read_csv('./data/train.csv')
pd_train_data['title'] = pd_train_data['title'].fillna('')
pd_train_data['abstract'] = pd_train_data['abstract'].fillna('')

# 读取测试集数据，并处理缺失值
test_data = pd.read_csv('./data/testB.csv')
test_data['title'] = test_data['title'].fillna('')
test_data['abstract'] = test_data['abstract'].fillna('')

# 将训练集中的多个字段拼接为一个文本字段
pd_train_data['text'] = pd_train_data['title'].fillna('') + ' ' + pd_train_data['author'].fillna('') + ' ' + pd_train_data['abstract'].fillna('')+ ' ' + pd_train_data['Keywords'].fillna('')

# 将测试集中的多个字段拼接为一个文本字段
test_data['text'] = test_data['title'].fillna('') + ' ' + test_data['author'].fillna('') + ' ' + test_data['abstract'].fillna('')+ ' ' + pd_train_data['Keywords'].fillna('')

# 从训练集中随机采样测试集
validation_data = pd_train_data.sample(frac=validation_ratio)
train_data = pd_train_data[~pd_train_data.index.isin(validation_data.index)]

# 构建自定义的Dataset类
class MyDataset(Dataset):
    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()
        self.mode = mode
        
        # 根据mode设置对应的数据集
        if mode == 'train':
            self.dataset = train_data
        elif mode == 'validation':
            self.dataset = validation_data
        elif mode == 'test':
            self.dataset = test_data
        else:
            raise Exception("Unknown mode {}".format(mode))

    def __getitem__(self, index):
        # 获取指定索引的数据
        data = self.dataset.iloc[index]
        text = data['text']  # 获取文本内容
        
        if self.mode == 'test':
            label = data['uuid']  # 在测试模式下，将uuid作为标签
        else:
            label = data['label']  # 获取数据标签
        
        return text, label  # 返回文本内容和标签

    def __len__(self):
        return len(self.dataset)  # 返回数据集的长度

# 创建训练集和验证集的Dataset对象
train_dataset = MyDataset('train')
validation_dataset = MyDataset('validation')

# 调用Dataset对象的__getitem__()方法，获取第一个样本
train_dataset.__getitem__(0)

# 获取Bert预训练模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#接着构造我们的Dataloader。
def collate_fn(batch):
    """
    将一个batch的文本句子转成tensor，并组成batch。
    :param batch: 一个batch的句子，例如: [('推文', target), ('推文', target), ...]
    :return: 处理后的结果，例如：
             src: {'input_ids': tensor([[ 101, ..., 102, 0, 0, ...], ...]), 'attention_mask': tensor([[1, ..., 1, 0, ...], ...])}
             target：[1, 1, 0, ...]
    """
    # 解压batch中的文本和标签
    text, label = zip(*batch)
    text, label = list(text), list(label)

    # 使用tokenizer对文本进行编码、填充和裁剪
    src = tokenizer(text, padding='max_length', max_length=text_max_length, return_tensors='pt', truncation=True)

    return src, torch.LongTensor(label)

# 创建训练集和验证集的DataLoader对象，并使用collate_fn对数据进行处理
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 获取一个batch的数据并输出
inputs, targets = next(iter(train_loader))
print("inputs:", inputs)
print("targets:", targets)

# 定义预测模型，该模型由Bert模型加上最后的预测层组成
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # 加载Bert模型
        self.bert = BertModel.from_pretrained('bert-base-uncased', mirror='ustc')

        # 最后的预测层
        self.predictor = nn.Sequential(
            nn.Linear(768, 256),  # 输入维度为768，输出维度为256
            nn.ReLU(),  # 使用ReLU激活函数
            nn.Linear(256, 1),  # 输入维度为256，输出维度为1
            nn.Sigmoid()  # 使用Sigmoid激活函数
        )

    def forward(self, src):
        """
        :param src: 分词后的推文数据
        """

        # 将src直接序列解包传入Bert，因为Bert和tokenizer是一套的，所以可以这么做。
        # 得到encoder的输出，使用[CLS]的输出作为最终线性层的输入
        outputs = self.bert(**src).last_hidden_state[:, 0, :]

        # 使用线性层进行最终的预测
        return self.predictor(outputs)

# 定义模型
model = MyModel()
model = model.to(device)

# 定义损失函数和优化器。使用 Binary Cross Entropy Loss 和 Adam 优化器
criteria = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 由于 inputs 是字典类型，定义一个辅助函数帮助将数据移动到指定设备
def to_device(dict_tensors):
    result_tensors = {}
    for key, value in dict_tensors.items():
        result_tensors[key] = value.to(device)
    return result_tensors

# 定义一个验证方法，用于计算验证集的精确率和损失
def validate():
    model.eval()
    total_loss = 0.
    total_correct = 0

    for inputs, targets in validation_loader:
        inputs, targets = to_device(inputs), targets.to(device)
        outputs = model(inputs)
        loss = criteria(outputs.view(-1), targets.float())
        total_loss += float(loss)

        # 计算正确预测的数量
        correct_num = (((outputs >= 0.5).float() * 1).flatten() == targets).sum()
        total_correct += correct_num

    return total_correct / len(validation_dataset), total_loss / len(validation_dataset)

# 将模型设置为训练模式
model.train()

# 清空一下cuda缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 定义几个变量，帮助打印loss
total_loss = 0.
# 记录步数
step = 0

# 记录在验证集上最好的准确率
best_accuracy = 0

# 开始训练
for epoch in range(epochs):  # 对每个epoch进行循环

    model.train()  # 设置模型为训练模式

    for i, (inputs, targets) in enumerate(train_loader):  # 遍历训练数据加载器
        # 从batch中拿到训练数据
        inputs, targets = to_device(inputs), targets.to(device)  # 将输入数据和目标数据移动到相应的设备上

        # 传入模型进行前向传递
        outputs = model(inputs)  # 模型进行前向传播，得到输出

        # 计算损失
        loss = criteria(outputs.view(-1), targets.float())  # 计算损失

        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()  # 清空梯度

        total_loss += float(loss)  # 累加本批次的损失值
        step += 1  # 更新步数

        if step % log_per_step == 0:  # 如果达到指定的步数
            print("Epoch {}/{}, Step: {}/{}, total loss:{:.4f}".format(epoch+1, epochs, i, len(train_loader), total_loss))
            total_loss = 0  # 打印累加的损失值，并重置为0

        del inputs, targets  # 删除输入数据和目标数据的引用，释放内存

    # 一个epoch后，使用验证集进行验证
    accuracy, validation_loss = validate()  # 使用验证集进行验证，得到准确率和验证损失
    print("Epoch {}, accuracy: {:.4f}, validation loss: {:.4f}".format(epoch+1, accuracy, validation_loss))

    torch.save(model, model_dir / f"model_{epoch}.pt")  # 保存当前模型到指定路径

    # 保存最好的模型
    if accuracy > best_accuracy:  # 如果当前准确率更高于最佳准确率
        torch.save(model, model_dir / f"model_best.pt")  # 将当前模型保存为最佳模型
        best_accuracy = accuracy  # 更新最佳准确率
        
# 加载最佳模型并进行测试集的预测
model = torch.load(model_dir / f"model_best.pt")  # 加载最佳模型
model = model.eval()  # 设置模型为评估模式
test_dataset = MyDataset('test')  # 创建测试集对象
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  # 创建测试数据加载器
results = []  # 存储预测结果

for inputs, ids in test_loader:  # 遍历测试数据加载器
    outputs = model(inputs.to(device))  # 使用模型进行预测
    outputs = (outputs >= 0.5).int().flatten().tolist()  # 将概率转换为二分类结果
    ids = ids.tolist()  # 将id转换为列表形式
    results = results + [(id, result) for result, id in zip(outputs, ids)]  # 将结果与对应的id组合并存储

```

**其中涉及到的包、模型、方法在这里给大家详细解读**

* 这里导入了一些必要的库和模块，包括操作系统相关的os模块，用于数据处理的pandas库，深度学习框架PyTorch的相关模块，以及应用于自然语言处理任务的BERT模型和分词器相关模块

  ```python
  import os  # 导入操作系统相关的模块
  import pandas as pd  # 导入用于数据处理的pandas库
  import torch  # 导入PyTorch库
  from torch import nn  # 导入PyTorch的nn模块
  from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据加载和处理模块
  from transformers import AutoTokenizer  # 导入用于加载BERT模型的分词器
  from transformers import BertModel  # 导入BERT模型
  from pathlib import Path  # 导入用于处理文件路径的模块
  
  batch_size = 16  # 批量大小
  text_max_length = 128  # 文本的最大长度
  epochs = 50  # 总训练的epochs数
  lr = 3e-5  # 学习率
  validation_ratio = 0.1  # 取多少训练集的数据作为验证集
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否可用GPU
  log_per_step = 50  # 每多少步，打印一次loss
  ```

* **PyTorch 是一个基于 Python 的深度学习框架**，它提供了用于构建和训练神经网络的高级接口和工具。PyTorch 被广泛应用于机器学习和人工智能领域，特别是在计算机视觉、自然语言处理等任务中。

  PyTorch 的核心功能包括：

  1. 张量操作：PyTorch 提供了强大的张量操作功能，类似于 NumPy，用于在 CPU 或 GPU 上执行数值计算。
  2. 自动微分：PyTorch 提供自动微分（Autograd）功能，它能够根据定义的计算图自动计算张量操作的导数，方便进行反向传播和梯度下降等优化算法。
  3. 神经网络模块：PyTorch 提供了丰富的神经网络模块和函数，可用于构建各种类型的神经网络模型，包括卷积神经网络（CNN）、递归神经网络（RNN）、生成对抗网络（GAN）等。
  4. 模型训练与部署：PyTorch 提供了易于使用的工具和函数，用于定义训练循环、计算损失函数、应用优化算法等。此外，PyTorch 还支持将训练好的模型导出为可部署的格式，方便在生产环境中使用。
  5. 支持多种硬件：PyTorch 支持在多种硬件平台上运行，包括 CPU、GPU 和专用加速器（如 NVIDIA 的 CUDA 平台）等。

  PyTorch 提供了简单易用的 Python API，使得构建和训练深度学习模型更加直观和灵活。它也是一个开源项目，并有活跃的社区支持，提供了广泛的教程、文档和示例，方便用户学习和使用

* **BERT（Bidirectional Encoder Representations from Transformers）**是一种基于Transformer模型的预训练语言表示模型。由Google在2018年发布，它在自然语言处理（NLP）领域取得了重要的突破。BERT 乃至目前正火的 LLM 的成功，都离不开 Attention 机制与基于 Attention 机制搭建的 Transformer 架构.

  BERT模型的核心思想是**通过在大规模未标记的文本数据上进行无监督的预训练，学习得到丰富的语言表示**。BERT采用**双向（bidirectional）的Transformer模型**结构，使得模型可以同时考虑到一个词的上下文信息。

  具体来说，BERT模型使用了两个主要任务进行预训练：掩码语言模型（Masked Language Model，MLM）和下一句预测（Next Sentence Prediction，NSP）。

  1. **掩码语言模型（MLM）**：在预训练阶段，BERT模型会将输入文本中的一些词随机选择并替换为掩码符号（例如，将"我爱吃苹果"替换为"我爱吃[掩码]"），然后让模型预测被掩码的词。这样的预训练方式可以使得模型能够有效地学习到词语之间的上下文关系。
  2. **下一句预测（NSP）**：在预训练阶段，BERT模型也会使用一对连续的句子作为输入，并让模型预测这两个句子是否是连续的。这样的任务可以帮助模型学习到句子级别的语义关系。

  通过这样的预训练方式，BERT模型可以学习到丰富的语言表示，包括词语级别和句子级别的语义信息。在实际应用中，可以使用预训练好的BERT模型作为特征提取器，或者使用其作为初始模型，在特定任务上进行微调。

* **Transformer**

  **Transformer是一种用于自然语言处理和机器翻译等任务的深度学习模型**，通过自我注意力机制（Self-Attention）实现了高效的序列建模和特征提取。Transformer模型主要由两个关键组件构成：

  1. **自我注意力机制（Self-Attention）**：自我注意力机制用于计算输入序列中各个位置的重要性权重，将每个位置与其他位置的信息进行交互和整合。它通过计算每个位置与其它位置的相关度得到一个权重分布，用于加权求和得到每个位置的上下文表示。
  2. **前馈神经网络（Feed-Forward Neural Network）**：前馈神经网络是Transformer模型中的另一个组件，用于在每个位置独立地对自我注意力机制的输出进行进一步的非线性变换和特征提取。

  Transformer模型通过自我注意力机制实现了对输入序列内部的位置和位置之间的关系进行建模，使得模型能够有效地处理长序列和捕获复杂的依赖关系，为自然语言处理任务带来了重要的突破。

* **Attention机制**

  Attention 机制的特点是通过计算 **Query** (查询值)与**Key**(键值)的相关性为真值加权求和，从而拟合序列中每个词同其他词的相关关系。其大致计算过程如图：

  ![](https://s2.loli.net/2023/08/18/XRep2AMQtBuNSPg.png)

  Transformer 正是基于 Attention 机制搭建了 Encoder-Decoder（编码器-解码器）结构，主要适用于 Seq2Seq（序列到序列）任务，即输入是一个自然语言序列，输出也是一个自然语言序列。其整体架构如下：

  ![](https://s2.loli.net/2023/08/18/nziFNSIEreZOQ8d.png)

  Transformer 由一个 Encoder，一个 Decoder 外加一个 Softmax 分类器与两层编码层构成。上图中左侧方框为 Encoder，右侧方框为 Decoder。在训练时，句对会被划分为输入语料和输出语料，输入语料将从左侧通过编码层进入 Encoder，输出语料将从右侧通过编码层进入 Decoder。Encoder 的主要任务是对输入语料进行编码再输出给 Decoder，Decoder 再根据输出语料的历史信息与 Encoder 的输出进行计算，输出结果再经过一个线性层和 Softmax 分类器即可输出预测的结果概率。

* **DataSet**

  下面我们将解释**Dataset**，我们先简单了解一下该函数中三种方法的作用：
  
  * `__init__`：数据加载、预处理，并根据需要对数据集创建索引
  * `__getitem__`：通过提供索引返回数据，也就是提供 DataLoader获取数据的方式
  * `__len__`：返回数据集的长度，DataLoader依据 len 确定自身索引采样器的长度
  
  ```python
  class MyDataset(Dataset):
      def __init__(self, mode='train'):
          super(MyDataset, self).__init__()
          self.mode = mode
          
          # 根据mode设置对应的数据集
          if mode == 'train':
              self.dataset = train_data
          elif mode == 'validation':
              self.dataset = validation_data
          elif mode == 'test':
              self.dataset = test_data
          else:
              raise Exception("Unknown mode {}".format(mode))
  
      def __getitem__(self, index):
          # 获取指定索引的数据
          data = self.dataset.iloc[index]
          text = data['text']  # 获取文本内容
          
          if self.mode == 'test':
              label = data['uuid']  # 在测试模式下，将uuid作为标签
          else:
              label = data['label']  # 获取数据标签
        
          return text, label  # 返回文本内容和标签

      def __len__(self):
        return len(self.dataset)  # 返回数据集的长度
  ```

  当构建自定义的Dataset类时，我们需要继承PyTorch中的Dataset类，并重写其中的方法以适应我们的数据集。

  * 在`__init__`方法中，我们先调用父类的`__init__`方法，确保Dataset类的初始化得到正确的处理。然后，我们将传入的`mode`参数保存到实例变量中，以便后续使用。接下来，根据`mode`参数，我们选择相应的数据集。通过判断`mode`的值，我们分别将原始数据集赋值给`self.dataset`，其中`train_data`、`validation_data`和`test_data`是原始数据的变量或数组。如果`mode`参数不是’训练’、‘验证’或’测试’，则抛出异常。

  * 在`__getitem__`方法中，我们传入一个索引`index`，以获取指定索引的数据。首先，我们使用`index`值从`self.dataset`中获取对应的数据。根据数据集的结构，我们假设数据集是一个包含文本内容和标签的DataFrame。我们通过`data['text']`获取文本内容并将其赋值给变量`text`。如果`mode`为`text`，我们将使用`data['uuid']`作为标签，并将其赋值给`label`变量；否则，我们将使用`data['label']`作为标签。最后，我们返回文本内容`text`和标签`label`。

  * 在`__len__`方法中，我们返回数据集的长度，即`self.dataset`的长度，以提供在迭代时使用。
  
* **Tokenizer**

  Tokenizer是自然语言处理中的一个重要工具，它用于将文本序列分割成一个个离散的单元，例如单词、子词或字符。Tokenizer的主要目的是将连续的文本序列转换成模型可处理的离散输入。它在文本预处理和特征提取中扮演了关键的角色。

  **AutoTokenizer**是Hugging Face库中的一个类，这行代码使用了`AutoTokenizer`类，并通过`from_pretrained`方法加载了一个预训练的BERT模型的tokenizer,`"bert-base-uncased"`是指加载基于BERT模型的预训练权重。

  ```python
  # 获取Bert预训练模型的tokenizer
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  ```

* **Dataloader**

  DataLoader是PyTorch中的一个实用工具，用于数据的批量加载和并行处理。它可以将数据集封装成可迭代的批次，便于在训练过程中高效地加载和处理数据。

  这里我们定义了一个`collate_fn`函数，用于将一个批量的文本数据转换成模型所需的输入形式，并组成一个批量的张量数据。函数用于将一个批量的文本数据转换成模型所需的输入形式，并组成一个批量的张量数据。

  ```python
  def collate_fn(batch):
      """
      将一个batch的文本句子转成tensor，并组成batch。
      :param batch: 一个batch的句子，例如: [('推文', target), ('推文', target), ...]
      :return: 处理后的结果，例如：
               src: {'input_ids': tensor([[ 101, ..., 102, 0, 0, ...], ...]), 'attention_mask': tensor([[1, ..., 1, 0, ...], ...])}
               target：[1, 1, 0, ...]
      """
      # 解压batch中的文本和标签
      text, label = zip(*batch)
      text, label = list(text), list(label)
  
      # 使用tokenizer对文本进行编码、填充和裁剪
      src = tokenizer(text, padding='max_length', max_length=text_max_length, return_tensors='pt', truncation=True)
  
      return src, torch.LongTensor(label)
  ```

  具体解释如下：

  1. 解压文本和标签：利用`zip(*batch)`将一个批次的文本和标签进行解压，得到单独的文本列表（`text`）和标签列表（`label`）。
  2. 将文本和标签转为列表：将解压后的文本和标签转换为列表形式，以方便后续处理。
  3. 使用tokenizer对文本进行编码和处理：利用`tokenizer`对文本进行编码、填充和裁剪等处理。其中，`tokenizer`是之前使用`AutoTokenizer.from_pretrained("bert-base-uncased")`加载的BERT模型的tokenizer。通过传入`text`，`padding='max_length'`，`max_length=text_max_length`，`return_tensors='pt'`和`truncation=True`等参数，将文本转换为BERT模型所需的输入格式。`padding='max_length'`表示将文本填充至最大长度，`max_length=text_max_length`表示指定的最大长度，`return_tensors='pt'`表示返回PyTorch张量的格式，`truncation=True`表示对超过最大长度的文本进行裁剪。
  4. 返回处理后的结果：将处理后的文本（`src`）和标签（`label`）作为结果返回。其中，`src`是一个字典，包含两个键值对：'input_ids’表示编码后的文本张量，'attention_mask’表示表示解码时需要注意的位置的张量（标识有效token位置）。`label`是标签的张量表示。

  这样，`collate_fn`函数可以作为参数传递给`DataLoader`，在训练过程中对一个批量的文本数据进行处理，并返回模型所需的输入形式。

  下面我们通过DataLoader类创建train_loader和validation_loader对象，用于加载训练数据和验证数据，并进行数据批量化和预处理。

  ```python
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
  validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
  ```

  其中的参数和配置：

  - train_dataset：表示训练数据集的对象
  - validation_dataset：表示验证数据集的对象
  - batch_size：表示每个批次中的样本数量。通过设置合适的batch_size来控制在每次训练中使用的样本数量。
  - shuffle：表示是否在每个epoch开始时对数据进行洗牌。当设置为True时，可以打乱数据的顺序，以减少模型对顺序的依赖性。对于训练数据而言，通常需要进行洗牌；而对于验证数据，则不需要洗牌。默认值为False。
  - collate_fn：表示用于处理每个批次数据的函数。这个函数将接受一个批次的数据样本作为输入，并返回模型所需的输入形式。

* **Model**

  ```python
  # 定义预测模型，该模型由Bert模型加上最后的预测层组成
  class MyModel(nn.Module):
      def __init__(self):
          super(MyModel, self).__init__()
  
          # 加载Bert模型
          self.bert = BertModel.from_pretrained('bert-base-uncased', mirror='ustc')
  
          # 最后的预测层
          self.predictor = nn.Sequential(
              nn.Linear(768, 256),  # 输入维度为768，输出维度为256
              nn.ReLU(),  # 使用ReLU激活函数
              nn.Linear(256, 1),  # 输入维度为256，输出维度为1
              nn.Sigmoid()  # 使用Sigmoid激活函数
          )
  
      def forward(self, src):
          """
          :param src: 分词后的推文数据
          """
  
          # 将src直接序列解包传入Bert，因为Bert和tokenizer是一套的，所以可以这么做。
          # 得到encoder的输出，使用[CLS]的输出作为最终线性层的输入
          outputs = self.bert(**src).last_hidden_state[:, 0, :]
  
          # 使用线性层进行最终的预测
          return self.predictor(outputs)
  
  ```

  具体解释如下：

  1. 加载Bert模型：使用`BertModel.from_pretrained('bert-base-uncased', mirror='ustc')`加载预训练的BERT模型。在这里，使用的是`bert-base-uncased`预训练模型，并指定了`mirror='ustc'`参数以使用USTC镜像源。
  2. 定义最后的预测层：使用`nn.Sequential`定义了一个由多个线性层和激活函数组成的预测层。其中，第一个线性层的输入维度为768（BERT模型输出的隐藏状态维度），输出维度为256；接着使用ReLU激活函数；然后是第二个线性层，输入维度为256，输出维度为1；最后使用Sigmoid激活函数将输出值限制在0到1之间。
  3. 前向传播函数 `forward`：这个函数接收一个分词后的推文数据作为输入（即`src`），并进行前向计算。首先将`src`传入Bert模型，得到Bert模型的输出。在此例中，只使用了输出的`last_hidden_state`并提取了`[CLS]`位置的表示（即`[:, 0, :]`）作为最终线性层的输入。然后利用预测层对提取的表示进行最终的预测，得到预测结果。

  通过定义并使用`MyModel`模型，可以将Bert模型和自定义的最后的预测层结合起来，完成对文本数据的预测任务。

  * 注意：在这个预测模型中，使用ReLU函数作为第一个线性层后的激活函数，以引入非线性变换，并增加模型的拟合能力。而使用Sigmoid函数作为最后一个线性层后的激活函数，将输出限制在0到1之间，用于进行二分类任务中的概率预测。

  * 在预测模型的示例中，`forward`函数接收一个分词后的推文数据 `src` 作为输入，主要由以下两个步骤组成：

    1.使用BERT模型对输入进行编码和特征提取：将输入数据 `src` 传递给BERT模型，得到BERT模型的输出。具体来说，这里通过 `self.bert(**src)` 将 `src` 序列解包传入BERT，并获取BERT的输出 `outputs`。由于`src` 中只包含输入序列，而不包含其他参数，因此使用 `**src` 进行解包传入。

    2.使用最后的线性层进行预测：将 `outputs` 传递给最后的预测层 `self.predictor`，得到最终的预测结果。在这个示例中，`outputs` 是BERT模型的最后一个隐藏状态的表示，通过选择 `[CLS]` 位置的表示进行预测，最终得到二分类问题的预测结果。

* **损失函数**

  通过定义损失函数和优化器，可以在训练过程中使用损失函数计算模型的损失，并通过优化器来更新模型的参数，从而逐步优化模型以提高性能。

  ```python
  # 定义损失函数和优化器。使用 Binary Cross Entropy Loss 和 Adam 优化器
  criteria = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  ```

  1. 二分类交叉熵损失（`nn.BCELoss()`）：
     二分类交叉熵损失函数是一种常用于二分类任务的损失函数。在这个示例中，使用`nn.BCELoss()`来定义二分类交叉熵损失函数 `criteria`。
  2. Adam优化器：
     Adam是一种自适应学习率优化算法，在训练深度学习模型时广泛使用，它能够自适应地调整不同参数的学习率，并有助于加快模型的训练速度和提高收敛性。

* **辅助函数to_device**

  这里定义了一个辅助函数`to_device`，用于将字典中的张量数据移动到GPU上。

  ```python
  # 由于 inputs 是字典类型，定义一个辅助函数帮助将数据移动到指定设备
  def to_device(dict_tensors):
      result_tensors = {}
      for key, value in dict_tensors.items():
          result_tensors[key] = value.to(device)
      return result_tensors
  ```

* **validate**

  这是一个用于执行验证（validation）的函数`validate()`。该函数用于评估训练得到模型在验证数据集上的性能，并返回验证结果。

  ```python
  def validate():
      model.eval()
      total_loss = 0.
      total_correct = 0
  
      for inputs, targets in validation_loader:
          inputs, targets = to_device(inputs), targets.to(device)
          outputs = model(inputs)
          loss = criteria(outputs.view(-1), targets.float())
          total_loss += float(loss)
  
          # 计算正确预测的数量
          correct_num = (((outputs >= 0.5).float() * 1).flatten() == targets).sum()
          total_correct += correct_num
  
      return total_correct / len(validation_dataset), total_loss / len(validation_dataset)
  ```

  函数内部逐步执行以下操作：

  1. 将模型设置为评估模式(`model.eval()`)。这是为了确保在验证过程中，模型不进行梯度计算和参数更新，只进行正向传播和计算损失。
  2. 初始化总损失(`total_loss`)为0、总正确预测数量(`total_correct`)为0。
  3. 遍历验证数据集(`validation_loader`)中的每个批次(`inputs, targets`)，其中`inputs`是输入数据，`targets`是目标标签。
  4. 使用辅助函数`to_device`将`inputs`中的数据移动到指定设备，并将`targets`也移动到指定设备。
  5. 将移动后的`inputs`传入模型(`model`)进行正向传播，得到模型输出(`outputs`)。
  6. 使用定义的二分类交叉熵损失函数(`criteria`)计算模型输出与目标标签之间的损失(`loss`)。
  7. 将损失(`loss`)转化为浮点数(`float(loss)`)并累加到总损失(`total_loss`)中。
  8. 计算出`outputs`中大于等于0.5的元素对应的下标是否与`targets`中对应元素相等，并统计出正确预测的数量。
  9. 将正确预测的数量(`correct_num`)累加到总正确预测数量(`total_correct`)中。
  10. 返回验证集上的准确率(`total_correct / len(validation_dataset)`)和平均损失(`total_loss / len(validation_dataset)`)。其中准确率是正确预测的数量除以验证集样本总数，平均损失是总损失除以验证集样本总数。

* **train**

  正式开始训练模型

  ```python
  # 开始训练
  for epoch in range(epochs):  # 对每个epoch进行循环
  
      model.train()  # 设置模型为训练模式
  
      for i, (inputs, targets) in enumerate(train_loader):  # 遍历训练数据加载器
          # 从batch中拿到训练数据
          inputs, targets = to_device(inputs), targets.to(device)  # 将输入数据和目标数据移动到相应的设备上
  
          # 传入模型进行前向传递
          outputs = model(inputs)  # 模型进行前向传播，得到输出
  
          # 计算损失
          loss = criteria(outputs.view(-1), targets.float())  # 计算损失
  
          loss.backward()  # 反向传播，计算梯度
          optimizer.step()  # 更新模型参数
          optimizer.zero_grad()  # 清空梯度
  
          total_loss += float(loss)  # 累加本批次的损失值
          step += 1  # 更新步数
  
          if step % log_per_step == 0:  # 如果达到指定的步数
              print("Epoch {}/{}, Step: {}/{}, total loss:{:.4f}".format(epoch+1, epochs, i, len(train_loader), total_loss))
              total_loss = 0  # 打印累加的损失值，并重置为0
  
          del inputs, targets  # 删除输入数据和目标数据的引用，释放内存
  
      # 一个epoch后，使用验证集进行验证
      accuracy, validation_loss = validate()  # 使用验证集进行验证，得到准确率和验证损失
      print("Epoch {}, accuracy: {:.4f}, validation loss: {:.4f}".format(epoch+1, accuracy, validation_loss))
  
      torch.save(model, model_dir / f"model_{epoch}.pt")  # 保存当前模型到指定路径
  
      # 保存最好的模型
      if accuracy > best_accuracy:  # 如果当前准确率更高于最佳准确率
          torch.save(model, model_dir / f"model_best.pt")  # 将当前模型保存为最佳模型
          best_accuracy = accuracy  # 更新最佳准确率
  ```

  循环迭代逐步执行以下操作：

  1. 外层循环通过`range(epochs)`对每个epoch进行循环。
  2. 在每个epoch开始时，将模型设置为训练模式（`model.train()`）。
  3. 使用`enumerate(train_loader)`遍历训练数据加载器中的每个批次（`inputs, targets`）。
  4. 使用辅助函数`to_device`将输入数据`inputs`和目标数据`targets`移动到指定设备上。
  5. 将移动后的`inputs`传入模型（`model(inputs)`）进行正向传播，得到模型输出`outputs`。
  6. 使用定义的二分类交叉熵损失函数`criteria`计算模型输出与目标标签之间的损失`loss`。
  7. 调用`loss.backward()`进行反向传播，计算梯度。
  8. 调用`optimizer.step()`进行参数更新，即根据计算得到的梯度更新模型参数。
  9. 调用`optimizer.zero_grad()`清空梯度，为下个批次的计算做准备。
  10. 累加本批次的损失值到总损失中（`total_loss += float(loss)`）。
  11. 更新步数（`step += 1`）。
  12. 如果达到指定的步数（`step % log_per_step == 0`），打印当前训练状态，包括当前epoch、总步数、总损失等。
  13. 删除输入数据和目标数据的引用，以释放内存（`del inputs, targets`）。
  14. 在完成一个epoch后，使用验证集进行验证，调用`validate`函数得到准确率和验证损失。
  15. 打印当前epoch的准确率和验证损失。
  16. 使用`torch.save`保存当前模型到指定路径（`model_dir / f"model_{epoch}.pt"`）。
  17. 如果当前准确率更高于最佳准确率，将当前模型保存为最佳模型（`torch.save(model, model_dir / f"model_best.pt")`）并更新最佳准确率。

* **加载预测最佳模型**

  ```python
  # 加载最佳模型并进行测试集的预测
  model = torch.load(model_dir / f"model_best.pt")  # 加载最佳模型
  model = model.eval()  # 设置模型为评估模式
  test_dataset = MyDataset('test')  # 创建测试集对象
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  # 创建测试数据加载器
  results = []  # 存储预测结果
  
  for inputs, ids in test_loader:  # 遍历测试数据加载器
      outputs = model(inputs.to(device))  # 使用模型进行预测
      outputs = (outputs >= 0.5).int().flatten().tolist()  # 将概率转换为二分类结果
      ids = ids.tolist()  # 将id转换为列表形式
      results = results + [(id, result) for result, id in zip(outputs, ids)]  # 将结果与对应的id组合并存储
  ```

  在完成以上步骤后，`results`列表中存储了所有测试样本的预测结果，每个元素为一个二元组，表示样本的id和对应的预测结果。可以根据需要进一步分析或进行后续处理。

### 2.3进阶实战-Roberta微调+特征融合+后处理

* RoBERTa（Robustly Optimized BERT approach）是在BERT基础上进行改进和优化的一种预训练语言模型。

  它与Bert的区别在于：①Roberta在预训练的阶段中没有对下一句话进行预测（**NSP**）②采用了动态掩码 ③使用字符级和词级别表征的混合文本编码。

* 与Bert模型接分类器不同，进阶模型对网络结构进行了更进一步的改进，具体细节如下：

  在模型结构上使用了以下两个特征：

   ①特征1：MeanPooling(768维) -> fc(128维) 

   ②特征2：Last_hidden (768维) -> fc(128维)

   其中，特征1指的是将Roberta所输出的全部序列分词的表征向量**先进行一个平均池化再接一个全连接层**(fc，Fully Connected Layer)；特征2指的是将Roberta的**pooled_output接一个全连接层**(fc，Fully Connected Layer)。（pooled_output = [CLS]的表征向量接入一个全连接层，再输入至Tanh激活函数）

   然后，将这两个特征进行加权并相加即可输进分类器进行训练。

*  后处理方面，将训练好的模型用于推理测试集，并根据标签数目的反馈，对预测阈值进行调整。

在代码部分中，主要分为四个模块：1.数据处理 2.模型训练 3.模型评估 4.测试集推理

#### **代码精读**

**1.数据处理**

```python
from transformers import AutoTokenizer  # 导入AutoTokenizer类，用于文本分词
import pandas as pd  # 导入pandas库，用于处理数据表格
import numpy as np  # 导入numpy库，用于科学计算
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import torch  # 导入torch库，用于深度学习任务
from torch.nn.utils.rnn import pad_sequence  # 导入pad_sequence函数，用于填充序列，保证向量中各序列维度的大小一样

MAX_LENGTH = 128  # 定义最大序列长度为128

def get_train(model_name, model_dict):
    model_index = model_dict[model_name]  # 获取模型索引
    train = pd.read_csv('./dataset/train.csv')  # 从CSV文件中读取训练数据
    train['content'] = train['title'] + train['author'] + train['abstract']  # 将标题、作者和摘要拼接为训练内容
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=MAX_LENGTH, cache_dir=f'./premodels/{model_name}_saved')  # 实例化分词器对象
    # 通过分词器对训练数据进行分词，并获取输入ID、注意力掩码和标记类型ID（这个可有可无）
    input_ids_list, attention_mask_list, token_type_ids_list = [], [], []
    y_train = []  # 存储训练数据的标签
    
    for i in tqdm(range(len(train['content']))):  # 遍历训练数据
        sample = train['content'][i]  # 获取样本内容
        tokenized = tokenizer(sample, truncation='longest_first')  # 分词处理，使用最长优先方式截断
        input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']  # 获取输入ID和注意力掩码
        input_ids, attention_mask = torch.tensor(input_ids), torch.tensor(attention_mask)  # 转换为PyTorch张量
        try:
            token_type_ids = tokenized['token_type_ids']  # 获取标记类型ID
            token_type_ids = torch.tensor(token_type_ids)  # 转换为PyTorch张量
        except:
            token_type_ids = input_ids
        input_ids_list.append(input_ids)  # 将输入ID添加到列表中
        attention_mask_list.append(attention_mask)  # 将注意力掩码添加到列表中
        token_type_ids_list.append(token_type_ids)  # 将标记类型ID添加到列表中
        y_train.append(train['label'][i])  # 将训练数据的标签添加到列表中
    # 保存
    input_ids_tensor = pad_sequence(input_ids_list, batch_first=True, padding_value=0)  # 对输入ID进行填充，保证向量中各序列维度的大小一样，生成张量
    attention_mask_tensor = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)  # 对注意力掩码进行填充，保证向量中各序列维度的大小一样，生成张量
    token_type_ids_tensor = pad_sequence(token_type_ids_list, batch_first=True, padding_value=0)  # 对标记类型ID进行填充，保证向量中各序列维度的大小一样，生成张量
    x_train = torch.stack([input_ids_tensor, attention_mask_tensor, token_type_ids_tensor], dim=1)  # 将输入张量堆叠为一个张量
    x_train = x_train.numpy()  # 转换为NumPy数组
    np.save(f'./models_input_files/x_train{model_index}.npy', x_train)  # 保存训练数据
    y_train = np.array(y_train)  # 将标签列表转换为NumPy数组
    np.save(f'./models_input_files/y_train{model_index}.npy', y_train)  # 保存标签数据
    
def get_test(model_name, model_dict):
    model_index = model_dict[model_name]  # 获取模型索引
    test = pd.read_csv('./dataset/testB.csv')  # 从CSV文件中读取测试数据
    test['content'] = test['title'] + ' ' + test['author'] + ' ' + test['abstract']  # 将标题、作者和摘要拼接为测试内容
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=MAX_LENGTH,cache_dir=f'./premodels/{model_name}_saved')  # 实例化分词器对象
    # 通过分词器对测试数据进行分词，并获取输入ID、注意力掩码和标记类型ID（可有可无）
    input_ids_list, attention_mask_list, token_type_ids_list = [], [], []
    
    for i in tqdm(range(len(test['content']))):  # 遍历测试数据
        sample = test['content'][i]  # 获取样本内容
        tokenized = tokenizer(sample, truncation='longest_first')  # 分词处理，使用最长优先方式截断
        input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']  # 获取输入ID和注意力掩码
        input_ids, attention_mask = torch.tensor(input_ids), torch.tensor(attention_mask)  # 转换为PyTorch张量
        try:
            token_type_ids = tokenized['token_type_ids']  # 获取标记类型ID
            token_type_ids = torch.tensor(token_type_ids)  # 转换为PyTorch张量
        except:
            token_type_ids = input_ids
        input_ids_list.append(input_ids)  # 将输入ID添加到列表中
        attention_mask_list.append(attention_mask)  # 将注意力掩码添加到列表中
        token_type_ids_list.append(token_type_ids)  # 将标记类型ID添加到列表中
    
    # 保存
    input_ids_tensor = pad_sequence(input_ids_list, batch_first=True, padding_value=0)  # 对输入ID进行填充，保证向量中各序列维度的大小一样，生成张量
    attention_mask_tensor = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)  # 对注意力掩码进行填充，保证向量中各序列维度的大小一样，生成张量
    token_type_ids_tensor = pad_sequence(token_type_ids_list, batch_first=True, padding_value=0)  # 对标记类型ID进行填充，保证向量中各序列维度的大小一样，生成张量
    x_test = torch.stack([input_ids_tensor, attention_mask_tensor, token_type_ids_tensor], dim=1)  # 将输入张量堆叠为一个张量
    x_test = x_test.numpy()  # 转换为NumPy数组
    np.save(f'./models_input_files/x_test{model_index}.npy', x_test)  # 保存测试数据
    
def split_train(model_name, model_dict):
    # 处理样本内容
    model_index = model_dict[model_name]  # 获取模型索引
    train = np.load(f'./models_input_files/x_train{model_index}.npy')  # 加载训练数据
    state = np.random.get_state()  # 获取随机数状态，保证样本间的随机是可重复的
    np.random.shuffle(train)  # 随机打乱训练数据
    # 训练集:验证集 = 9 : 1
    val = train[int(train.shape[0] * 0.90):]  # 划分验证集
    train = train[:int(train.shape[0] * 0.90)]  # 划分训练集
    np.save(f'./models_input_files/x_train{model_index}.npy', train)  # 保存训练集
    np.save(f'./models_input_files/x_val{model_index}.npy', val)  # 保存验证集
    train = np.load(f'./models_input_files/y_train{model_index}.npy')  # 加载标签数据
    
    # 处理样本标签
    np.random.set_state(state)  # 恢复随机数状态，让样本标签的随机可重复
    np.random.shuffle(train)  # 随机打乱标签数据
    # 训练集:验证集 = 9 : 1
    val = train[int(train.shape[0] * 0.90):]  # 划分验证集
    train = train[:int(train.shape[0] * 0.90)]  # 划分训练集
    np.save(f'./models_input_files/y_train{model_index}.npy', train)  # 保存训练集标签
    np.save(f'./models_input_files/y_val{model_index}.npy', val)  # 保存验证集标签
    
    print('split done.')
           
if __name__ == '__main__':
    model_dict = {'xlm-roberta-base':1, 'roberta-base':2, 'bert-base-uncased':3, 
                  'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext':4, 'dmis-lab/biobert-base-cased-v1.2':5, 'marieke93/MiniLM-evidence-types':6,
                  'microsoft/MiniLM-L12-H384-uncased':7, 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext':8,'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract':9,
                  'microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract':10}
    model_name = 'roberta-base'
    get_train(model_name, model_dict)
    get_test(model_name, model_dict)
    split_train(model_name, model_dict)

```

* **get_train**函数

  该函数用于准备训练数据，函数首先根据模型名称从模型字典中获取模型索引，然后从CSV文件中读取训练数据。接着，将标题、作者和摘要拼接为训练内容。使用相应模型的Tokenizer进行分词处理，并获取输入ID、注意力掩码和标记类型ID。循环遍历训练数据，将样本内容转化为输入张量，并将其添加到对应的列表中。最后使用**pad_sequence函数**对输入ID、注意力掩码和标记类型ID进行填充，保证各序列维度的大小一致，并将它们转换为张量。将输入张量堆叠为一个张量，并将其保存为NumPy数组。将标签列表转换为NumPy数组，并保存为标签数据。

* **pad_sequence**函数

  该函数用于对一批序列进行填充，使得序列具有相同的长度。函数的返回值是一个填充后的张量，所有序列的维度将与最长序列相匹配。

* **get_test**函数

  该函数用于测试数据的准备，具体流程与get_train类似

* **split_train**函数

  该函数的目的是将训练数据和标签数据划分为训练集和验证集（9：1），并将它们保存到适当的文件中，以便在模型训练时使用。使用`np.random.shuffle`函数随机打乱训练数据，并使用`np.random.get_state`和`np.random.set_state`函数获取和恢复随机数状态，以确保样本间的随机是可重复的。

**2.模型训练**

```python
# 超参数类 - 可修改的所有超参数都在这里~
class opt:
    seed               = 42 # 随机种子
    batch_size         = 16 # 批处理大小
    set_epoch          = 5  # 训练轮数 
    early_stop         = 5  # 提前停止epoch数
    learning_rate      = 1e-5 # 学习率
    weight_decay       = 2e-6 # 权重衰减,L2正则化
    device             = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 选择设备,GPU或CPU
    gpu_num            = 1 # GPU个数
    use_BCE            = False # 是否使用BCE损失函数
    models             = ['xlm-roberta-base', 'roberta-base', 'bert-base-uncased',  
                          'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 'dmis-lab/biobert-base-cased-v1.2', 'marieke93/MiniLM-evidence-types',  
                          'microsoft/MiniLM-L12-H384-uncased','cambridgeltl/SapBERT-from-PubMedBERT-fulltext', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                          'microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract'] # 模型名称列表
    model_index        = 2 # 根据上面选择使用的模型，这里填对应的模型索引
    model_name         = models[model_index-1] # 使用的模型名称
    continue_train     = False # 是否继续训练
    show_val           = False # 是否显示验证过程

# 定义模型
class MODEL(nn.Module):
    def __init__(self, model_index):
        super(MODEL, self).__init__()
        # 若是第一次下载权重，则下载至同级目录的./premodels/内，以防占主目录的存储空间
        self.model = AutoModel.from_pretrained(opt.models[model_index-1], cache_dir='./premodels/'+opt.models[model_index-1]+'_saved', from_tf=False) # 加载预训练语言模型
        # 加载模型配置，可以直接获得模型最后一层的维度，而不需要手动修改
        config = AutoConfig.from_pretrained(opt.models[model_index-1], cache_dir='./premodels/'+opt.models[model_index-1]+'_saved') # 获取配置
        last_dim = config.hidden_size # 最后一层的维度
        if opt.use_BCE:out_size = 1 # 损失函数如果使用BCE,则输出大小为1
        else          :out_size = 2 # 否则则使用CE,输出大小为2
        feature_size = 128 # 设置特征的维度大小
        self.fc1 = nn.Linear(last_dim, feature_size) # 全连接层1
        self.fc2 = nn.Linear(last_dim, feature_size) # 全连接层2
        self.classifier = nn.Linear(feature_size, out_size) # 分类器
        self.dropout = nn.Dropout(0.3) # Dropout层

            
    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x[:,0],x[:,1],x[:,2] # 获取输入
        x = self.model(input_ids, attention_mask) # 通过模型
        
        all_token     = x[0] # 全部序列分词的表征向量
        pooled_output = x[1] # [CLS]的表征向量+一个全连接层+Tanh激活函数

        feature1 = all_token.mean(dim=1) # 对全部序列分词的表征向量取均值
        feature1 = self.fc1(feature1)    # 再输入进全连接层，得到feature1
        feature2 = pooled_output      # [CLS]的表征向量+一个全连接层+Tanh激活函数
        feature2 = self.fc2(feature2) # 再输入进全连接层，得到feature2
        feature  = 0.5*feature1 + 0.5*feature2 # 加权融合特征
        feature  = self.dropout(feature) # Dropout

        x  = self.classifier(feature) # 分类
        return x

# 数据加载
def load_data():
    train_data_path     = f'models_input_files/x_train{model_index}.npy' # 训练集输入路径
    train_label_path    = f'models_input_files/y_train{model_index}.npy' # 训练集标签路径
    val_data_path       = f'models_input_files/x_val{model_index}.npy'   # 验证集输入路径
    val_label_path      = f'models_input_files/y_val{model_index}.npy'   # 验证集标签路径
    test_data_path      = f'models_input_files/x_test{model_index}.npy'  # 测试集输入路径
    
    train_data          = torch.tensor(np.load(train_data_path  , allow_pickle=True).tolist()) # 载入训练集数据
    train_label         = torch.tensor(np.load(train_label_path  , allow_pickle=True).tolist()).long() # 载入训练集标签  
    val_data            = torch.tensor(np.load(val_data_path  , allow_pickle=True).tolist()) # 载入验证集数据
    val_label           = torch.tensor(np.load(val_label_path  , allow_pickle=True).tolist()).long() # 载入验证集标签
    test_data           = torch.tensor(np.load(test_data_path  , allow_pickle=True).tolist()) # 载入测试集数据

    train_dataset       = torch.utils.data.TensorDataset(train_data  , train_label) # 构造训练集Dataset
    val_dataset         = torch.utils.data.TensorDataset(val_data  , val_label) # 构造验证集Dataset
    test_dataset        = torch.utils.data.TensorDataset(test_data) # 构造测试集Dataset
    
    return train_dataset, val_dataset, test_dataset # 返回数据集

# 模型预训练
def model_pretrain(model_index, train_loader, val_loader):
    # 超参数设置
    set_epoch          = opt.set_epoch  # 训练轮数
    early_stop         = opt.early_stop # 提前停止epoch数
    learning_rate      = opt.learning_rate # 学习率
    weight_decay       = opt.weight_decay  # 权重衰减
    device             = opt.device  # 设备 
    gpu_num            = opt.gpu_num # GPU个数
    continue_train     = opt.continue_train # 是否继续训练
    model_save_dir     = 'checkpoints' # 模型保存路径
    
    # 是否要继续训练，若是，则加载模型进行训练；若否，则跳过训练，直接对测试集进行推理
    if not continue_train:
        # 判断最佳模型是否已经存在,若存在则直接读取,若不存在则进行训练
        if os.path.exists(f'checkpoints/best_model{model_index}.pth'): 
            best_model = MODEL(model_index)
            best_model.load_state_dict(torch.load(f'checkpoints/best_model{model_index}.pth')) # 加载模型
            return best_model
        else:
            pass
            

    # 模型初始化
    model = MODEL(model_index).to(device) 
    if continue_train:
        model.load_state_dict(torch.load(f'checkpoints/best_model{model_index}.pth')) # 继续训练加载模型

    # 优化器初始化
    if device    != 'cpu' and gpu_num > 1:  # 多张显卡
        optimizer = torch.optim.AdamW(model.module.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = torch.nn.DataParallel(optimizer, device_ids=list(range(gpu_num))) # 多GPU
    else: # 单张显卡
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # 单GPU
    
    # 损失函数初始化
    if opt.use_BCE:
        loss_func = nn.BCEWithLogitsLoss() # BCE损失
    else:
        loss_func = nn.CrossEntropyLoss() # 交叉熵损失（CE）

    # 模型训练
    best_epoch         = 0 # 最佳epoch
    best_train_loss    = 100000 # 最佳训练损失
    train_acc_list     = [] # 训练准确率列表
    train_loss_list    = [] # 训练损失列表
    val_acc_list       = [] # 验证准确率列表 
    val_loss_list      = [] # 验证损失列表
    start_time         = time.time() # 训练开始时间

    for epoch in range(set_epoch): # 轮数
        model.train() # 模型切换到训练模式
        train_loss = 0 # 训练损失
        train_acc = 0 # 训练准确率
        for x, y in tqdm(train_loader): # 遍历训练集
            # 训练前先将数据放到GPU上
            x        = x.to(device)
            y        = y.to(device)
            outputs  = model(x) # 前向传播
            
            if opt.use_BCE: # BCE损失
                loss = loss_func(outputs, y.float().unsqueeze(1)) 
            else: # 交叉熵损失
                loss = loss_func(outputs, y)
            train_loss += loss.item() # 累加训练损失
            optimizer.zero_grad() # 清空梯度
            loss.backward() # 反向传播

            if device != 'cpu' and gpu_num > 1: # 多GPU更新
                optimizer.module.step()  
            else:
                optimizer.step() # 单GPU更新
            
            if not opt.use_BCE: # 非BCE损失
                _, predicted = torch.max(outputs.data, 1) # 预测结果
            else:
                predicted = (outputs > 0.5).int() # 预测结果
                predicted = predicted.squeeze(1) 
            train_acc   += (predicted == y).sum().item() # 计算训练准确率
            
        average_mode = 'binary'
        train_f1     = metrics.f1_score(y.cpu(), predicted.cpu(), average=average_mode) # 计算F1
        train_pre    = metrics.precision_score(y.cpu(), predicted.cpu(), average=average_mode) # 计算精确率
        train_recall = metrics.recall_score(y.cpu(), predicted.cpu(), average=average_mode) # 计算召回率


        train_loss /= len(train_loader) # 平均所有步数的训练损失作为一个epoch的训练损失
        train_acc  /= len(train_loader.dataset) # 平均所有步数训练准确率作为一个epoch的准确率
        train_acc_list.append(train_acc)   # 添加训练准确率
        train_loss_list.append(train_loss) # 添加训练损失

        print('-'*50)
        print('Epoch [{}/{}]\n Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch + 1, set_epoch, train_loss, train_acc))
        print('Train-f1: {:.4f}, Train-precision: {:.4f} Train-recall: {:.4f}'.format(train_f1, train_pre, train_recall))

        if opt.show_val: # 显示验证过程
            # 验证
            model.eval() # 模型切换到评估模式
            val_loss = 0 # 验证损失
            val_acc = 0 # 验证准确率
    
            for x, y in tqdm(val_loader): # 遍历验证集
                # 训练前先将数据放到GPU上
                x = x.to(device) 
                y = y.to(device)
                outputs = model(x) # 前向传播
                if opt.use_BCE: # BCE损失
                    loss = loss_func(outputs, y.float().unsqueeze(1))
                else: # 交叉熵损失  
                    loss = loss_func(outputs, y)
                
                val_loss += loss.item() # 累加验证损失
                if not opt.use_BCE: # 非BCE损失
                    _, predicted = torch.max(outputs.data, 1) 
                else:
                    predicted = (outputs > 0.5).int() # 预测结果
                    predicted = predicted.squeeze(1)
                val_acc += (predicted == y).sum().item() # 计算验证准确率
    
            val_f1     = metrics.f1_score(y.cpu(), predicted.cpu(), average=average_mode) # 计算F1
            val_pre    = metrics.precision_score(y.cpu(), predicted.cpu(), average=average_mode) # 计算精确率
            val_recall = metrics.recall_score(y.cpu(), predicted.cpu(), average=average_mode) # 计算召回率
    
            val_loss /= len(val_loader) # 平均验证损失
            val_acc /= len(val_loader.dataset) # 平均验证准确率
            val_acc_list.append(val_acc)   # 添加验证准确率
            val_loss_list.append(val_loss) # 添加验证损失
            print('\nVal Loss: {:.4f}, Val Acc: {:.4f}'.format(val_loss, val_acc))
            print('Val-f1: {:.4f}, Val-precision: {:.4f} Val-recall: {:.4f}'.format(val_f1, val_pre, val_recall))

        if train_loss < best_train_loss: # 更新最佳训练损失
            best_train_loss = train_loss
            best_epoch = epoch + 1
            if device == 'cuda' and gpu_num > 1: # 多GPU保存模型
                torch.save(model.module.state_dict(), f'{model_save_dir}/best_model{model_index}.pth')
            else:
                torch.save(model.state_dict(), f'{model_save_dir}/best_model{model_index}.pth') # 单GPU保存模型
        
        # 提前停止判断
        if epoch+1 - best_epoch == early_stop:  
            print(f'{early_stop} epochs later, the loss of the validation set no longer continues to decrease, so the training is stopped early.')
            end_time = time.time()
            print(f'Total time is {end_time - start_time}s.')
            break

    best_model = MODEL(model_index) # 初始化最佳模型
    best_model.load_state_dict(torch.load(f'checkpoints/best_model{model_index}.pth')) # 加载模型参数
    return best_model # 返回最佳模型

# 模型推理
def model_predict(model, model_index, test_loader):
    device = 'cuda'
    model.to(device) # 模型到GPU
    model.eval()  # 切换到评估模式

    test_outputs = None
    with torch.no_grad():  # 禁用梯度计算
        for i, data in enumerate(tqdm(test_loader)):
            data = data[0].to(device) # 测试数据到GPU
            outputs = model(data) # 前向传播
            if i == 0: 
                test_outputs = outputs # 第一个batch直接赋值
            else:
                test_outputs = torch.cat([test_outputs, outputs], dim=0) # 其余batch拼接

            del data, outputs  # 释放不再需要的Tensor

    # 保存预测结果    
    if not opt.use_BCE: 
        test_outputs = torch.softmax(test_outputs, dim=1) # 转换为概率
    torch.save(test_outputs, f'./models_prediction/{model_index}_prob.pth') # 保存概率

def run(model_index):
    # 固定随机种子
    seed = opt.seed  
    torch.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    train_dataset, val_dataset, test_dataset = load_data() # 加载数据集
    # 打印数据集信息
    print('-数据集信息:')
    print(f'-训练集样本数:{len(train_dataset)},测试集样本数:{len(test_dataset)}')
    train_labels = len(set(train_dataset.tensors[1].numpy()))
    # 查看训练样本类别均衡状况
    print(f'-训练集的标签种类个数为:{train_labels}') 
    numbers = [0] * train_labels
    for i in train_dataset.tensors[1].numpy():
        numbers[i] += 1
    print(f'-训练集各种类样本的个数:')
    for i in range(train_labels):
        print(f'-{i}的样本个数为:{numbers[i]}')

    batch_size   = opt.batch_size # 批处理大小
    # 构建DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader   = torch.utils.data.DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

    best_model   = model_pretrain(model_index, train_loader, val_loader)

    # 使用验证集评估模型
    model_predict(best_model, model_index, test_loader) # 模型推理

if __name__ == '__main__':
    model_index = opt.model_index # 获取模型索引
    run(model_index) # 运行程序
```

* **Model**

  该类用于定义模型。其中包含初始化函数和前向传播函数，

  在__init__函数中，首先根据模型索引加载预训练语言模型，并从加载的配置中获取最后一层输出的维度。根据是否使用二分类损失函数，设置输出大小为1或2。然后定义两个全连接层和一个分类器，并设置特征的维度大小为128。最后添加一个Dropout层（防止过拟合）。

  **前向传播函数**，用于对输入数据进行前向计算并生成预测结果。

  ```python
  	 def forward(self, x):
          input_ids, attention_mask, token_type_ids = x[:,0],x[:,1],x[:,2] # 获取输入
          x = self.model(input_ids, attention_mask) # 通过模型
          
          all_token     = x[0] # 全部序列分词的表征向量
          pooled_output = x[1] # [CLS]的表征向量+一个全连接层+Tanh激活函数
  
          feature1 = all_token.mean(dim=1) # 对全部序列分词的表征向量取均值
          feature1 = self.fc1(feature1)    # 再输入进全连接层，得到feature1
          feature2 = pooled_output      # [CLS]的表征向量+一个全连接层+Tanh激活函数
          feature2 = self.fc2(feature2) # 再输入进全连接层，得到feature2
          feature  = 0.5*feature1 + 0.5*feature2 # 加权融合特征
          feature  = self.dropout(feature) # Dropout
  
          x  = self.classifier(feature) # 分类
          return x
  ```

  **函数具体流程如下：**

  在函数的开头，通过索引`x[:,0], x[:,1], x[:,2]`，将输入`x`中的三个特征分别赋值给变量`input_ids`、`attention_mask`和`token_type_ids`。这些特征通常用于处理自然语言处理任务中的BERT模型。

  然后，将这些特征作为输入传递给模型`self.model`。根据代码的逻辑，模型会返回两个输出，即`all_token`和`pooled_output`。

  其中，`all_token`表示所有序列分词的表征向量，它的维度是`(batch_size, sequence_length, hidden_size)`，其中`hidden_size`表示BERT模型的隐藏层的维度。
  `pooled_output`表示对应于模型输出的第一个特征（通常是[CLS]标记）的表征向量，它的维度是`(batch_size, hidden_size)`。

  接下来，对于`all_token`，通过`mean(dim=1)`对序列的维度求均值，得到`feature1`。然后，将`feature1`输入全连接层`self.fc1`，得到`feature1`的新表示。

  对于`pooled_output`，直接将其作为输入传递给全连接层`self.fc2`，得到`feature2`的新表示。

  然后，对`feature1`和`feature2`进行加权融合，即将两者按照0.5的权重相加得到`feature`。

  接下来，对`feature`应用Dropout操作，通过`self.dropout`函数对其进行随机失活，以减少过拟合。

  最后，将`feature`作为输入传递给分类器`self.classifier`，并返回其输出`x`，即预测结果。

* **融合特征**的目的是增强模型的表征能力和预测性能。在这段代码中，特征融合是通过加权平均的方式进行的，其中`feature1`和`feature2`分别代表两种不同的特征表征。特征融合可以有助于提取更全面、更丰富的信息，并在不同方面的特征上取得平衡。

  `feature1`来自BERT模型中所有序列分词的均值表征向量，可以表示整体的句子或文本信息，而`feature2`则强调了对应于[CLS]标记的表征向量，其更加关注整体句子的分类特征。通过融合这两种不同层次的信息，可以在全局和局部级别上对数据进行建模。

* **model_pretrain**模型预训练

  该函数**对模型、优化器、损失函数进行初始化**。在每个epoch结束后，计算该epoch的平均训练损失和训练准确率，并将它们分别添加到训练损失列表(train_loss_list)和训练准确率列表(train_acc_list)中。最后打印出该epoch的训练损失、训练准确率、F1值、精确率和召回率。

  对于每个验证样本，将数据移动到GPU上并进行前向传播。根据是否使用BCE损失函数，计算损失值。对于非BCE损失函数，根据模型输出的类别标签进行预测，并计算验证准确率。

* **model_predict**模型预测

  该函数的主要目的是**对测试集进行预测，生成预测结果**，并将概率结果保存。

  函数遍历测试集数据加载器，并将测试数据移动到GPU上。通过模型的前向传播获得预测输出(outputs)。在第一个batch时，将预测输出直接赋值给test_outputs；对于其余的batch，通过使用`torch.cat()`函数将预测输出拼接到test_outputs中，随后释放不再需要的Tensor，以节省内存。

  最后，如果不使用BCE损失函数，通过对test_outputs使用`torch.softmax()`函数将输出转换为概率形式，并将概率结果保存。

**3.评估模型**

```python
# 推理
def inference(model_indexs, use_BCE):
    device = 'cuda'  # 设备选择为cuda
    for model_index in model_indexs:
        # 加载模型
        model = MODEL(model_index).to(device)  # 创建MODEL类的实例，并将模型移至设备(device)
        model.load_state_dict(torch.load(f'checkpoints/best_model{model_index}.pth'))  # 加载模型的权重参数
        model.eval()  # 切换到评估模式
        # 加载val数据
        val_data_path = f'models_input_files/x_val{model_index}.npy'  # val数据的路径
        val_data = torch.tensor(np.load(val_data_path, allow_pickle=True).tolist())  # 加载val数据，并转换为Tensor格式
        val_dataset = torch.utils.data.TensorDataset(val_data)  # 创建val数据集
        val_loader  = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)  # 创建val数据的数据加载器
        val_outputs = None  # 初始化val_outputs变量
        with torch.no_grad():  # 禁用梯度计算
            for i, data in enumerate(tqdm(val_loader)):  # 遍历val_loader，显示进度条
                data = data[0].to(device)  # 将数据移至GPU
                outputs = model(data)  # 模型推理，获取输出
                if i == 0:
                    val_outputs = outputs  # 若为第一次迭代，直接赋值给val_outputs
                else:
                    val_outputs = torch.cat([val_outputs, outputs], dim=0)  # 否则在dim=0上拼接val_outputs和outputs

                del data, outputs  # 释放不再需要的Tensor对象

        # 输出预测概率
        if not use_BCE:
            val_outputs = torch.softmax(val_outputs, dim=1)  # 对val_outputs进行softmax操作
        torch.save(val_outputs, f'evaluate_prediction/{model_index}_prob.pth')  # 保存预测概率结果


def run(model_indexs, use_BCE):
    # 读取所有的model_prob.pth，并全加在一起
    avg_pred = None  # 初始化avg_pred变量
    for i in model_indexs:
        pred = torch.load(f'evaluate_prediction/{i}_prob.pth').data  # 加载预测概率结果
        if use_BCE:
            # 选取大于0.5的作为预测结果
            pred = (pred > 0.5).int()  # 将大于0.5的值转换为整数（0或1）
            pred = pred.reshape(-1)  # 将预测结果进行形状重塑
        else:
            # 选取最大的概率作为预测结果
            pred = torch.argmax(pred, dim=1)  # 获取最大概率的索引作为预测结果
        pred = pred.cpu().numpy()  # 将预测结果转移到CPU上，并转换为NumPy数组

        # to_evaluate
        # 读取真实标签
        val_label_path = f'models_input_files/y_val{i}.npy'  # 真实标签的路径
        y_true = np.load(val_label_path)  # 加载真实标签
        # 分类报告
        print(f'model_index = {i}:')
        print(classification_report(y_true, pred, digits=4))  # 打印分类报告，包括精确度、召回率等指标

        zero_acc = 0; one_acc = 0 # 初始化0类和1类的准确率
        zero_num = 0; one_num= 0  # 初始化0类和1类的样本数量
        for i in range(pred.shape[0]):
            if y_true[i] == 0:
                zero_num += 1  # 统计0类的样本数量
            elif y_true[i] == 1:
                one_num += 1  # 统计1类的样本数量
            if pred[i] == y_true[i]:
                if pred[i] == 0:
                    zero_acc += 1  # 统计0类的正确预测数量
                elif pred[i] == 1:
                    one_acc += 1  # 统计1类的正确预测数量

        zero = np.sum(pred == 0) / pred.shape[0]  # 计算预测为0类的样本占比
        zero_acc /= zero_num  # 计算0类的正确率
        print(f'预测0类占比：{zero}  0类正确率：{zero_acc}')
        one = np.sum(pred == 1) / pred.shape[0]  # 计算预测为1类的样本占比
        one_acc /= one_num  # 计算1类的正确率
        print(f'预测1类占比：{one}  1类正确率：{one_acc}')
        print('-' * 80)


if __name__ == '__main__':
    use_BCE = False  # 是否使用BCE损失函数的标志，这里我只用交叉熵CE，所以是False
    inference([2], use_BCE=use_BCE)  # 进行推理，传入模型索引和use_BCE标志
    model_indexs = [2]  # 模型索引列表
    run(model_indexs, use_BCE=use_BCE)  # 进行运行，传入模型索引和use_BCE标志
```

* 这是一个用于推理和评估模型性能的代码示例。

  首先，使用`inference`函数进行推理操作，模型索引取为2（roberta模型）。在推理过程中，加载已训练好的模型权重参数，加载验证数据，对验证数据进行推理，并保存预测概率结果。

  接下来，使用`run`函数对模型性能进行评估。首先，加载预测概率结果，并根据是否使用BCE损失函数对预测结果进行处理。然后，读取真实标签，计算分类报告并打印出相关指标。接着，计算每个类别的准确率。最后，打印预测为0类和1类的样本占比以及对应的正确率。

  最后，在主程序中调用`inference`函数和`run`函数，传入正确的模型索引列表和使用BCE损失函数的标志。

  该代码的主要目的是进行模型推理和评估，输出模型的性能指标，如精确率、召回率等，并打印预测为每个类别的样本占比和对应的正确率。

* `inference`函数：这个函数用于**进行模型推理**。它根据传入的模型索引，加载对应的模型权重参数，并进行推理操作。推理过程中，使用验证数据加载器遍历验证数据，将数据传入模型中进行前向传播，获得预测输出。最后，将预测概率结果保存在特定路径下。

* `run`函数：这个函数用于**评估模型性能**。它首先加载之前保存的预测概率结果，根据是否使用BCE损失函数对预测结果进行处理（选取概率大于0.5作为预测结果或选取概率最大值的索引作为预测结果）。然后，读取对应的真实标签，并计算分类报告，输出相关的指标。接着，计算预测为0类和1类的样本占比以及对应的正确率，并打印出来。

**4.测试集推理**

```python
import torch
import pandas as pd
import warnings # 过滤警告
warnings.filterwarnings('ignore')

def run(model_indexs, use_BCE):
    # 记录模型数量
    model_num = len(model_indexs)
    # 读取所有的model_prob.pth，并全加在一起
    for i in model_indexs:
        # 加载模型在训练完成后对测试集推理所得的预测文件
        pred = torch.load(f'./models_prediction/{i}_prob.pth', map_location='cpu').data
        # 这里的操作是将每个模型对测试集推理的概率全加在一起
        if i == model_indexs[0]:
            avg_pred = pred
        else:
            avg_pred += pred
        
    # 取平均
    avg_pred /= model_num # 使用全加在一起的预测概率除以模型数量

    if use_BCE:
        # 选取概率大于0.5的作为预测结果
        pred = (avg_pred > 0.5).int()
        pred = pred.reshape(-1)
    else:
        # 后处理 - 根据标签数目的反馈，对预测阈值进行调整
        pred[:, 0][pred[:, 0]>0.001] = 1
        pred[:, 1][pred[:, 1]>0.999] = 1.2
        # 选取最大的概率作为预测结果
        pred = torch.argmax(avg_pred, dim=1)
    pred = pred.cpu().numpy()

    # to_submit
    # 读取test.csv文件
    test = pd.read_csv('./dataset/testB_submit_exsample.csv')

    # 开始写入预测结果
    for i in range(len(pred)):
        test['label'][i] = pred[i]

    print(test['label'].value_counts())
    # 保存为提交文件
    test.to_csv(f'submit.csv',index=False)

if __name__ == '__main__':
    run([2], use_BCE=False)
    # run([1,2,3,4,5,6,7,8,9,10], use_BCE=False)

```

* 该函数用于处理模型预测的结果，并将结果保存为提交文件

* 函数主要执行以下过程

  首先，根据传入的模型索引列表，遍历每个模型的预测文件。其中，预测文件的命名规则为`model_index_prob.pth`，表示对测试数据进行推理后得到的预测概率。使用`torch.load`函数加载预测文件并将其保存为`pred`变量。

  接下来，对于第一个模型，初始化`avg_pred`变量为`pred`；对于其他模型，将其与`avg_pred`相加。最终，将`avg_pred`除以模型数量，以求得预测概率的平均值。

  根据是否使用BCE损失函数（即`use_BCE`变量），进行不同的后处理操作。如果使用BCE损失函数，则将概率大于0.5的设置为1，否则将概率最大的类别设置为预测结果。

  接下来，将处理后的预测结果（`pred`）转移到CPU，并保存为Numpy数组。

  然后，读取测试数据的提交样例文件`testB_submit_exsample.csv`。使用`pd.read_csv`函数读取该文件，保存为`test`变量。

  接下来，遍历每个预测结果，将结果写入到`test`的`label`列中。

  最后，统计预测结果中每个类别的样本数量，并打印出来；将更新后的`test`数据保存为提交文件`submit.csv`。

  在主程序中，调用`run`函数，传入模型索引列表`[2]`和`use_BCE`标志设为`False`。这将运行`run`函数来处理预测结果，并生成提交文件。还可以根据需要传入不同的模型索引列表和`use_BCE`标志来进行实验和生成提交文件。

```python
# 后处理 - 根据标签数目的反馈，对预测阈值进行调整
	pred[:, 0][pred[:, 0]>0.001] = 1
	pred[:, 1][pred[:, 1]>0.999] = 1.2
```

* 其中这部分代码根据标签数目的反馈，对预测阈值进行调整。在训练模型时，如果发现某个标签的样本数量过少或过多，可能会需要调整预测阈值，以使模型在预测这类样本时更加准确。

  对于第一类别（索引为0），将大于0.001的概率设置为1；对于第二类别（索引为1），将大于0.999的概率设置为1.2。

  **注意：**在这段代码中，将大于0.999的概率设置为1.2，为了对预测结果的置信度进行加强。将概率设置为1.2可以表示模型对该类别的预测结果更加自信和确定。通常情况下，我们认为概率介于0和1之间，表示对该类别的置信度。

  作者回答：方便后面取argmax。但其实后处理是怎么来都可以的，关键是根据分布调整模型对不同标签的预测数量，所以在数值上可以随意设置测试。前提是训练的模型泛化能力足够强，那么当通过后处理的方法把不同标签的预测数量控制到与测试集分布一致的时候，拿1其实很容易的。

* `pred = torch.argmax(avg_pred, dim=1)`是将预测结果的概率转换为具体的类别标签。

  `torch.argmax`函数的作用是在指定的维度上返回张量中最大值的索引。在这段代码中，`avg_pred`是一个2维的张量，每一行表示一个样本的预测概率，每一列表示对应类别的预测概率。`dim=1`表示按行的维度进行计算。

## 3.总结

* 用传统的机器学习进行分类时，使用了LogisticRegression算法得分0.67112
* 用Bert模型预训练+微调，得分0.999，后调整参数达到了0.9995
* 更换Roberta模型并进行特征融合+后处理，成功拿到1
* 在这次夏令营NLP方向的学习中，我算是正式解了解决二分类问题的基本流程，并且学会了比较主流的预训练+微调的模式，也对transformer有了认识，感谢datawhale给我们提供的详细的教程，希望有一天我也可以成为其中的一份子，为开源社区做出自己的贡献。


>本文引用
>>"净好大佬的笔记" : https://tvq27xqm30o.feishu.cn/docx/V2gfdvAMGoxgozx0OkMchp4znnf  
>"AI夏令营-NLP实践教程" : https://datawhaler.feishu.cn/docx/HGiNdHedwoAtcVx0kkScwaI3nKc	   
>"深度学习topline": https://tvq27xqm30o.feishu.cn/docx/U1fzdqdE0o6SWnxixyrc3gnLnJg	



