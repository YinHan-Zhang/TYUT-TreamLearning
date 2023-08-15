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

* **PyTorch 是一个基于 Python 的深度学习框架**，它提供了用于构建和训练神经网络的高级接口和工具。PyTorch 被广泛应用于机器学习和人工智能领域，特别是在计算机视觉、自然语言处理等任务中。

  PyTorch 的核心功能包括：

  1. 张量操作：PyTorch 提供了强大的张量操作功能，类似于 NumPy，用于在 CPU 或 GPU 上执行数值计算。
  2. 自动微分：PyTorch 提供自动微分（Autograd）功能，它能够根据定义的计算图自动计算张量操作的导数，方便进行反向传播和梯度下降等优化算法。
  3. 神经网络模块：PyTorch 提供了丰富的神经网络模块和函数，可用于构建各种类型的神经网络模型，包括卷积神经网络（CNN）、递归神经网络（RNN）、生成对抗网络（GAN）等。
  4. 模型训练与部署：PyTorch 提供了易于使用的工具和函数，用于定义训练循环、计算损失函数、应用优化算法等。此外，PyTorch 还支持将训练好的模型导出为可部署的格式，方便在生产环境中使用。
  5. 支持多种硬件：PyTorch 支持在多种硬件平台上运行，包括 CPU、GPU 和专用加速器（如 NVIDIA 的 CUDA 平台）等。

  PyTorch 提供了简单易用的 Python API，使得构建和训练深度学习模型更加直观和灵活。它也是一个开源项目，并有活跃的社区支持，提供了广泛的教程、文档和示例，方便用户学习和使用

* **BERT（Bidirectional Encoder Representations from Transformers）**是一种基于Transformer模型的预训练语言表示模型。由Google在2018年发布，它在自然语言处理（NLP）领域取得了重要的突破。

  BERT模型的核心思想是**通过在大规模未标记的文本数据上进行无监督的预训练，学习得到丰富的语言表示**。BERT采用**双向（bidirectional）的Transformer模型**结构，使得模型可以同时考虑到一个词的上下文信息。

  具体来说，BERT模型使用了两个主要任务进行预训练：掩码语言模型（Masked Language Model，MLM）和下一句预测（Next Sentence Prediction，NSP）。

  1. **掩码语言模型（MLM）**：在预训练阶段，BERT模型会将输入文本中的一些词随机选择并替换为掩码符号（例如，将"我爱吃苹果"替换为"我爱吃[掩码]"），然后让模型预测被掩码的词。这样的预训练方式可以使得模型能够有效地学习到词语之间的上下文关系。
  2. **下一句预测（NSP）**：在预训练阶段，BERT模型也会使用一对连续的句子作为输入，并让模型预测这两个句子是否是连续的。这样的任务可以帮助模型学习到句子级别的语义关系。

  通过这样的预训练方式，BERT模型可以学习到丰富的语言表示，包括词语级别和句子级别的语义信息。在实际应用中，可以使用预训练好的BERT模型作为特征提取器，或者使用其作为初始模型，在特定任务上进行微调。

  BERT模型在多项自然语言处理任务上取得了很好的效果，包括问答系统、文本分类、命名实体识别等。它的发布开启了预训练语言模型在NLP领域的新篇章，并对后续的研究和应用产生了深远影响。

* **Transformer是一种用于自然语言处理和机器翻译等任务的深度学习模型**，通过自我注意力机制（Self-Attention）实现了高效的序列建模和特征提取。

  Transformer模型主要由两个关键组件构成：

  1. **自我注意力机制（Self-Attention）**：自我注意力机制用于计算输入序列中各个位置的重要性权重，将每个位置与其他位置的信息进行交互和整合。它通过计算每个位置与其它位置的相关度得到一个权重分布，用于加权求和得到每个位置的上下文表示。
  2. **前馈神经网络（Feed-Forward Neural Network）**：前馈神经网络是Transformer模型中的另一个组件，用于在每个位置独立地对自我注意力机制的输出进行进一步的非线性变换和特征提取。

  Transformer模型通过自我注意力机制实现了对输入序列内部的位置和位置之间的关系进行建模，使得模型能够有效地处理长序列和捕获复杂的依赖关系，为自然语言处理任务带来了重要的突破。



### 2.3进阶实战-大模型法

* todo

## 3.总结

* todo



[1]: https://tvq27xqm30o.feishu.cn/docx/V2gfdvAMGoxgozx0OkMchp4znnf	"净好大佬的笔记"
[2]: https://datawhaler.feishu.cn/docx/HGiNdHedwoAtcVx0kkScwaI3nKc	"AI夏令营-NLP实践教程"





