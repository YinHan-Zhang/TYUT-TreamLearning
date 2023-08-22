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
