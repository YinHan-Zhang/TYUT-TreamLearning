# 导入需要的库
import numpy as np  # 导入numpy库，用于科学计算
import torch  # 导入torch库，用于深度学习任务
import torch.nn as nn  # 导入torch.nn模块，用于神经网络相关操作
from sklearn import metrics  # 导入sklearn库，用于评估指标计算
import os    # 导入os库，用于操作系统相关功能
import time  # 导入time库，用于时间相关操作
from transformers import AutoModel, AutoConfig  # 导入AutoModel和AutoConfig类，用于加载预训练模型
from tqdm import tqdm  # 导入tqdm库，用于显示进度条

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
        # 判断最佳模型是否已经存在,若存在则直接读取,若不存在则进行预训练
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