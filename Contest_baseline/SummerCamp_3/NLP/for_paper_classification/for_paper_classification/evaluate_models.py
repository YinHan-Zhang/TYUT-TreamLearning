import torch  # 导入torch库，用于深度学习任务
import pandas as pd  # 导入pandas库，用于处理数据表格
from models_training import MODEL  # 从本地文件models_training.py中导入MODEL类
import torch  # 导入torch库，用于深度学习任务
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
from sklearn.metrics import classification_report  # 从sklearn.metrics模块导入classification_report函数，用于输出分类报告，看各标签的F1值
import numpy as np  # 导入numpy库，用于科学计算

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