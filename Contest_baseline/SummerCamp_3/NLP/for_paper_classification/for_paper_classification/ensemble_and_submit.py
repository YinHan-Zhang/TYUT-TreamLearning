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
