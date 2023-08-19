MLbaseline解读


```Python
'''
datawhale官方版本
'''
# 1. 导入需要用到的相关库
# 导入 pandas 库，用于数据处理和分析
import pandas as pd
# 导入 numpy 库，用于科学计算和多维数组操作
import numpy as np
# 从 sklearn.tree 模块中导入 DecisionTreeClassifier 类
# DecisionTreeClassifier 用于构建决策树分类模型
from sklearn.tree import DecisionTreeClassifier


# 2. 读取训练集和测试集
# 使用 read_csv() 函数从文件中读取训练集数据，文件名为 'train.csv'
train_data = pd.read_csv('用户新增预测挑战赛公开数据/train.csv')
# 使用 read_csv() 函数从文件中读取测试集数据，文件名为 'test.csv'
test_data = pd.read_csv('用户新增预测挑战赛公开数据/test.csv')


# 3. 将 'udmap' 列进行 One-Hot 编码 
# 数据样例：
#                    udmap  key1  key2  key3  key4  key5  key6  key7  key8  key9
# 0           {'key1': 2}     2     0     0     0     0     0     0     0     0
# 1           {'key2': 1}     0     1     0     0     0     0     0     0     0
# 2  {'key1': 3, 'key2': 2}   3     2     0     0     0     0     0     0     0

# 在 python 中, 形如 {'key1': 3, 'key2': 2} 格式的为字典类型对象, 通过key-value键值对的方式存储
# 而在本数据集中, udmap实际是以字符的形式存储, 所以处理时需要先用eval 函数将'udmap' 解析为字典

# 具体实现代码：
# 定义函数 udmap_onethot，用于将 'udmap' 列进行 One-Hot 编码
def udmap_onethot(d):
    v = np.zeros(9)  # 创建一个长度为 9 的零数组
    if d == 'unknown':  # 如果 'udmap' 的值是 'unknown'
        return v  # 返回零数组
    d = eval(d)  # 将 'udmap' 的值解析为一个字典
    for i in range(1, 10):  # 遍历 'key1' 到 'key9', 注意, 这里不包括10本身
        if 'key' + str(i) in d:  # 如果当前键存在于字典中
            v[i-1] = d['key' + str(i)]  # 将字典中的值存储在对应的索引位置上
            
    return v  # 返回 One-Hot 编码后的数组

# 注: 对于不理解的步骤, 可以逐行 print 内容查看
# 使用 apply() 方法将 udmap_onethot 函数应用于每个样本的 'udmap' 列
# np.vstack() 用于将结果堆叠成一个数组
train_udmap_df = pd.DataFrame(np.vstack(train_data['udmap'].apply(udmap_onethot)))
test_udmap_df = pd.DataFrame(np.vstack(test_data['udmap'].apply(udmap_onethot)))
# 为新的特征 DataFrame 命名列名
train_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
test_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
# 将编码后的 udmap 特征与原始数据进行拼接，沿着列方向拼接
train_data = pd.concat([train_data, train_udmap_df], axis=1)
test_data = pd.concat([test_data, test_udmap_df], axis=1)


# 4. 编码 udmap 是否为空
# 使用比较运算符将每个样本的 'udmap' 列与字符串 'unknown' 进行比较，返回一个布尔值的 Series
# 使用 astype(int) 将布尔值转换为整数（0 或 1），以便进行后续的数值计算和分析
train_data['udmap_isunknown'] = (train_data['udmap'] == 'unknown').astype(int)
test_data['udmap_isunknown'] = (test_data['udmap'] == 'unknown').astype(int)


# 5. 提取 eid 的频次特征
# 使用 map() 方法将每个样本的 eid 映射到训练数据中 eid 的频次计数
# train_data['eid'].value_counts() 返回每个 eid 出现的频次计数
train_data['eid_freq'] = train_data['eid'].map(train_data['eid'].value_counts())
test_data['eid_freq'] = test_data['eid'].map(train_data['eid'].value_counts())


# 6. 提取 eid 的标签特征
# 使用 groupby() 方法按照 eid 进行分组，然后计算每个 eid 分组的目标值均值
# train_data.groupby('eid')['target'].mean() 返回每个 eid 分组的目标值均值
train_data['eid_mean'] = train_data['eid'].map(train_data.groupby('eid')['target'].mean())
test_data['eid_mean'] = test_data['eid'].map(train_data.groupby('eid')['target'].mean())


# 7. 提取时间戳
# 使用 pd.to_datetime() 函数将时间戳列转换为 datetime 类型
# 样例：1678932546000->2023-03-15 15:14:16
# 注: 需要注意时间戳的长度, 如果是13位则unit 为 毫秒, 如果是10位则为 秒, 这是转时间戳时容易踩的坑
# 具体实现代码：
train_data['common_ts'] = pd.to_datetime(train_data['common_ts'], unit='ms')
test_data['common_ts'] = pd.to_datetime(test_data['common_ts'], unit='ms')

# 使用 dt.hour 属性从 datetime 列中提取小时信息，并将提取的小时信息存储在新的列 'common_ts_hour'
train_data['common_ts_hour'] = train_data['common_ts'].dt.hour
test_data['common_ts_hour'] = test_data['common_ts'].dt.hour


# 8. 加载决策树模型进行训练(直接使用sklearn中导入的包进行模型建立)
clf = DecisionTreeClassifier()
# 使用 fit 方法训练模型
# train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1) 从训练数据集中移除列 'udmap', 'common_ts', 'uuid', 'target'
# 这些列可能是特征或标签，取决于数据集的设置
# train_data['target'] 是训练数据集中的标签列，它包含了每个样本的目标值
clf.fit(
    train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),  # 特征数据：移除指定的列作为特征
    train_data['target']  # 目标数据：将 'target' 列作为模型的目标进行训练
)


# 9. 对测试集进行预测，并保存结果到result_df中
# 创建一个DataFrame来存储预测结果，其中包括两列：'uuid' 和 'target'
# 'uuid' 列来自测试数据集中的 'uuid' 列，'target' 列将用来存储模型的预测结果
result_df = pd.DataFrame({
    'uuid': test_data['uuid'],  # 使用测试数据集中的 'uuid' 列作为 'uuid' 列的值
    'target': clf.predict(test_data.drop(['udmap', 'common_ts', 'uuid'], axis=1))  # 使用模型 clf 对测试数据集进行预测，并将预测结果存储在 'target' 列中
})


# 10. 保存结果文件到本地
# 将结果DataFrame保存为一个CSV文件，文件名为 'submit.csv'
# 参数 index=None 表示不将DataFrame的索引写入文件中
result_df.to_csv('submit.csv', index=None)
```

