sklearn 学习

## 调参

模型在未知数据上的准确率受什么因素影响？在机器学习中，我们用来衡量模型在位置数据上的准确率的指标交**泛化误差** 

泛化误差由 方差、偏差、和噪声共同决定

![1634651657236](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1634651657236.png)

* 1）模型太复杂或者太简单，都会让泛化误差高，我们追求的是位于中间的平衡点 

* 2）模型太复杂就会过拟合，模型太简单就会欠拟合 

* 3）对树模型和树的集成模型来说，树的深度越深，枝叶越多，模型越复杂 

* 4）树模型和树的集成模型的目标，都是减少模型复杂度，把模型往图像的左边移动 

具体的每个参数，都如何影响我们的复杂度和模型呢？

**随机森林的调参方向是降低复杂度**

#### 参考调参顺序：

![1634651926665](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1634651926665.png)

#### 偏差和方差：

偏差：**模型的预测值与真实值之间的差异**，即每一个红点到蓝线的距离。在集成算法中，每个基评估器都会有 自己的偏差，集成评估器的偏差是所有基评估器偏差的均值。模型越精确，偏差越低。 

方差：反映的是模型每一次输出结果与模型预测值的**平均水平之间的误差**，即每一个红点到红色虚线的距离，  衡量模型的稳定性。模型越稳定，方差越低。 

![1634652171975](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1634652171975.png)

**方差和偏差时此消彼长的**

![1634652196210](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1634652196210.png)

由图可知，复杂度高，方差高，总泛化误差高 ； 复杂度低，偏差低，总泛化误差低



### 画图调参

* 单个超参数图像

```
第一步：写一个定义的储存分数的数组
第二步：规定参数的步长 ， 一般使用range() 或者 np.linspace()
第三步：写一个for循环 for i in range() 或者 np.linspace()：
第四步：实例化模型，将调的参数值设置成 i ，然后fit()传入数据
第五步：调用实例化模型的score(数据)属性获得分数，append()方法加入数组
第六步：plot画图，plt.plot(range(),数组,label='...')
```

* 网格搜索

```
from sklearn.model_selection import GridSearchCV

第一步：写参数列表，把参数值的区间放进去
第二部：实例化模型
第三步：放入GridSearchCV()函数，(模型，x，y，评估指标)
第四步：best_param_ , best_score_ 属性拿到最好的属性和分数

像画图的话：
调用这个方法：GS.cv_results_['mean_test_score'] 得到分数的列表
然后直接plot()画图步骤同上
```

网格搜索代码：

```
rfr = RandomForestRegressor()

parameters={'n_estimators':[50,100,200],'max_depth':[1,2,3]}
clf = GridSearchCV(rfr,parameters,cv=5)
clf.fit(train.data,train.target)
score_test = mean_squared_error(test_target,clf.predict(test_data))

sorted(clf.cv_results_.keys())
```

随机森林网格调参：

```
from sklearn.model_selection import RandomizedSearchCV

rfr = RandomForestRegressor()

parameters={
'n_estimators':[50,100,200,300],
'max_depth':[1,2,3,4,5]
}
clf = RandomizedSearchCV(rfr,parameters,cv=5)
clf.fit(xtrain,ytrain)
score_test = mean_squared_error(xtest,clf.predict(xtest))

sorted(clf.cv_results_.keys())
```

参考：

![1644026187569](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644026187569.png)



![1644026285278](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644026285278.png)

### ROC画图代码

```
def plot_AUC(model,X_test,y_test):
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
```





## 采样

>  不平衡数据集的问题会影响机器学习算法的学习阶段和后续的预测。平衡问题对应于不同类中样本数量的差异。如下图所示，当不平衡比例较大时，决策函数倾向于样本数量较大的类。 

## **2、** **过采样**

### **2.1.1 Naive random over-sampling**

采用随机过采样的方式

**from** **imblearn.over_sampling** **import** RandomOverSampler

ros = RandomOverSampler(random_state=0)

X_resampled, y_resampled = ros.fit_resample(X, y)

### **2.1.2 SMOTE和ADASYN**

**from** **imblearn.over_sampling** **import** SMOTE, ADASYN

X_resampled, y_resampled = SMOTE().fit_resample(X, y)

X_resampled, y_resampled = ADASYN().fit_resample(X, y)

当随机过采样器通过复制少数类的一些原始样本进行过采样时，SMOTE和ADASYN通过插值产生新的样本

**2.1.3 SMOTE变体**

BorderlineSMOTE: Hui Han, Wen-Yuan Wang, and Bing-Huan Mao. Borderline-smote: a new over-sampling method in imbalanced data sets learning. In International conference on intelligent computing, 878–887. Springer, 2005.

**from** imblearn.over_sampling **import BorderlineSMOTE**

SVMSMOTE: Hien M Nguyen, Eric W Cooper, and Katsuari Kamei. Borderline over-sampling for imbalanced data classification. In *Proceedings: Fifth International Workshop on Computational Intelligence & Applications*, volume 2009, 24–29. IEEE SMC Hiroshima Chapter, 2009.

**from** **imblearn.over_sampling** **import** SVMSMOTE

KMeansSMOTE: Felix Last, Georgios Douzas, and Fernando Bacao. Oversampling for imbalanced learning based on k-means and smote. arXiv preprint arXiv:1711.00837, 2017.

**from** **imblearn.over_sampling** **import** KMeansSMOTE

[SMOTENC](https://zhuanlan.zhihu.com/p/137826761/edit#imblearn.over_sampling.SMOTENC) :Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic minority over-sampling technique. *Journal of artificial intelligence research*, 16:321–357, 2002.

该方法适合 混合数据类型，如包含连续和分类特征

**from imblearn.over_sampling import SMOTENC**

**smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)**

**X_resampled, y_resampled = smote_nc.fit_resample(X, y)**



## **3、** **下采样**

**3.1 ClusterCentroids**

**ClusterCentroids** 利用K-means来减少样本的数量。因此，每个类的合成都将以K-means方法的中心点来代替原始样本。

**from** **imblearn.under_sampling** **import** ClusterCentroids

**3.2 RandomUnderSampler**

RandomUnderSampler是一种快速和简单的方法来平衡数据，随机选择一个子集的数据为目标类，且可以对异常数据进行处理

**from** **imblearn.under_sampling** **import** RandomUnderSampler

**3.3 EditedNearestNeighbours**

应用最近邻算法，通过删除与邻域“不够一致”的样本来“编辑”数据集

**from** **imblearn.under_sampling** **import** EditedNearestNeighbours



**3.4 RepeatedEditedNearestNeighbours**

是EditedNearestNeighbours的扩展，通过多次重复该算法形成editednearestneighbour

**from** **imblearn.under_sampling** **import** RepeatedEditedNearestNeighbours



**3.5 AllKNN**

与RepeatedEditedNearestNeighbours不同的是，该算法内部的最近邻算法的近邻数在每次迭代中都会增加。

**from** **imblearn.under_sampling** **import** AllKNN

**3.6 CondensedNearestNeighbour**

CondensedNearestNeighbour使用1近邻规则迭代地决定是否应该删除一个样本，该算法对噪声敏感且会添加噪声样本

**from** **imblearn.under_sampling** **import** CondensedNearestNeighbour

**3.7 OneSidedSelection**

该算法可以消除噪声样本

**from** **imblearn.under_sampling** **import** OneSidedSelection



**3.8 NeighbourhoodCleaningRule**

**from** **imblearn.under_sampling** **import** NeighbourhoodCleaningRule

**3.9 InstanceHardnessThreshold**

InstanceHardnessThreshold是一种特定的算法，对数据进行分类器训练，去掉概率较低的样本

**from** **imblearn.under_sampling** **import** InstanceHardnessThreshold

该函数有两个参数可以设置：estimator 和CV



### **4、** **上采样和下采样的融合**

因为过采样会产生过多的噪声，通过先使用SMOTE之后再连接清洗的方法来得到相对好的空间样本

#### 4.1 SMOTEENN:

**from** **imblearn.combine** **import** SMOTEENN

#### 4.2 SMOTETomek:

**from** **imblearn.combine** **import** SMOTETomek



### **5分类器集成**

#### 5.1 BalancedBaggingClassifier

在集成分类器中，bagging方法是在不同的随机选择数据子集上建立多个估计器。在scikit-learn中，这个分类器被称为BaggingClassifier。但是，这个分类器不允许平衡每个数据子集。因此，当对不平衡数据集进行训练时，该分类器将有利于样本数目多的类别。

BalancedBaggingClassifier允许在训练集成的每个估计器之前重新采样每个数据子集。

**from** **imblearn.ensemble** **import** BalancedBaggingClassifier

bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),

sampling_strategy='auto',replacement=**False**,random_state=0)

bbc.fit(X_train, y_train)

#### 5.2 BalancedRandomForestClassifier

BalancedRandomForestClassifier是另一种集成方法，在这种方法中，将为森林中的每棵树提供一个平衡的样本。

**from** **imblearn.ensemble** **import** BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)

brf.fit(X_train, y_train)

#### 5.3 [RUSBoostClassifier](https://zhuanlan.zhihu.com/p/137826761/edit#imblearn.ensemble.RUSBoostClassifier)

RUSBoostClassifier在执行一个增强迭代之前对数据集随机抽样

**from** **imblearn.ensemble** **import** RUSBoostClassifier

rusboost = RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R',

random_state=0)

rusboost.fit(X_train, y_train)

#### 5.4 EasyEnsembleClassifier

easybleclassifier允许bag一个在平衡数据训练的Adaboost

**from** **imblearn.ensemble** **import** EasyEnsembleClassifier

eec = EasyEnsembleClassifier(random_state=0)

eec.fit(X_train, y_train)







## 决策树

### 优势：

```
1.生成容易理解的分类规则，意思就是这个模型的可解释性很好  例如：生成的规则是未满18岁，并对驾驶感兴趣，那么不准考驾驶证
2.很容易可视化，画个图就能看清树的结构
3.训练和预测超级快
4.较少的参数数目
5.支持数值和类别特征
```

### 劣势

```
1.决策树对输入数据的噪声非常的敏感
2.决策树的构建边界有局限性，它由一个垂直于坐标轴的超平面组成，会出现分类边界过于复杂
3.我们需要通过剪枝，设定叶子结点的最小数目，设定树的决策树的最大深度来避免过拟合
4.不稳定性，数据的微小变化，会导致整个树发生变化
5.如果数据中出现缺失值，将难以创建决策树
6.决策树只能内插，不能外推（随机森林和树提升方法也是如此），就是说一旦预测的数据出现在训练集的特征空间之外，那么决策树只能做出一样的预测，例如：黄球和蓝球模型中，预测的球>19或者<0，决策树模型将做出相同的预测
```

### 实际使用技巧

- 对于拥有大量特征的数据决策树会出现过拟合的现象。获得一个合适的样本比例和特征数量十分重要，因为在高维空间中只有少量的样本的树是十分容易过拟合的。
- 考虑事先进行降维( [PCA](https://sklearn.apachecn.org/#/decomposition.html?id=pca) , [ICA](https://sklearn.apachecn.org/#/decomposition.html?id=ica) ，使您的树更好地找到具有分辨性的特征。
- 通过 `export` 功能可以可视化您的决策树。使用 `max_depth=3` 作为初始树深度，让决策树知道如何适应您的数据，然后再增加树的深度。
- 请记住，填充树的样本数量会增加树的每个附加级别。使用 `max_depth` 来控制输的大小防止过拟合。
- 通过使用 `min_samples_split` 和 `min_samples_leaf` 来控制叶节点上的样本数量。当这个值很小时意味着生成的决策树将会过拟合，然而当这个值很大时将会不利于决策树的对样本的学习。所以尝试 `min_samples_leaf=5` 作为初始值。如果样本的变化量很大，可以使用浮点数作为这两个参数中的百分比。两者之间的主要区别在于 `min_samples_leaf` 保证叶结点中最少的采样数，而 `min_samples_split` 可以创建任意小的叶子，尽管在文献中 `min_samples_split` 更常见。
- 在训练之前平衡您的数据集，以防止决策树偏向于主导类.可以通过从每个类中抽取相等数量的样本来进行类平衡，或者优选地通过将每个类的样本权重 (`sample_weight`) 的和归一化为相同的值。还要注意的是，基于权重的预修剪标准 (`min_weight_fraction_leaf`) 对于显性类别的偏倚偏小，而不是不了解样本权重的标准，如 `min_samples_leaf` 。
- 如果样本被加权，则使用基于权重的预修剪标准 `min_weight_fraction_leaf` 来优化树结构将更容易，这确保叶节点包含样本权重的总和的至少一部分。
- 所有的决策树内部使用 `np.float32` 数组 ，如果训练数据不是这种格式，将会复制数据集。
- 如果输入的矩阵X为稀疏矩阵，建议您在调用fit之前将矩阵X转换为稀疏的`csc_matrix` ,在调用predict之前将 `csr_matrix` 稀疏。当特征在大多数样本中具有零值时，与密集矩阵相比，稀疏矩阵输入的训练时间可以快几个数量级。



### DecisionTreeClassifier

#### criterion: 决定不纯度的衡量指标

两种选择：默认基尼系数

entropy:信息熵   

gini：基尼系数

#### scikit-learn 的DecisionTreeClassifier类

* sklearn.tree.DecisionTreeClassifier(max_depth,max_features,min_samples_leaf) 
* max_depth: 树的最大深度 
* max_features：搜索最大分区的特征数（特征很多使这个参数很重要，因为基于所有的特征搜索分区很昂贵）
*  min_samples_leaf：叶节点的最小样本数

### 剪枝参数

- max_depth :限制树的最大深度
- min_samples_leaf:一个结点在分支后至少包含min个sample，不然不分结点，一般从5开始
- min_samples_split:当前的结点质保包含min个sample才能分支
- max_features: 限制分枝时的特征个数，属性太多，只要一部分，！降维思想
- min_impurity_decrease : 限制信息增益的大小，信息增益小于某个值不能分枝
- class_weight&min_weight_fraction_leaf:样本数据不平衡，有明显的偏重的时候使用

###  重要的属性和接口

- clf.apply(Xtest):返回每个测试样本所在的叶子结点的索引，注意！！只需要特征数据集
- clf.predict(Xtest):返回每个测试样本的分类结果



### 超参数曲线，来确定最优参数

每一个属性的分数都可以拿来画图，观察哪一个参数最好



### 实战代码

```python
Xtrain,Xtest,Ytrain,Ytest =train_test_split(wine.data,wine.target,test_size=0.3)

clf = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30
                                 ,splitter="best" #这个参数控制随机选项“random/best”
                                 )
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)

#画图代码
import graphviz 
dot_data = tree.export_graphviz(clf,
                                class_names=["琴酒","雪莉","贝尔摩德"]
                               ,filled=True   #颜色填充
                               ,rounded=True) # 圆形输出框
graph = graphviz.Source(dot_data)
graph

#查看决策树属性重要性
[*zip(feature_name,clf.feature_importances_)]


#使用剪枝参数
clf = tree.DecisionTreeClassifier(criterion = 'entropy'
                                 ,random_state=30
                                 ,splitter="random"
                                  ,max_depth=3
#                                  ,min_samples_leaf=10
                                 ,min_samples_split=10
                                 )
clf = clf.fit(Xtrain,Ytrain)
dot_data = tree.export_graphviz(clf,
                                class_names=["琴酒","雪莉","贝尔摩德"]
                               ,filled=True   #颜色填充
                               ,rounded=True) # 圆形输出框
graph = graphviz.Source(dot_data)
graph

score_train = clf.score(Xtrain,Ytrain)
score_train

#超参数曲线
import matplotlib.pyplot as plt
test = [] #根据分数来画图
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i+1
                                     ,criterion="entropy"
                                     ,random_state=30
                                     ,splitter="random"
                                     )
    clf = clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest) 
    test.append(score)
plt.plot(range(1,11),test,color="red",label="max_depth")
plt.legend()
plt.show()

clf.apply(Xtest)#每个样本所在的叶子标签索引
clf.predict(Xtest) #每个标签的预测值
```

## 树模型可视化

性能度量指数的图像：

```
plt.figure(figsize=(6, 4))
xx = np.linspace(0, 1, 50)
plt.plot(xx, [2 * x * (1 - x) for x in xx], label="gini")
plt.plot(xx, [4 * x * (1 - x) for x in xx], label="2*gini")
plt.plot(xx, [-x * np.log2(x) - (1 - x) * np.log2(1 - x) for x in xx], label="entropy")
plt.plot(xx, [1 - max(x, 1 - x) for x in xx], label="missclass")
plt.plot(xx, [2 - 2 * max(x, 1 - x) for x in xx], label="2*missclass")
plt.xlabel("p+")
plt.ylabel("criterion")
plt.title("Criteria of quality as a function of p+ (binary classification)")
plt.legend()
```

### 多模型对比可视化

#### 校准曲线

下面，我们使用小型训练数据集训练四个模型中的每一个，然后使用测试数据集的预测概率绘制校准曲线（也称为可靠性图）。校准曲线是通过对预测概率进行分箱创建的，然后将每个箱中的平均预测概率与观察到的频率（“阳性分数”）作图。在校准曲线下方，我们绘制了一个直方图，显示了预测概率的分布，或者更具体地说，是每个预测概率箱中的样本数。

```python
import numpy as np

from sklearn.svm import LinearSVC


class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = NaivelyCalibratedLinearSVC(C=1.0)
rfc = RandomForestClassifier()

clf_list = [
    (lr, "Logistic"),
    (gnb, "Naive Bayes"),
    (svc, "SVC"),
    (rfc, "Random forest"),
]

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")

# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()
```

模型误差分析：

[`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)返回经过良好校准的预测，因为它直接优化了对数损失。相比之下，其他方法返回有偏差的概率，每种方法都有不同的偏差：

- [`GaussianNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)倾向于将概率推到 0 或 1（见直方图）。这主要是因为朴素贝叶斯方程仅在特征条件独立的假设成立时才提供正确的概率估计[2](https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#id6)。然而，特征往往是正相关的，这个数据集就是这种情况，它包含 2 个作为信息特征的随机线性组合生成的特征。这些相关特征被有效地“计算了两次”，从而将预测概率推向 0 和 1 [3](https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#id7)。
- [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)显示相反的行为：直方图显示峰值在大约。0.2 和 0.9 的概率，而接近 0 或 1 的概率非常罕见。[Niculescu-Mizil 和 Caruana 1](https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#id5)对此进行了解释：“像套袋和随机森林这样的方法，从一组基本模型中平均预测可能难以做出接近 0 和 1 的预测，因为基础基本模型中的方差会使预测偏离这些值接近 0 或 1。因为预测被限制在区间 [0,1] 内，由方差引起的误差往往是单边的，接近零和一。例如，如果一个模型应该预测一个案例的 p = 0，那么 bagging 可以实现这一点的唯一方法是如果所有 bagged trees 都预测为零。如果我们向 bagging 平均的树添加噪声，这种噪声将导致一些树在这种情况下预测大于 0 的值，从而使 bagged ensemble 的平均预测远离 0。我们在随机森林中观察到这种影响最为强烈，因为使用随机森林训练的基础级树由于特征子集而具有相对较高的方差。” 结果，校准曲线显示出特征性的 sigmoid 形状，表明分类器信心不足，可能返回更接近 0 或 1 的概率。
- 为了展示 的性能[`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)，我们通过应用 min-max 缩放天真地将[decision_function](https://scikit-learn.org/stable/glossary.html#term-decision_function)的输出缩放为 [0, 1]，因为 SVC 默认情况下不输出概率。 [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)显示了比 更sigmoid 的曲线 [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)，这是最大边际方法[1](https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#id5)的典型特征，因为它们专注于难以分类的靠近决策边界（支持向量）的样本。



#### 等高线决策边界绘制

```python
def plot_decision_boundary(model, axis):
# meshgrid函数用两个坐标轴上的点在平面上画格，返回坐标矩阵
    X0, X1 = np.meshgrid(
# 随机两组数，起始值和密度由坐标轴的起始值决定
    np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
    np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),)
  
# ravel()方法将高维数组降为一维数组，c_[]将两个数组以列的形式拼接起来，形成矩阵
    X_grid_matrix = np.c_[X0.ravel(), X1.ravel()]
# 通过训练好的逻辑回归模型，预测平面上这些点的分类
    y_predict = model.predict(X_grid_matrix)
    y_predict_matrix = y_predict.reshape(X0.shape)

# 设置色彩表
    from matplotlib.colors import ListedColormap
    my_colormap = ListedColormap(['#DB7093', '#40E0D0', '#FFFF00'])

# 绘制等高线，并且填充等高区域的颜色
    plt.contourf(X0, X1, y_predict_matrix, cmap=my_colormap)
    
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt

iris = datasets.load_iris()
x = iris.data[:,:2]           #属性数据集
y =iris.target          #标签数据集

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

y0 = y_test.reshape(-1,1) #把y_test由一行n列的矩阵转化为n行一列的矩阵
y = y0.reshape(-1,1)
X = np.hstack((x_test,y)) #把x的测试集和y的测试集水平拼接起来

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)

plot_decision_boundary(knn_clf, axis=[4, 8, 1, 5])  #x轴间距4-8，y轴1-5
plt.scatter(X[y_test==0, 0], X[y_test==0, 1], color='r')  
#合成的矩阵中，一列一个属性,所以按照列来画图，选两个属性进行可视化，取出第一列和第二列中标签属于0的，画图
plt.scatter(X[y_test==1, 0], X[y_test==1, 1], color='g')
# 取出第一列和第二列中标签属于1的，画图
plt.scatter(X[y_test==2, 0], X[y_test==2, 1], color='b')
plt.show()
```

### 折线决策边界

```
y_1 = clf.predict(X[:,:-1])
plt.plot(X[:,:-1], y_1, color="cornflowerblue", label="max_depth=3", linewidth=2)
```

### 画图总结

* 画点图，scatter的x，y分别是两个属性具有相同标签的，因为二维空间可视化只能研究两个属性之间的关系
* 画折线可视化边界，plot的x和y分别是xtest和y_predict ，也就分类器的预测结果用折线的形式表达出来

### 多分类标签绘制

```
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification as make_ml_clf

COLORS = np.array(
    [
        "!",
        "#FF3333",  # red
        "#0198E1",  # blue
        "#BF5FFF",  # purple
        "#FCD116",  # yellow
        "#FF7216",  # orange
        "#4DBD33",  # green
        "#87421F",  # brown
    ]
)

# Use same random seed for multiple calls to make_multilabel_classification to
# ensure same distributions
RANDOM_SEED = np.random.randint(2 ** 10)


def plot_2d(ax, n_labels=1, n_classes=3, length=50):
    X, Y, p_c, p_w_c = make_ml_clf(
        n_samples=150,
        n_features=2,
        n_classes=n_classes,
        n_labels=n_labels,
        length=length,
        allow_unlabeled=False,
        return_distributions=True,
        random_state=RANDOM_SEED,
    )

    ax.scatter(
        X[:, 0], X[:, 1], color=COLORS.take((Y * [1, 2, 4]).sum(axis=1)), marker="."
    )
    ax.scatter(
        p_w_c[0] * length,
        p_w_c[1] * length,
        marker="*",
        linewidth=0.5,
        edgecolor="black",
        s=20 + 1500 * p_c ** 2,
        color=COLORS.take([1, 2, 4]),
    )
    ax.set_xlabel("Feature 0 count")
    return p_c, p_w_c


_, (ax1, ax2) = plt.subplots(1, 2, sharex="row", sharey="row", figsize=(8, 4))
plt.subplots_adjust(bottom=0.15)

p_c, p_w_c = plot_2d(ax1, n_labels=1)
ax1.set_title("n_labels=1, length=50")
ax1.set_ylabel("Feature 1 count")

plot_2d(ax2, n_labels=3)
ax2.set_title("n_labels=3, length=50")
ax2.set_xlim(left=0, auto=True)
ax2.set_ylim(bottom=0, auto=True)

plt.show()

print("The data was generated from (random_state=%d):" % RANDOM_SEED)
print("Class", "P(C)", "P(w0|C)", "P(w1|C)", sep="\t")
for k, p, p_w in zip(["red", "blue", "yellow"], p_c, p_w_c.T):
    print("%s\t%0.2f\t%0.2f\t%0.2f" % (k, p, p_w[0], p_w[1]))
```



### ROC绘制

```
#二分类
fpr, tpr, thresholds = roc_curve(ytest, yprob, pos_label=1)
fig,ax = plt.subplots()
x=np.linspace(0,1,352)
ax.plot(fpr,tpr)  #绘制ROC曲线
ax.plot(x,x,color='red',linestyle='--')
plt.xlim()
plt.ylim()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating characteristic curve')
plt.show()

#多分类
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
X=x
y = label_binarize(ytest, classes=[0, 1, 2])  # 将标签二值化
y_score = bst.predict(dtest,ntree_limit=bst.best_ntree_limit)

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())   #这个相当于是把所有的分类全部展开重新计算ROC, 看成一个大的二分类的结果
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(figsize=(12,6))
# plt.plot(fpr["micro"], tpr["micro"],
#          label='单类平均ROC curve (AUC= {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='三类平均ROC curve (AUC= {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('所有类别的ROC曲线')
plt.legend(loc="lower right")
plt.show()

```





##  回归树

#### criterion

- mse :使用均方误差 ，作为评估标准
- friedman_mse :使用费尔德曼均方误差
- mae :使用绝对均方误差

回归树的score返回的是 1-（残差平方和/均方误差的和），如果残差平方和远大于均方误差的和，模型非常糟糕

sklearn使用的是负均方误差



### cross_val_score参数设置

```
sklearn.model_selection.cross_val_score(

estimator, 需要使用的交叉验证的算法

X, y=None, x，y数据集

groups=None, 将数据集分割为训练/测试集时使用的样本的组标签（一般用不到）

scoring=None, 交叉验证最重要的就是他的验证方式，选择不同的评价方法，会产生不同的评价结果。具体可用哪							些评价指标，官方已给出详细解释，链 接：https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

cv=’warn’, 交叉验证的折数

n_jobs=None,调用CPU的内核数

verbose=0, 详细程度

fit_params=None, pre_dispatch=‘2*n_jobs’,

error_score=’raise-deprecating’)

参数： random_state=? 这个参数是拿来控制随机状态的，因为模型的构建、数据集的生成、数据集的拆分都是一个随机的过程。 所以如果我们要得到固定的结果就需要固定随机参数 固定random_state后，每次构建的模型是相同的、生成的数据集是相同的、每次的拆分结果也是相同的。
```



### 实战代码

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0)
cross_val_score(regressor #实例分类器
                ,boston.data #特征矩阵
                ,boston.target #标签矩阵
                ,cv=10
#                ,scoring = "neg_mean_squared_error" #负的均方误差
               ).mean()
               

#画图代码类似于决策树
```

## 梯度提升树GBDT

**相同点**：

- 都是由多棵树组成，最终的结果都是由多棵树一起决定。
- RF和GBDT在使用CART树时，可以是分类树或者回归树。

**不同点**：

- 组成随机森林的树可以并行生成，而GBDT是串行生成
- 随机森林的结果是多数表决表决的，而GBDT则是多棵树累加之和
- 随机森林对异常值不敏感，而GBDT对异常值比较敏感
- 随机森林是减少模型的方差，而GBDT是减少模型的偏差
- 随机森林不需要进行特征归一化。而GBDT则需要进行特征归一化

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

train_feature = np.genfromtxt("train_feat.txt",dtype=np.float32)
num_feature = len(train_feature[0])
train_feature = pd.DataFrame(train_feature)

train_label = train_feature.iloc[:, num_feature - 1]
train_feature = train_feature.iloc[:, 0:num_feature - 2]
train_feature

test_feature = np.genfromtxt("test_feat.txt",dtype=np.float32)
num_feature = len(test_feature[0])
test_feature = pd.DataFrame(test_feature)

test_label = test_feature.iloc[:, num_feature - 1]
test_feature = test_feature.iloc[:, 0:num_feature - 2]
test_feature

gbdt = GradientBoostingRegressor(
  loss = 'ls'
, learning_rate = 0.1
, n_estimators = 100
, subsample = 1
, min_samples_split = 2
, min_samples_leaf = 1
, max_depth = 3
, init = None
, random_state = None
, max_features = None
, alpha = 0.9
, verbose = 0
, max_leaf_nodes = None
, warm_start = False
)

gbdt.fit(train_feature, train_label)
pred = gbdt.predict(test_feature)
total_err = 0

for i in range(pred.shape[0]):
    print('pred:', pred[i], ' label:', test_label[i])
print('均方误差:', np.sqrt(((pred - test_label) ** 2).mean()))


```

### 树模型算法区别

所有种类的决策树算法有哪些以及它们之间的区别？scikit-learn 中实现何种算法呢？

[ID3](https://en.wikipedia.org/wiki/ID3_algorithm)（Iterative Dichotomiser 3）由 Ross Quinlan 在1986年提出。该算法创建一个多路树，找到每个节点（即以贪心的方式）分类特征，这将产生分类目标的最大信息增益。决策树发展到其最大尺寸，然后通常利用剪枝来提高树对未知数据的泛华能力。

C4.5 是 ID3 的后继者，并且通过动态定义将连续属性值分割成一组离散间隔的离散属性（基于数字变量），消除了特征必须被明确分类的限制。C4.5 将训练的树（即，ID3算法的输出）转换成 if-then 规则的集合。然后评估每个规则的这些准确性，以确定应用它们的顺序。如果规则的准确性没有改变，则需要决策树的树枝来解决。

C5.0 是 Quinlan 根据专有许可证发布的最新版本。它使用更少的内存，并建立比 C4.5 更小的规则集，同时更准确。

[CART](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29)（Classification and Regression Trees （分类和回归树））与 C4.5 非常相似，但它不同之处在于它支持数值目标变量（回归），并且不计算规则集。CART 使用在每个节点产生最大信息增益的特征和阈值来构造二叉树。





## 随机森林

![1634651520320](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1634651520320.png)

### 随机森林中的属性

* rfc.estimators_ :查看森林中树的情况
    rfc.estimator_[0]:单独查看

#### 为什么随机森林里的树会有众多不同的判断结果？

随机森林本身就有random_state，用法和决策树一样，所以radnom_state时控制森林是否一样而不是树是否一样，所以里面的每棵树不一样

* booststrap : 控制有放回的采样技术的参数，随机采样的自助集，因为自助集互不相同所以树各不相同，一般来说自助集平均会有63%的的原始数据
* oob_score：out of data 就是被浪费的37%的数据

**用这个方法可以不用划分数据集和测试集了**

* feature_importances_ :特征重要性即信息增益
* apply(Xtest)：
* predict(Xtest)：
* predict_proba(Xtest)：测试样本被分到每一个类的概率

R^2 = 负无穷到1，越靠近1越好
neg_mean_squared_error :负的均方误差

### 用随机森林填补缺失值

在sklearn中，sklearn.impute.Simplemputer来轻松地均值、中值或其他最常见的数值填补缺失值

各行各列都要有缺失的数据，需要行索引和列索引
然后用0，均值和随机森林来填补缺失值



#### 原理：

> 特征矩阵和标签有某种联系，可以通过特征矩阵预测标签，也可以通过标签预测特征值

假设有n个特征的数据，特征T有缺失值，那么我们要预测的标签就是特征T，剩下的特征和原本的标签成为新的特征矩阵，

特征T不缺失值的其他特征+原来的标签 = X_train
特征T（不缺失的列）没有缺失 = Y_train

特征T缺失的值（发生缺失的行）对应的其他特征值 = X_test
特征T缺失的值：未知（预测的目标）

如果不止一个特征有缺失，先从缺失最少的开始填充，确定一个要填补的特征，其他缺失的特征用0填补

#### 实战代码：

```python
 %matplotlib inline 
import matplotlib.pyplot as plt
#我们画图的时候需要这个环境，把以下的代码全部放进这个环境
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
x = wine.data 
y = wine.target

Xtrain,xtest,Ytrain,ytest = train_test_split(x,y,test_size=0.3)
clf = DecisionTreeClassifier(random_state=0)
random = RandomForestClassifier(random_state=0) #默认10棵树
clf = clf.fit(Xtrain,Ytrain)
random = random.fit(Xtrain,Ytrain)
score_c =clf.score(xtest,ytest)
score_r = random.score(xtest,ytest)
print(f'随机森林：{score_r}\n决策树：{score_c}')

#交叉验证曲线
from sklearn.model_selection import cross_val_score
random = RandomForestClassifier(n_estimators=25)
random_s = cross_val_score(random , x,y,cv=10)

clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf,x,y,cv=10)

plt.plot(range(1,11),random_s,label='RandomForest')
plt.plot(range(1,11),clf_s,label='DecisionTree')
plt.legend()
plt.show()

#取平均值
label = "RandomForest"
for model in [RandomForestClassifier(n_estimators=25),DecisionTreeClassifier()]:
    score = cross_val_score(model,x,y,cv=10)
    print(f"{label} ：{score.mean()}") #分数的平均值
    plt.plot(range(1,11),score,label=label)
    plt.legend()
    label= "DecisionTree"  #更新以下label

#交叉验证学习曲线    
rfc_l =[]
clf_l = []
for i in range(10): #循环10次10次交叉验证，就是100次
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc,x,y,cv=10).mean() 
    rfc_l.append(rfc_s)
    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf,x,y,cv=10).mean()
    clf_l.append(clf_s)
plt.plot(range(1,11),rfc_l,label='RandomForest')
plt.plot(range(1,11),clf_l,label='DecisionTree')
plt.legend()
plt.show()

#n_estimators 学习曲线
superpa =[]
for i in range(100):
    rfc = RandomForestClassifier(n_estimators=i+1)
    rfc_s = cross_val_score(rfc,x,y,cv=10)
    superpa.append(rfc_s)
print(f'最高的分数：{max(superpa)}',f'获得最高分数时森林中树的数量：{superpa.index(max(superpa))}')
plt.figure(figize=[20,5])
plt.plot(range(1,101),superpa)
plt.show()


```

### 随机森林调参
#### 一个一个的去调：
* n_estimators:每10个数为一个阶段
* max_depth:数据集较小的情况采用1-20;大的数据30-50
* max_feature :param_grid {'max_features':np.arange(5,30,1)}
* criterion : {'criterion':['gini','entropy']}

* 调参的结果可能分数上升的很小，比如0.005

#### 多个参数一起调：

params = {'n_estimators':[],'max_depth':[] ...}

### 调参思想：

复杂度-泛化误差的图像（偏差-方差）

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

data = load_breast_cancer()
x = data.data
y = data.target # 二分类数据

data.data.shape  #569个样本数据，30个特征，样本特征比很小，很容易过拟合

rfc = RandomForestClassifier(n_estimators=100,random_state=90)
score_pre = cross_val_score(rfc,x,y,cv=10).mean()
score_pre

```

超参数图像：

```python
scorel = []
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1
                                ,random_state=90)
    score = cross_val_score(rfc ,x ,y, cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*10)+1) #打印出最高的分数和对应的树的数量
plt.figure(figsize=[20,5])
plt.plot(range(1,201,10),scorel)
plt.show()
```

```python
scorel = []
for i in range(60,80):
    rfc = RandomForestClassifier(n_estimators=i
                                ,random_state=90)
    score = cross_val_score(rfc ,x ,y, cv=10).mean()
    scorel.append(score)
print(max(scorel),([*range(60,80)][scorel.index(max(scorel))])) #打印出最高的分数和对应的树的数量
plt.figure(figsize=[20,5])
plt.plot(range(60,80),scorel)
plt.show()
```



```python
#调整max_depth
param_grid = {'max_depth':np.arange(1,20,1)}

rfc = RandomForestClassifier(n_estimators=73
                            ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(x,y)
GS.best_params_ , GS.best_score_
```

max_depth 设置为有限之后，准确率下降了,说明我们降低模型复杂度，泛化误差上升，说明在误差曲线的左边

所以，要增加模型复杂度，达到最低繁华误差

- min_samples_leaf 和 min_sample_split 是降低模型复杂度的,往左调的
- max_features可以往右侧调,max_features的最小值是特征数的开平方

```python
param_grid = {'max_features':np.arange(5,30,1)}

rfc = RandomForestClassifier(n_estimators=73
                            ,random_state=90
                            )
GS = GridSearchCV(rfc ,param_grid ,cv=10)
GS.fit(x,y)
GS.best_params_ , GS.best_score_
```

**这里我们发现模型的分数上升了，说明调参思路正确，有效

**如果向右调整，模型分数反而降低了，就不用调了，因为泛化误差已经在最低点附近了。

要再想提高

- 1.更换算法
- 2.更换预处理方式

```
GS.cv_results_['mean_test_score']
```



## 数据预处理：Preprocessing&Impute



### 数据无量纲化

概念：将不同规格的数据转换到统一规格，或将不同分布的数据转换统一分布 作用：无量纲化可以加快运行速度

数据无量纲化分为线性和非线性

线性包括中心化处理和缩放处理

- 中心化 ：让所有的记录加上或减去一个固定值，让数据移动到某个位置
- 缩放：让所有数据乘以或除以一个固定值，让数据固定再某个范围，比如取对数

非线性：



####  数据归一化

preprocessing.MinMaxScaler

**数据归一化（Normalization或Min-Max Scaing）： 当数据按照最小值中心化之后，再按照极差缩放，数据移动最小值个单位，并且位于[0,1]之间，也可以是其他范围 ** 

 ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASkAAABgCAYAAABBuYSSAAAVPUlEQVR4Ae2dh/MUtRfA/Z9UVISxoo6oqAjK2AA7Ko5i772jgr1X7FiwYu+9915QwYIFFbvE+eT3e0c2372yu8nelvdmbvZ2b5PNviSfe3lpqxiVSmpg+fLlZvHixUP5/PDDD5XUSahE/fHHH2bJkiVD0a2fp0uXLjUrVqwI9WqNjGeVRr5VQ14KWPiFOvZ50wElRUNBJZqo/lEhVfE8KhNUbQGUZLmCSjRR7aNCqtr5Y1NXBqjaBijJdgWVaKK6R4VUdfMmkbKYoGoroETBCirRRDWPCqlq5ktqqmKAqu2AEkUrqEQT1TsqpKqXJz1TFBJUCqikqhVUSX1U5UwhVZWcyJCOEKBSQKUrXEGVrpdhXlVIDVP7BZ5dBFQKqN6KV1D11k/ZvyqkytZ4wOflAZUCarAMUFANpqcy7lJIlaHliM/IAioFVLaM+P3333VkejaVRblbIRVFreVGOgioFFD58kRBlU9vIUMppEJqc4hx9QKVAqpYxiioiumvaGiFVFENVih8GqgUUGEySEEVRo95YlFI5dFahcO4oFJAhc0oBVVYfQ4am0JqUE3V6D7gpICKk2EKqjh67RWrQqqXdvQ31UCKBhRUKUqJeEkhFVG5GnVzNaCgKi9vFVLl6Vqf1DANKKjKyVCFVDl61qc0VAMKqvgZq5CKr2N9QsM1oKCKm8EKqbj61dhbogEFVbyMVkjF063G3DINKKjiZLhCKo5eNdaWakBBFT7jFVLhdaoxtlwDCqqwBUAhFVafGptqwGpAQRWuICikwulSY1INJDSgoEqoI/eJQiq36jSgaqC/BhRU/XXU7w6FVD8N6e+qgYIaUFAVU6BCqpj+NLRqYCANKKgGUlPqTQqpVLXoRdVAeA0oqPLpVCGVT28aSjWQSwMKquxqU0hl15mGUA0U0oCCKpv6FFLZ9KV3qwaCaGDFihVm6dKlZvHixUP/LFmyxLDPYFVFIVXVnNF0NV4DCqrBslghNZie9K6GaeCXX34xJ5xwgqHplVXOPvts8+mnn2YNlnq/gipVLYmLCqmEOvSk6hr48ccfDRW7iNC02W233czLL7+cK5rvvvvOTJ482e5unCsCL5CCylOId6qQ8hSip9XVwG233WZWXXVV88gjjxRK5IUXXmhOPvnkQnHMnz/fzJw5szAwJREKKtHEyKNCaqRO9EpFNQAYpk+fbr7++uvcKfz222/NOuusYzgWkX/++cdsscUW5uGHHy4STSKsgiqhjs6JQqqjCv3SBg2cc845ZtasWUFe9dxzzzV77rlnkLgkEgWVaGLlUSG1Uhf6rQUamDBhgrn++uuDvOmbb75pVlttNYOfLKQoqJLaVEgl9dGos3fffdccdNBBZpdddjEHH3yw+eKLL8yRRx5p9thjD3PGGWcE86f0UtpLL71krY3tttvOfPLJJ+bZZ5+1adp1113N9ttvbz788EPz2WefmWOPPdbsvffeZptttjGPPvpoIsrHH3/cpnnbbbc177//fue3L7/80uy3335m4sSJ5qmnnrJxHX300WbGjBlm0qRJ5rnnnuvcyxd2dcan1c1h/tZbb1kriybl4YcfbhYtWmSP6Gvu3Lkj9PXbb7/Z+J555pnEc0KcKKhWalEhtVIXjfrGIMF9993X/stT4HfeeWczZswY895771lAUVmL+mX6Kezff/+1vWj0pm255ZYGK+bEE080y5cvtxUeSAEvQANwkNmzZ5u1117bEBb56quv7O9///239QExbEDkkEMOsUMBAPG4ceMsYOSdjjvuOLPxxhvLrfYInHjv77//PnGdE4BEOn7++WebNoCI74qhBjyTcMuWLRsRjmfMmzdvxPUQFxRU/9OiQipEaapgHEcddZT56KOPOimj0mFRIRynTp1qqPgx5dVXX7XQ+fPPP80aa6xh9tprr4Q1Ajg32GCDRHOJMUg0oSRtZ511lnnjjTesFQgorr76aptkxjlheSHAbvz48RZ+8j6AZezYsXJqj/fcc4+FzV9//ZW4zgnAw9IUAahu/LvvvrvBWe7LDjvsYE499VT/crBzBZUxCqlgxalaETHVQQQLgAp+0UUXyaWeRywdKi0WyiCf559/PjU+QIJlIhbMQw891LkP62rNNdccUcEBKE0+Ed6DinrxxRfbd5CePUDDeCXiB2rnn3++BLFHoAwEXbn99tttHMTni6sv4kVfV111lX/biHNgj9UWU9oOKoVUzNJVkbgfe+wxW+leeeWVgVNEk4i5ZYN80iwT90GXXHKJfT4+IRF8VYDgwQcflEv2WVy74IILOtf4QtOP7v599tkncZ2TJ554wsaDr0vk888/t9euvfZauWSPCxYssNfTLCL3xvvv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) 

x= x-数据中的最小值/极差（最大值-最小值）

- 参数feature_range: 实现将数据归一化到[0,1]之外的范围

**如果数据太多fit报错，可以采用partial_fit()**

```python
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = [[-1,2],[-0.5,6],[0,10],[1,18]]
pd.DataFrame(data)

scaler = MinMaxScaler()
scaler = scaler.fit(data)#生成min(x),max(x)
result = scaler.transform(data) #导出结果

# result = scaler.fit_transform(data)  # 也可以一步到位
result
```







#### 数据标准化

preprocessing.StandardScaler

当数据按均值中心化后，再按照标准差缩放，数据就会服从均值为0，方差为1的正态分布 **这个过程交数据标准化** x = x-μ（均值）/σ（标准差）

- inverse_transform(x_std): 逆转标准化，就是将数据还原

```python
from sklearn.preprocessing import StandardScaler
data = [[-1,2],[-0.5,6],[0,10],[1,18]]
scaler = StandardScaler()
scaler.fit(data) #生成均值和方差

#查看方差和均值
scaler.mean_,scaler.var_

x_std = scaler.transform(data)
x_std
```



### 缺失值处理

#### Impute.SimpleImputer

- missing_values :告诉SimpleImputer ，数据中的缺失值长啥样，默认空值 np.nan
- strategy : mean/median/most_frequent/constant
- fill_value : 当strategy为 constant 是 就是用fill_value的值

```python
#数据升到二维 ， 要导入sklearn数据必须最少是二维
Age = data.loc[:,['Age']].values.reshape(-1,1)
```

```python
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer() #默认是均值填补
imp_median = SimpleImputer(strategy='median')#中位数
imp_0=SimpleImputer(strategy='constant',fill_value=0)#0填充

Embarked = data.loc[:,'Embarked'].values.reshape(-1,1)
imp_mode = SimpleImputer(strategy='most_frequent')#众数填充
data.loc[:,'Embarked'] = imp_mode.fit_transform(Embarked)

```





#### 另外的填补空值的方法

```python
#fillna填补空值
data.loc[:,'Age']=data.loc[:,'Age'].fillna(data.loc[:,'Age'].median())

#dropna删除空值
data.dropna(axis=0,inplace=True)
```



## 处理分类型特征：编码与哑变量

在机器学习中，算法在fit时必须用数值型的数据，而不是文字型的数据，现在就是将文字转换维数值

- sklearn.processing.LabelEncoder :标签专用，将分类转换为分类数值型 ，可以导入一维数组

- sklearn.preprocessing.OrdinalEncoder:特征专用，将分类特征转换维数值型

- sklearn.preprocessing.OneHotEncoder: 独热编码，哑变量 1.特征分类之间没有联系，可以直接转换，名词性变量

  2.特征分类之间有联系，不能计算，抽象性变量

  3.特征分类之间有联系而且可以计算

  **所有要用标签向量的方式表示**

#### 特征转换

```
#categories_ :查看所有特征的类别
OrdinalEncoder().fit(data1.iloc[:,1:-1]).categories_
```

```python
data1.iloc[:,1:-1] = OrdinalEncoder().fit_transform(data.iloc[:,1:-1])
data1.head()
```

#### 独热编码

```python
from sklearn.preprocessing import OneHotEncoder
x = data.iloc[:,1:-1]

enc = OneHotEncoder(categories='auto').fit(x)
result = enc.transform(x).toarray() #导出结果并转换成数组
# 这是一个稀疏矩阵
result

#一步到位
OneHotEncoder(categories='auto').fit_transform(x)

#还原
enc.inverse_transform(result)
#查看特征名
enc.get_feature_names()
```



 特征可以做哑变量，标签可以吗？可以 sklearn.preprocessing.LabelBinarizer可以对做哑变量 

### 处理连续性特征：二值化与分段

* sklearn.preprocessing.Binarizer
根据阈值将数据二值化（特征值设置为0或1），用于处理连续性变量

* preprocessing.KBinsDiscretizer
将连续变量分成多个类，能够将连续性变量排序后按顺序分箱后编码

| 参数     | 意义                   |
| -------- | ---------------------- |
| n_bins   | 每个特征分箱的个数     |
| encode   | onehot,ordinal         |
| strategy | quantile,uniform,keans |

**threshold是设置阈值的参数，大于阈值的为1，小于阈值的为0**

```python
from sklearn.preprocessing import KBinsDiscretizer
x = data.iloc[:,0].values.reshape(-1,1)

est = KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='uniform')
est.fit_transform(x)

#查看分箱类别
set(est.fit_transform(x).ravel())
est = KBinsDiscretizer(n_bins=3,encode='onehot',strategy='uniform')
est.fit_transform(x).toarray()

```



## 特征工程：

特征提取：从文字、图像、声音等数据中提取特征

特征创造：把现有的特征进行组合，创造新的特征

特征选择：从所有特征中，选择出有意义的特征进行训练，尤其是要选择相关性比较高的特征

四种方法：过滤法、嵌入法、包装法和降维算法

###  Filter 过滤法

全部特征-》最佳特征子集-》算法-》模型

### 方差过滤 VarianceThreshold :

  **舍弃所有方差小于threshold的特征,默认为0**
  如果我们只需要一半的特征，直接令方差为中位数

```
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=方差阈值)	#不填参数，默认方差为0
x_var0 = selector.fit_transform(x) 	#获取删除不合格特征之后的新特征矩阵
x_var0.shape
```



#### 方差过滤对特征的影响

过滤法的目的：在维持算法表现下，降低算法的运算时间的成本
* 节省算法运行时间，正确率稍有上升

方差过滤对于计算量大，需要遍历特征或升维运算的算法影响特别大，例如：KNN，单颗决策树，SVM，

**随机森林会随机抽取10-20个特征，过滤法对随机森林无效**

* 方差设置的阈值可能让模型变得更糟，代表有效的特征被过滤了
**如果特征中很多都是有效的特征，那就放弃过滤法，选择其他的方法**
* 如何选择Threshold？
自己调试，hahaha



### 卡方过滤 ：

    专门对离散型标签的相关性过滤

卡方检验本质：推测两组数据之间的差异，前提假设两组数据之间相互独立 卡方检验返回卡方值和P值，卡方值无法界定，但是P值可以  

P值<=0.05 :特征相关 ；P值>0.05:特征相互独立 ； P值为0，都是相关的！
    
过程：卡方检验计算特征和标签之间的卡方统计量，并按照卡方统计量由高到低为特征进行排名，再结合feature_selection,输入一个**评分标准**来选出k个分数最高的特征的类，由此删除掉与标签无关的特征

**！卡方检验只能计算非负的数据**

* SelectKBest : 用来选择k个分数最高的特征类

  * 相关信息法

  ```
  #第一个参数是估计评估特征后的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，p值），k是选择特征的个数
  from scipy.stats import pearsonr
  SelectKbest(
  	lambda X,Y:np.array(list(map(lambda x:pearsonr(x,Y),X.T)).T[0],k=2).fit_transform(iris.data,iris.target)
  
  ```

  * 最大信息系数法

  ```
  from mineppy import MINE
  def mic(x,y):
  	m=MINE()
  	m.conpute_score(x,y)
  	return (m.mic(),0.5)	#把p值定为0.5
  	
  SelectKbest(
  	lambda X,Y:np.array(list(map(lambda x:mic(x,Y),X.T)).T[0],k=2).fit_transform(iris.data,iris.target)
  ```

  

* chi2 ：卡方检验类

  ```python
  #这里采用已经经过方差过滤后的数据进行卡方检验，因为方差检验的结果提升了
  
  from sklearn.ensemble import RandomForestClassifier as RFC
  from sklearn.model_selection import cross_val_score
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import chi2
  #假设我们要筛选出300个特征
  x_fschi = SelectKBest(chi2 , k=300).fit_transform(x_select_by_median_var,y)
  x_fschi.shape
  
  cross_val_score(RFC(n_estimators=10,random_state=0),x_fschi,y,cv=5).mean()
  #如果效果降低了，说明k=300删除了有效特征
  ```

  

```
如何判断k等于多少时，效果最好呢？

1.学习曲线

2.P值
```

 ![img](data:image/png;base64,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) 

```python
#获取卡方值和P值
chivalue , Pvalue = chi2(x_select_by_median_var,y)

#k取多少？我们要消去p>0.05的
k= chivalue.shape[0]-(Pvalue>0.05).sum()#特征的数量-P值大于0.05的特征数量，剩下的就是有效的

#这样k就可以确定啦！
x_fschi = SelectKBest(chi2 , k=？).fit_transform(x_select_by_median_var,y)
cross_val_score(RFC(n_estimators=10,random_state=0),x_fschi,y,cv=5).mean()
```



### F检验：

    方差齐性检验，用来捕捉每个特征和标签之间的线性关系的过滤方法
本质：寻找两组数据之间的线性关系，衡量标准P值和卡方检验一样

feature_selection.f_classif (F检验分类，离散型变量)和 feature_select.f_regresion（F检验回归，连续性变量）

**数据正态分布效果比较好** 
所以用无量纲化和标准化转换数据为正态分布



### 互信息法：
用来捕捉每个特征与标签之间的**任意关系**的过滤方法，回归和分类

返回的值时>0(相关)和<=0（不相关）

* feature_selection.mutual_info_classif 和feature_selection.mutual_info_regression



### Embedded 嵌入法

嵌入法：是一种让算法自己决定使用哪些特征的方法 ，即特征选择和算法训练同时进行。得到各个特征的权值系数，根据权值系数从大到小进行选择，选权值系数大的特征 ， 权值系数表示的是特征对模型的贡献，越大越好！

![1634918728699](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1634918728699.png)



**因此，无关的特征和无区分度的特征（需要方差过滤的特征）会被直接删除掉，是过滤法的进化版**



#### 缺点：

- 1.过滤法中使用的统计量有查找范围，比如P值<0.05 ， 而嵌入法的权值系数却没有这样的范围可以找--**没有一个临界值**

这种情况下

1.画学习曲线，根据图像逐渐缩小区间

2.根据模型本身的性质区判断

- 2.嵌入法可能时间耗费大，因为使用了算法，需要大量计算

#### feature_selection.SelectFromModel

SelectFromModel 是一个元变换器 ，可以与任何在拟合之后具有coef_ ,feature_importances_ 属性，或者模型参数中带有**可惩罚项**一起使用。

比如：随机森林和树模型中的feature_importances_ , 逻辑回归就带有L1，L2惩罚项，支持向量机也支持L2惩罚项

参数：

- estimator :使用的模型评估器
- threshold ： 阈值，小于阈值的舍弃掉

```
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

import pandas as pd
data = pd.read_csv(r'D:\360Downloads\sklearn\digit recognizor.csv')
x = data.iloc[:,1:]
y = data.iloc[:,0]
```

```
rfc = RFC(n_estimators=10,random_state=0)
x_embedded = SelectFromModel(rfc,threshold=0.005).fit_transform(x,y)
x_embedded.shape

#0.005这个阈值对于780个特征来说，是非常高的阈值，大约只能分到0.001的featur_inportance_
#特征从400多降到了47

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

SelectFromModel(
	LogisticRegression(penalty='l2',C=0.1,solver='lbfgs',
	multi_class='auto')
).fit_transform(x,y)

```

 

**但是这个阈值到底取到那里最好呢？ 画学习曲线 **

```
#time warning 10mins
import numpy as np
import matplotlib.pyplot as plt

rfc.fit(x,y).feature_importances_

#x取值要根据超参数的取值来定 ， threshold要根据feature_importance_来确定
threshold = np.linspace(0,(rfc.fit(x,y).feature_importances_).max(),20)

score =[]
for i in threshold:
    x_embedded = SelectFromModel(rfc,threshold=i).fit_transform(x,y)
    once = cross_val_score(rfc,x_embedded,y,cv=5).mean()
    score.append(once)
plt.plot(threshold,score)
plt.show()
#找thrshold 增大，score没怎么降低的那个转折点！！！，选择转折点附件的值，带入模型跑一跑
```

**我们发现模型分数比方差过滤的分数还高，特征还少，计算还快**

然而大型数据中我们还是会优先考虑过滤法



### 包装法

也是特征选择和算法训练同时进行的方法 它是效果最好的，用最少的特征就可以达到最好的效果 **不同之处**

包装法使用一个目标函数帮助我们选取特征，而不是自己输入某个评估指标

它会在初始特征集上训练评分标准coef_属性或者feature_importance_来获得每个特征的重要性，然后从当前的一组特征中修建最不重要的特征

选一次，就会保留最佳特征或删掉最差特征，下一次选择就根据上一次没有被选中的特征的进行选择，递归以上过程，直到所有特征耗尽，然后根据选出来的特征进行排名，最终选出一个最佳的子集

所以包装法的计算成本位于过滤法和嵌入法中间

####  feature_selection.RFE

- estimator:实例化后的评估器
- n_feature_to_select:想要选择的特征个数
- step : 每次迭代筛选出多少个特征

属性

- support_ : 返回所有的特征是否被选中的布尔矩阵，表示特征是否被选中了

- ranking ：返回的特征按次数迭代中的总和重要性排名

  ```python
  from sklearn.feature_selection import RFE
  rfc = RFC(n_estimators=10,random_state=0)
  selector = RFE(rfc ,n_features_to_select=340,step=50).fit(x,y)
  # step : 每次迭代筛选出多少个特征
  selector.support_.sum()
  selector.ranking_
  x_wrapper = selector.transform(x)
  cross_val_score(rfc,x_wrapper,y,cv=5).mean()
  ```

  

### 写在最后

- 数据量很大的时候：优先使用方差过滤和互信息法，再上其他方法
- 逻辑回归：优先使用嵌入法
- SVM：优先使用包装法
- 迷茫的时候，过滤法走起



## PCA

sklearn中降维算法放在decomposition中，本质是一个矩阵分解模块

* 什么是维度？
  维度是指样本的数量或者特征的数量，一般无特别说明，指的是特征的数量

对于图像，特征向量可以理解为坐标轴，一个特征向量对应一条坐标轴

* 什么是降维？
  降维算法中的”降维“，指的是降低特征矩阵中特征的数量。

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C1.png)

PCA中使用的信息量衡量指标，是**样本方差** ，方差越大，特征所带的信息量越多
![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C2.png)

> 为什么方差计算中除数是n-1 不是 n ？
> 为了得到样本方差的无偏估计-》下去查资料

* 二维数据降维例子：
  ![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C3.png)

三个点(1,1) (2,2) (3,3) x1和x2的方差都为1
我们现在要把两条坐标轴换成一条轴，即将二维数据降为一位数据

x1和 x2 -> x1* 和 x2*
** 旋转45度后，点全部落在了x1* 轴上面，x2* 全部为0 **

> 此时我们根据信息含量的排序，取信息含量最大的一个特征，就是我们想要的数据，也就是将x1* 和x2* 放到一起，取数据大的那个
> x2* 没有信息了，直接删掉

二维变一维，两条坐标轴变一条，尽可能保留原始数据的信息量

过程：
![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C4.png)

步骤3中：我们用来找出n个新的特征向量，让数据能够被压缩到少数特征上并且让总信息量不损失太多的技术就是矩阵分解

### 实战代码

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

#鸢尾花数据集高维数据可视化

iris = load_iris()
x = iris.data
y = iris.target

#看看数据集是几为
x.shape


#数据可视化，但是4维数据画不了，需要降维

pca = PCA(n_components=2)

pca = pca.fit(x) #放入拟合模型

x_dr = pca.transform(x) #获取新的矩阵
x_dr.shape

# 两列特征向量，如何画出三种鸢尾花的点呢？
#我们要取出三种不同的点
x_dr [y==0,0] 
#使用布尔索引，y==0返回布尔索引 True和False ，再取出第一列，就是x坐标
#具体含义就是 取出标签为0的行，第一列

#具体取出三种花的x，y轴的数据
x_dr[y==0,0] ,x_dr[y==0,1]
x_dr[y==1,0] ,x_dr[y==1,1]
x_dr[y==2,0] ,x_dr[y==2,1]

------------------------------------------
plt.figure()
plt.scatter(x_dr[y==0, 0], x_dr[y==0, 1], c="red", label=iris.target_names[0])
plt.scatter(x_dr[y==1, 0], x_dr[y==1, 1], c="black", label=iris.target_names[1])
plt.scatter(x_dr[y==2, 0], x_dr[y==2, 1], c="orange", label=iris.target_names[2])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()

-------------------------------------------
#用for循环表示图例
colors = ['red', 'black', 'orange']
iris.target_names
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(x_dr[y == i, 0]
               ,x_dr[y == i, 1]
               ,alpha=.7
               ,c=colors[i]
               ,label=iris.target_names[i]
               )
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()

属性: explained_variance :查看降维后新特征向量上的所带的信息量的大小 explained_variance_ratio_ : 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比

#属性explained_variance ,查看降维后新特征向量上的所带的信息量的大小
pca.explained_variance_

pca.explained_variance_ratio_
#第一个特征带有原有数据的92%，第二个特征带有5%

#新特征矩阵总信息占比
pca.explained_variance_ratio_.sum()

```

当参数n_components 中不填写任何值，则默认返回min（X.shape）， 所以什么都不填相当于转换了新特征空间，但没有减少特征的个数。一般来说，不会使用这种输入方式。

**但我们却可以使用这种输入方式来画出累计可解释方差贡献率曲线，以此选择最好的n_components的整数取值。**

- 累积可解释方差贡献率曲线 一条以降维后保留的特征个数为横坐标，降维后新特征矩阵捕捉到的可解释方差贡献率为纵坐标的曲线，能够帮助我们决定n_components最好的取值。

```python
pca_line = PCA().fit(x)
pca_line.explained_variance_ratio_
#四个特征占原始数据的百分比，选几个特征拿到多少原数据的信息，
#帮助我们选择n_components选几

#方差贡献率曲线
import numpy as np
pca_line = PCA().fit(x)
plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1,2,3,4]) #这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratdio")
plt.show()

```

### 最大似然估计自选超参数法
让PCA用最大似然估计自选超参数的方法，输入mle作为n_components的参数估计

```
pca_mle = PCA(n_components='mle')
pca_mle = pca_mle.fit(x)
x_mle =pca_mle.transform(x)
x_mle
#自动选择了三个特征

pca_mle.explained_variance_ratio_.sum()


```

### 按信息量占比选超参数

输入[0,1]之间的浮点数，并让svd_solver == 'full'

**表示希望降维后的总解释性方差占比大于n_components指定的百分比**

> 即是说，希望保留百分之多少的信息量。比如说，如果我们希望保留97%的信息量，就可以输入n_components = 0.97，PCA会自动选出能够让保留的信息量超过97%的特征数量。

```
pca_f = PCA(n_components=0.97,svd_solver="full")
pca_f = pca_f.fit(x)
x_f = pca_f.transform(x)
pca_f.explained_variance_ratio_

x_f
#自动的选择了二维

pca_f.explained_variance_ratio_.sum()
```



### PCA中的SVD

SVD有一种惊人的书写性质，它可以不计算协方差矩阵，直接找出一个新特征向量组成的n维空间

**这是说，奇异值分解可以不计算协方差矩阵等等结构复杂计算冗长的矩阵，就直接求出新特征空间和降维后的特征矩阵。**

* SVD在矩阵分解中的过程比PCA简单快速
* SVD的信息量衡量指标比较复杂 ， SVD的衡量指标是奇异值

因此，sklearn将降维流程拆成了两部分：

一部分是计算特征空间V，由奇异值分解完成，
另一部分是映射数据和求解新特征矩阵。

由主成分分析完成，由SVD的性质减少计算量，却让信息量评估指标是方差





#### svd_solver 的参数：

* auto : 基于X.shape 和 n_components的默认策略来选择分解器：如果输入的提取的特征数小于数据的最小维度min(X.shape)的80% ，就启用randomized去计算。截断会在矩阵被分解完成之后有选择地发生
* full :  从scipy.linalg.svd 种调用标准的LAPACK分解器来生成完整的SVD，分解器根据原数据和输入的n_component值去寻找符合需求的新特征向量，适合数据量适中，计算时间充足的情况
* arpack : 从scipy.linalg.svd 种调用标准的LAPACK分解器来运行截断奇异值分解SVD将特征数量降到输入的n_component值 ，一般用于特征矩阵维稀疏矩阵的情况

* randomized :分解器会随机生成多个随机向量，如果其中的随机向量由符合条件的就保留，并基于保留的随机向量构建后续的新特征向量，适合特征矩阵非常庞大的情况

#### random_state参数 
在svd_solver的值为arpack 或randomized 的时候生效，通常选auto



实战代码一样，加入参数就ok

```
pca_f = PCA(n_components=0.97,svd_solver="full")
pca_f = pca_f.fit(x)
x_f = pca_f.transform(x)
```



## SVD

SVD使用奇异值分解来找出空间V，其中Σ也是一个对角矩阵，不过它对角线上的元素是奇异值，这也是SVD中用
来衡量特征上的信息量的指标。U和V^{T}分别是左奇异矩阵和右奇异矩阵，也都是辅助矩阵。


* 在数学原理中，无论是PCA和SVD都需要遍历所有的特征和样本来计算信息量指标。

并且在矩阵分解的过程之中，会产生比原来的特征矩阵更大的矩阵，
比如原数据的结构是(m,n)，在矩阵分解中为了找出最佳新特征空间V，可能
需要产生(n,n)，(m,m)大小的矩阵，还需要产生**协方差矩阵去计算更多的信息。**

而现在无论是Python还是R，或者其他的任何语言，在大型矩阵运算上都不是特别擅长，无论代码如何简化，我们不可避免地要等待计算机去完成这个非常庞大的数学计算过程。

因此，降维算法的计算量很大，运行比较缓慢，但无论如何，它们的功能无可替代，
它们依然是机器学习领域的宠儿。

## 降维和特征选择都是特征工程技术 ，它们有上面不同

特征工程有三种方式: 特征提取，特征创造和特征选择。

>特征选择
>是从已存在的特征中选取携带信息最多的，选完之后的特征依然具有可解释性，我们依然知道这个特
>征在原数据的哪个位置，代表着原数据上的什么含义。

---

>降维算法
>是将已存在的特征进行压缩，降维完毕后的特征不是原本的特征矩阵中的任何一个特征，而是通
>过某些方式组合起来的新特征。通常来说，在新的特征矩阵生成之前，我们无法知晓降维算法们都建立了怎样
>的新特征向量，**新特征矩阵生成之后也不具有可读性**，我们无法判断新特征矩阵的特征是从原数据中的什么特征组合而来，新特征虽然带有原始数据的信息，却已经不是原数据上代表着的含义了,就丢失了原有数据之间上的联系。

降维算法因此是特征创造（feature creation，或feature construction）的一种。



```
sklearn.decomposition.PCA(
n_component= 降维后需要的维度（特征数量），一般数据要降到3维以下
,copy=  
,whiten= 
,svd_solver=
,iterated_power = 'auto' 
,random_state = )
```



```python
#数据可视化，但是4维数据画不了，需要降维

pca = PCA(n_components=2)

pca = pca.fit(x) #放入拟合模型

x_dr = pca.transform(x) #获取新的矩阵

#y有三种值，0，1，2
# 两列特征向量，如何画出三种鸢尾花的点呢？
#我们要取出三种不同的点
x_dr [y==0,0] 
#使用布尔索引，y==0返回布尔索引 True和False ，再取出第一列，就是x坐标
#具体含义就是 取出标签为0的行，第一列

#具体取出三种花的x，y轴的数据
x_dr[y==0,0] ,x_dr[y==0,1]
x_dr[y==1,0] ,x_dr[y==1,1]
x_dr[y==2,0] ,x_dr[y==2,1]


plt.figure()
plt.scatter(x_dr[y==0, 0], x_dr[y==0, 1], c="red", label=iris.target_names[0])
plt.scatter(x_dr[y==1, 0], x_dr[y==1, 1], c="black", label=iris.target_names[1])
plt.scatter(x_dr[y==2, 0], x_dr[y==2, 1], c="orange", label=iris.target_names[2])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()

```

 ![img](data:image/png;base64,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RrBkyRLq6+sHvO7VV19l7ty5B6p1vv3tbw94fujQodx3331cffXV7Nmzh+HDh7NixQrmz5/PvHnzmDBhAkOGDOHuu+/mXe9614DXLliwgLlz5zJx4kRisVjReuhbe2RjTMlYe+Rw5doe2a60NcaYGmEB3xhjaoQFfGOMqREW8I0xpkZYwDfGmBphAd+YYtiYgPvHwT11zv1G6w1kys8CvjFh25iAVa3Q2wWoc7+q1YJ+RG3dupWLLroo59fNnDmTN954w3effFsnF4vV4RsTtvvHJYN9hlgzXNhZ6tFESs51+BsTsKYNejdBrAkmtUNLaS7K6+vrY8iQaK8Ca3X4xpRbr0dzL6/txl0Rvil5tUceP348AHfffTcXX3wxn/70pznrrLPo7e3ls5/9LBMnTuRzn/scp5xyCqmT0HHjxrF9+/YDbZSvuOIKTjjhBM466yz27NkDZG+d3NnZyUc/+lGmTJnClClT+P3vf5/37xaEBXxjwhbzaO7ltd24W9MG+zOa6+3vdbbnadasWdx3330HHi9btuxAd8uUlStXsnjxYh555BHuuOOOAx01v/a1r/H000+7vu8rr7zCl770JV588UUOP/xwfvKTnwx4fu/evXzuc59j4cKFrFmzhhUrVjB8+HDe85738Jvf/IZnnnmG++67j2uuuSbv3y2IaH9fMaYSTWp3zkTTg1V9zNlugivCN6X09sjd3d0cccQRg7pvzpgxgyOPPBJw2hZfe+21AIwfP56JEye6vm9LSwsnnngiACeddBKdnZ0DnndrnQzw1ltvcdVVV/Hss89SX18/qG1y2OwM31SEqCx1F0hLHKZ1ODl7xLmf1lGy3HPVKNI3pVR75Pvuu49Zs2YNej69HXHQOc70Zmj19fWDliP0ap18yy238N73vpc1a9awevVq9u7dG/TXyIud4ZvISy11l+qdn1rqDohuV82WuAX4QhXpm9KsWbO44oor2L59O48//jjvvPOO574f+chHWLZsGR//+Md56aWXeP755/M6Znrr5JNPPpldu3YxfPhwdu7cydixY6mrq2Px4sXs378/318rEDvDN5Hntv5tb28vbW3553JNBSjSN6XM9sh+5s+fT3d3NxMnTuS73/0uEydO5N3vfnfOx0xvnTxp0iRmzJjB22+/zfz581m8eDHTp0/n5ZdfDnWxEzdWlmkir66uznNFoFRvclMZKq098v79+9m3bx/Dhg3jz3/+M5/85Cd5+eWXGTp0aLmHBuRelmkpHRN5xVrqzphsent7+fjHP86+fftQVe68887IBPt8WMA3kee2/q0tg2hK4dBDD6WasgyWwzeRl1r/trm5GRGhubmZjo6O6E7YGl9RTiNXknz+He0M31SE1Pq3prINGzaMnp4eGhsbXcsUTTCqSk9PD8OGDcvpdRbwjTElM3bsWLZs2UJ3d3e5h1Lxhg0bxtixY3N6jQV8Y0zJNDQ00NLSUu5h1CzL4RtjTI2wgG+MMTXCAr4xxtSISF9pKyLdgMtKEqEZCWwv4vuHrZLGW0ljhcoar421eCppvF5jbVbVUW4viHTALzYRWe11CXIUVdJ4K2msUFnjtbEWTyWNN5+xWkrHGGNqhAV8Y4ypEbUe8DvKPYAcVdJ4K2msUFnjtbEWTyWNN+ex1nQO3xhjakmtn+EbY0zNsIBvjDE1ouYDvoh8U0SeE5FnReTXInJUucfkR0RuFpE/Jcf8MxE5vNxj8iIiF4vIiyLSLyKRLHUTkXNEZJ2IrBeRG8o9Hj8iskhEtonIC+UeSzYicrSIPCoia5N/A9eWe0xeRGSYiKwSkTXJsf5juceUjYjUi8gfReR/cnldzQd84GZVnaiqJwL/A3y9zOPJ5jfAeFWdCLwM3Fjm8fh5Afg74IlyD8SNiNQDtwOfAo4HLhGR48s7Kl93A+eUexAB9QH/W1WPA6YDX4rwv+07wCdUdRJwInCOiEwv75CyuhZYm+uLaj7gq+qbaQ8PASI9i62qv1bVvuTDPwC59UctIVVdq6rryj0OH9OA9aq6QVX3AkuBC8o8Jk+q+gSwo9zjCEJVX1PVZ5I/78IJTmPKOyp36tidfNiQvEU2DojIWOBc4Ie5vrbmAz6AiLSLyGYgTvTP8NNdDvyy3IOoYGOAzWmPtxDRoFTJRGQcMBl4ssxD8ZRMkTwLbAN+o6qRHStwK/APQH+uL6yJgC8iK0TkBZfbBQCq2qaqRwMJ4Kryjjb7eJP7tOF8bU6Ub6TBxhphbksuRfbMrhKJyAjgJ8CXM75NR4qq7k+mdccC00RkfJmH5EpEzgO2qerT+by+JhZAUdUzA+56D/AL4KYiDierbOMVkdnAecAntcwXUuTwbxtFW4Cj0x6PBbaWaSxVR0QacIJ9QlV/Wu7xBKGqb4jIYzhzJVGcHD8NOF9EZgLDgMNEZImqXhrkxTVxhu9HRI5Je3g+8KdyjSUIETkHuB44X1V7yz2eCvcUcIyItIjIUGAW8ECZx1QVxFmw9j+Btar6r+Uejx8RGZWqdhOR4cCZRDQOqOqNqjpWVcfh/L0+EjTYgwV8gO8kUxDPAWfhzH5H2W3AocBvkqWk3y/3gLyIyN+KyBbgVOAXIvJQuceULjn5fRXwEM6k4jJVfbG8o/ImIvcCK4EPisgWEflf5R6Tj9OAy4BPJP9On02elUbRaODRZAx4CieHn1O5Y6Ww1grGGFMj7AzfGGNqhAV8Y4ypERbwjTGmRljAN8aYGmEB3xhjaoQFfGOMqREW8I0xpkb8f+g5eryS6xJwAAAAAElFTkSuQmCC) 

* 属性:
  explained_variance :查看降维后新特征向量上的所带的信息量的大小
  explained_variance_ratio_ : 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比

```
#属性explained_variance ,查看降维后新特征向量上的所带的信息量的大小
pca.explained_variance_
#新特征矩阵总信息占比
pca.explained_variance_ratio_.sum()
```

###  n_components参数

当参数n_components 中不填写任何值，则默认返回min（X.shape），
 所以什么都不填相当于转换了新特征空间，但没有减少特征的个数。一般来说，不会使用这种输入方式。

**但我们却可以使用这种输入方式来画出累计可解释方差贡献率曲线，以此选择最好的n_components的整数取值。**

* 累积可解释方差贡献率曲线
  一条以降维后保留的特征个数为横坐标，降维后新特征矩阵捕捉到的可解释方差贡献率为纵坐标的曲线，能够帮助我们决定n_components最好的取值。

```
方差贡献率曲线
import numpy as np
pca_line = PCA().fit(x)
plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1,2,3,4]) #这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratdio")
plt.show()
```

 ![img](data:image/png;base64,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) 

### 最大似然估计自选超参数法

让PCA用最大似然估计自选超参数的方法，输入mle作为n_components的参数估计

```python
pca_mle = PCA(n_components='mle')
pca_mle = pca_mle.fit(x)
x_mle =pca_mle.transform(x)
pca_mle.explained_variance_ratio_.sum()
```



### 按信息量占比选超参数

输入[0,1]之间的浮点数，并让svd_solver == 'full'

**表示希望降维后的总解释性方差占比大于n_components指定的百分比**

>即是说，希望保留百分之多少的信息量。比如说，如果我们希望保留97%的信息量，就可以输入n_components = 0.97，PCA会自动选出能够让保留的信息量超过97%的特征数量





### PCA种的SVD

SVD有一种惊人的书写性质，它可以不计算协方差矩阵，直接找出一个新特征向量组成的n维空间

**这是说，奇异值分解可以不计算协方差矩阵等等结构复杂计算冗长的矩阵，就直接求出新特征空间和降维后的特征矩阵。**

* SVD在矩阵分解中的过程比PCA简单快速
* SVD的信息量衡量指标比较复杂 ， SVD的衡量指标是奇异值

因此，sklearn将降维流程拆成了两部分：

一部分是计算特征空间V，由奇异值分解完成，
另一部分是映射数据和求解新特征矩阵。

由主成分分析完成，由SVD的性质减少计算量，却让信息量评估指标是方差


![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C5.png)


#### svd_solver 的参数：

* auto : 基于X.shape 和 n_components的默认策略来选择分解器：如果输入的提取的特征数小于数据的最小维度min(X.shape)的80% ，就启用randomized去计算。截断会在矩阵被分解完成之后有选择地发生
* full :  从scipy.linalg.svd 种调用标准的LAPACK分解器来生成完整的SVD，分解器根据原数据和输入的n_component值去寻找符合需求的新特征向量，适合数据量适中，计算时间充足的情况
* arpack : 从scipy.linalg.svd 种调用标准的LAPACK分解器来运行截断奇异值分解SVD将特征数量降到输入的n_component值 ，一般用于特征矩阵维稀疏矩阵的情况

* randomized :分解器会随机生成多个随机向量，如果其中的随机向量由符合条件的就保留，并基于保留的随机向量构建后续的新特征向量，适合特征矩阵非常庞大的情况

#### random_state参数 

在svd_solver的值为arpack 或randomized 的时候生效，通常选auto



## 逻辑回归

>名为回归的分类器，逻辑回归是一种名为回归的线性分类器，其本质是由线性回归变化而来的

* 通过最小二乘法来求解线性回归中的参数，k和b

Sigmoid函数将任何实数映射到（0,1）之间，因为这个性质，Sigmoid函数也被当作是归一化的一种方法，与我们之前学过的MinMaxSclaer同理

**和MinMaxSclaer归一化处理的区别：能否取到0和1，Sigmoid不可以**

* 逻辑回归就是 对数几率回归 
  g(z)的形似几率取对数的本质其实就是我们的线性回归z，我们实际上是在对线性回归模型的预测结果取
  对数几率来让其的结果无限逼近0和1。因此，其对应的模型被称为”对数几率回归“（logistic Regression）

* 预测值代表了某一类标签的概率吗？
  逻辑回归返回的值带有概率的性质，所以认定逻辑回归返回的是概率

* 为什么需要逻辑回归？
  线性回归对数据的要求很严格，比如标签必须满足正态分布，特征之间的多重共线性需要消除等等，而现实中很多真实情景的数据无法满足这些要求，
  **因此线性回归在很多现实情境的应用效果有限。**

逻辑回归是由线性回归变化而来，因此它对数据也有一些要求，而我们之前已经学过了强大的分类模型决策树和随机森林，它们的分类效力很强，并且不需要对数据做任何预处理。

### 优点：

1.逻辑回归对线性关系的拟合效果非常非常的好，比如特征和标签之间的线性关系极强的数据，但是非线性回归不太行
2.逻辑回归计算快：对于线性数据，逻辑回归的拟合和计算非常的快
3.逻辑回归返回的分类结果本身不是固定的0，1 ，而是0到1之间的小数（概率数字），因此我们可以把逻辑回归返回的结果当成连续性数据处理
4.逻辑回归还有抗噪能力强的优点，逻辑回归在小数据集上表现更好，在大型的数据集上，树模型有着更好的表现。

### 参数：

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C6.png)

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C7.png)





### 损失函数

衡量参数 的优劣的评估指标，用来求解最优参数的工具

损失函数小，模型在训练集上表现优异，拟合充分，参数优秀

损失函数大，模型在训练集上表现差劲，拟合不足，参数糟糕

我们追求，能够让损失函数最小化的参数组合

**注意：没有”求解参数“需求的模型没有损失函数，比如KNN，决策树这些模型不用求解参数，直接输入超参数就好了**

逻辑回归的损失函数是由最大似然法来推导出来的，具体结果可以写作：
![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C11.png)

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C15.png)

---

如果模型在训练集上表示优秀，却在测试集上表现糟糕，模型就会过拟合。虽然逻辑回归和线性回归是天生欠拟合

的模型，但我们还是需要控制过拟合的技术来帮助我们调整模型，对逻辑回归中过拟合的控制，通过正则化来实现

---

### 正则化

> 正则化是用来防止模型过拟合的过程，常用的有L1正则化和L2正则化两种选项，分别通过在损失函数后加上参数向量 c塔 的L1范式和L2范式的倍数来实现，

* L1范式
  L1范数表现为参数向量中的每一个参数的绝对值之和
* L2范式
  L2范数表现为参数向量中的每个参数的平方和的开方值

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C12.png)

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C13.png)



其实是一样的，至于常数项C，是在损失函数的前面还是在正则项前面，都一样，C表示惩罚的力度
![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C14.png)

#### L1正则化和L2正则化的区别

L1正则化和L2正则化虽然都可以控制过拟合，但它们的效果并不相同。
当正则化强度逐渐增大（即C逐渐变小），参数c塔的取值会逐渐变小，**但L1正则化会将参数压缩为0，L2正则化只会让参数尽量小，不会取到0**

#### L1正则化

L1正则化的本质是一个特征选择的过程，掌管了参数的稀疏性。L1正则化越强，参数向量中就越多的参数为0，参数就越稀疏，选出来的特征就越少，以此来防止过拟合

> 特征量很大，数据维度很高，我们会倾向于使用L1正则化。由于L1正则化的这个性质，逻辑回归的特征选择可以由
> Embedded嵌入法来完成

#### L2正则化

L2正则化在加强的过程中，**会尽量让每个特征对模型都有一些小的贡献**，但携带信息少，对模型贡献不大的特征的参数会非常接近于0。

* 如果我们的主要目的只是为了防止过拟合，选择L2正则化就足够了


而两种正则化下C的取值，都可以通过学习曲线来进行调整。

### 实战代码

```
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data =  load_breast_cancer()
x = data.data
y = data.target

lrl1 = LR(penalty="l1",solver="liblinear",max_iter=1000)
lrl2 = LR(penalty="l2",solver="liblinear",max_iter=1000)

lrl1 = lrl1.fit(x,y)
#查看L1正则化后各个特征的参数
lrl1.coef_

#查看筛选后的特征数
(lrl1.coef_!=0).sum(axis=1) 

lrl2 = lrl2.fit(x,y)
lrl2.coef_
#可以看到，L2没有为0的，因为L2就是尽量让每一个数据发挥一点作用
```



### 调参

究竟哪个正则化的效果更好呢？还是都差不多？

画条曲线

* accurary_score(预测的值，真实值),看看预测的数值和真实的值差的有多大，打个分

```python
l1 = []
l2 = []
l1test = []
l2test = []

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=420)

for i in np.linspace(0.05,1,19):
    lrl1 = LR(penalty="l1",solver="liblinear",C=i,max_iter=1000)
    lrl2 = LR(penalty="l2",solver="liblinear",C=i,max_iter=1000)
    
    lrl1 = lrl1.fit(xtrain,ytrain)
    l1.append(accuracy_score(lrl1.predict(xtrain),ytrain))
    l1test.append(accuracy_score(lrl1.predict(xtest),ytest))
    
    lrl2 = lrl2.fit(xtrain,ytrain)
    l2.append(accuracy_score(lrl2.predict(xtrain),ytrain))
    l2test.append(accuracy_score(lrl2.predict(xtest),ytest))
    
graph = [l1,l2,l1test,l2test]
color = ["green","black","lightgreen","gray"]
label = ["L1","L2","L1test","L2test"] 

plt.figure(figsize=(6,6))
for i in range(len(graph)):
     plt.plot(np.linspace(0.05,1,19),graph[i],color[i],label=label[i])
plt.legend(loc=4) #图例的位置在哪里?4表示，右下角
plt.show()

#横坐标就是C的值，表示对正则化项的惩罚力度
```

 ![img](data:image/png;base64,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) 

### 逻辑回归中的特征工程

* PCA和SVD一般不用
* 统计方法可以使用，但不是非常必要

* 高效的嵌入法embedded

sklearn.feature_selection .FeatureFromModel
(模型，筛选指标，norm_order = 使用L1范式还是L2)

```python
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

data = load_breast_cancer()

LR_ =LR(solver="liblinear",C=0.9,random_state=420)
score = cross_val_score(LR_,data.data,data.target,cv=10).mean()

score

#嵌入法降维,通过L1范式降维，只剩下9个特征了
x_embeded = SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
x_embeded.shape

#所以的系数，系数越大，对逻辑回归贡献越高
abs(LR_.fit(data.data,data.target).coef_.max())

-----------------------------------------------------------
#特征选择后再次运行

fullx =[]#完整的特征矩阵
fsx =[]#特征选择后的矩阵
threshold = np.linspace(0,abs(LR_.fit(data.data,data.target).coef_).max(),20)

k=0
for i in threshold:
    #特征选择后的矩阵
    x_embeded =SelectFromModel(LR_ ,threshold=i,norm_order=1).fit_transform(data.data,data.target)
    fullx.append(cross_val_score(LR_ ,data.data,data.target,cv=5).mean())
    #特征选择后的交叉验证结果
    fsx.append(cross_val_score(LR_ ,x_embeded,data.target,cv=5).mean())
    print((threshold[k],x_embeded.shape[1]))
    #打印threhold参数,对应的特征选择剩下的特征数
    k+=1 #k是循环threhold的取值
    
plt.figure(figsize=(20,5))
plt.plot(threshold,fullx,label="full")
plt.plot(threshold,fsx,label="feature selection")
plt.xticks(threshold)
plt.legend()
plt.show()

-----------------------------------------------------------------------
#可以看见我们的threhold参数的范围太大了，我们应该缩小范围

fullx =[]#完整的特征矩阵
fsx =[]#特征选择后的矩阵
threshold = np.linspace(0,0.107,20)
​
k=0
for i in threshold:
    #特征选择后的矩阵
    x_embeded =SelectFromModel(LR_ ,threshold=i,norm_order=1).fit_transform(data.data,data.target)
    fullx.append(cross_val_score(LR_ ,data.data,data.target,cv=5).mean())
    #特征选择后的交叉验证结果
    fsx.append(cross_val_score(LR_ ,x_embeded,data.target,cv=5).mean())
    print((threshold[k],x_embeded.shape[1]))
    #打印threhold参数,对应的特征选择剩下的特征数
    k+=1 #k是循环threhold的取值
    
plt.figure(figsize=(20,5))
plt.plot(threshold,fullx,label="full")
plt.plot(threshold,fsx,label="feature selection")
plt.xticks(threshold)
plt.legend()
plt.show()
------------------------------------------------------------------

#由图像可知，取到0.00225的时候效果最好，这时候特征矩阵还有24个特征 但是对于显示来说，意义不大，从30个特征#降到24个，区别不大

fullx =[]#完整的特征矩阵
fsx =[]#特征选择后的矩阵

C = np.arange(0.01,10.01,0.5)
for i in C:
    #每一次循环都要循环建模
    LR_ = LR(solver="liblinear",C=i,random_state=420)
    fullx.append(cross_val_score(LR_ ,data.data,data.target,cv=10).mean())
    #特征选择后的交叉验证结果
    x_embeded =SelectFromModel(LR_ ,norm_order=1).fit_transform(data.data,data.target)
    fsx.append(cross_val_score(LR_ ,x_embeded,data.target,cv=10).mean())
    #打印最高分数，和对应的C取值
    print(max(fsx),C[fsx.index(max(fsx))])
    
plt.figure(figsize=(20,5))
plt.plot(C,fullx,label="full")
plt.plot(C,fsx,label="feature selection")
plt.xticks(C)
plt.legend()
plt.show()
------------------------------------------------------------
我们可以发现，选择特征之后，效果有超过原来的

接下来，细化C的取值范围

fullx =[]#完整的特征矩阵
fsx =[]#特征选择后的矩阵

C = np.arange(6.05,7.05,0.005)
for i in C:
    #每一次循环都要循环建模
    LR_ = LR(solver="liblinear",C=i,random_state=420)
    fullx.append(cross_val_score(LR_ ,data.data,data.target,cv=10).mean())
    #特征选择后的交叉验证结果
    x_embeded =SelectFromModel(LR_ ,norm_order=1).fit_transform(data.data,data.target)
    fsx.append(cross_val_score(LR_ ,x_embeded,data.target,cv=10).mean())
#打印最高分数，和对应的C取值
print(max(fsx),C[fsx.index(max(fsx))])
    
plt.figure(figsize=(20,5))
plt.plot(C,fullx,label="full")
plt.plot(C,fsx,label="feature selection")
plt.xticks(C)
plt.legend()
plt.show()
----------------------------------------------------------------
'''
我们可以看到很多次都超过了原来的模型，我们就可以选取最高的，作为我们的C的取值

接下来对比一下降维前后的分数

可以发现啊，还是提升了'''

LR_ = LR(solver="liblinear",C=6.079999999999999,random_state=420)
cross_val_score(LR_ , data.data,data.target,cv=10).mean

LR_ = LR(solver="liblinear",C=6.079999999999999,random_state=420)
x_embeded = SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
cross_val_score(LR_ ,x_embeded,data.target,cv=10).mean()

#分数高，而且维度低，这不香吗？
x_embeded.shape
```

### 梯度下降法

> 原理：让损失函数最小，规定步长和方向，找到最低点

![1644024622316](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644024622316.png)

找到最低点的c塔1 和c塔2 就可以找到了最小的损失函数的点

sklearn中逻辑回归的参数max_iter ：代表设置的最大步数

#### 误区
一个多元函数的梯度，是对其自变量求偏导的结果。但是在逻辑回归中，损失哈桑农户的自变量刚好是逻辑回归的预测函数y（x）的参数，所以造成了误解

正解：在多元函数(损失函数)上对自变量(逻辑回归的预测函数y(x)的预测函数)求偏导 ，**求偏导就是对损失函数上的c塔1和c塔2 求偏导**



### 二元回归和多元回归：重要参数solver&multi_class

* ovr :一对多 ，一种类型为1，其他类型为0
* Multinomial : 多对多，几个类型为1，剩下的几个类型为0
* auto : 自动 如果数据是二分类，或solver的取值是“liblinear”选ovr，反之，选multinomial

每种类型都可以配合L1和L2正则项来使用

solver有五种类型：

![1644024663544](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644024663544.png)

看看鸢尾花数据集上，multinomial 和 ovr 的区别

多对多和一对多的区别

```python
from sklearn.datasets import load_iris
iris = load_iris()

#打印多对多，一对多的分数
for multi_class in ('multinomial','ovr'):
        lr = LR(solver='sag',max_iter=100,random_state=42,
               multi_class=multi_class).fit(iris.data,iris.target)
        print("training score:%.3f(%s)"  % (lr.score(iris.data,iris.target),multi_class))
        
#看一下分数，多对多比较好   
```

### 样本不平衡的问题
>样本不平衡是指在一组数据集中，标签的一类天生占有很大的比例，或误分类的代价很高

因此，我们要使用class_weight 对样本进行一定的均衡，给少量的标签更多的权重，模型向少数类的方向偏移。默认为None，即相同的权重

banlance：平衡模式

* 采样法：通过重复样本的方式来平衡标签，可以进行上采样（增加少数类的样本）或下采样（减少多数类的样本），对于逻辑回归，上采样是最好的方法



##  朴素贝叶斯：

朴素贝叶斯是一种直接衡量标签和特征之间的概率关系的有监督算法，它既可以做回归也可以分类，不过多是用于分类之中。

朴素贝叶斯是一个不建模的算法。

**算法得出的结论，永远不是100%确定的，更多的是判断出了一种“样本的标签更可能是某类的可能性”，而非一种“确定”。**

每种算法使用不同的指标来衡量这种可能性。比如说，决策树使用的就是叶子节点上占比较多的标签所占的比例
（接口predict_proba调用），逻辑回归使用的是sigmoid函数压缩后的似然（接口predict_proba调用），而SVM
使用的是样本点到决策边界的距离（接口decision_function调用）。但这些指标的本质，其实都是一种“类概率”的
表示，我们可以通过归一化或sigmoid函数将这些指标压缩到0~1之间，让他们表示我们的模型对预测的结果究竟
有多大的把握（置信度）。

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C51.png)

#### 高斯朴素贝叶斯分布：

sklearn.naive_bayes.GaussianNB (priors=None, var_smoothing=1e-09)

高斯朴素贝叶斯，通过假设 是服从高斯分布（也就是正态分布），来估计每个特征下每个类别上的条件概率。

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C52.png)

高斯朴素贝叶斯以最大化 P(X|Y)为目标，来求解该式中的参数  和 。求解出参数后，带入一个 x的值，就能得到相应的概率取值.

* prior : 
  可输入任何类数组结构，形状为（n_classes，）.表示类的先验概率。如果指定，则不根据数据调整先验，如果不指定，则自行根据数据计算先验概率 

* var_smoothing :浮点数，可不填（默认值= 1e-9）,在估计方差时，为了追求估计的稳定性，将所有特征的方差中最大的方差以某个比例添加到估计的方差中。这个比例，由var_smoothing参数控制

#### 伯努利贝叶斯：

 ```sklearn.naive_bayes.BernoulliNB (alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)```

#### 多项式朴素贝叶斯：

```sklearn.naive_bayes.MultinomialNB (alpha=1.0, fit_prior=True, class_prior=None)```

#### 补集贝叶斯：

``` sklearn.naive_bayes.MultinomialNB (alpha=1.0, fit_prior=True, class_prior=None)```

### 概率类模型的评估指标：

sklearn.metrics.brier_score_loss
BS(y_true, y_prob, *, sample_weight=None, pos_label=None)

 * 布利尔指标：

 > 概率预测的准确程度被称为“校准程度”，是衡量算法预测出的概率和真实概率的差异的一种方式，在二分类中，最常用的指标叫做布里尔分数，它被计算为是概率预测相对于测试样本的均方误差

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C53.png)

其中N是样本数量， 为朴素贝叶斯预测出的概率， 是样本所对应的真实结果，只能取到0或者1，如果事件发生则为1，如果不发生则为0。这个指标衡量了我们的概率距离真实标签结果的差异，其实看起来非常像是均方误差。

**布里尔分数的范围是从0到1，分数越高则贝叶斯的预测结果越差劲**

## 使用混淆矩阵来查看分类结果

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C54.png)

* sklearn.metrics.confusion_matrix

混淆矩阵是二分类问题的多维衡量指标体系，在样本不平衡时及其有用。
少数类是正例，多数类是负例。
![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C55.png)

真实值在前面，预测值在后。对角线的11，00就是预测正确的。

#### 准确率：

橙色表示分母，绿色表示分子

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C56.png)

**准确率就是所有预测正确的样本除以总样本，越接近1越好**

#### 精确率：

> 又叫查准率，表示所有被我们预测的样本中，真正的少数类所占的比例。
> ![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C57.png)

精确度形象的表示为，决策边界上方的所有点中，红色点所占的比例，红色比例越高，精确度越高。
![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C58.png)

`True =1,预测对的除以所有的
(y[y == clf.predict(x)]==1).sum()/(clf.predict(x)==1).sum()`

精确度是将多数类分错后所需付出的成本

#### 召回率：

> 所有样本真实为1的样本中，被我们预测为正确的样本的比例。
> 召回率可表示为，决策边界上方的所有红色点占全部样本中红色点的比例

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C59.png)

召回率越高，代表我们尽量捕获出了越多的少数类，召回率越低，代表我们没有补货出足够的少数类。

`#predict为1 的点 / 全部为1的点的比例
(y[y == clf.predict(x)]==1).sum()/(y==1).sum()`

#### 假正率
随着Recall的增加，Precision会下降，我们希望理解我每判断正确一个少数类，就有多少个多数类会被判断错误。假正率来帮助我们衡量这个能力的变化。
因此，我们可以使用Recall和Precison之间的平衡，让我们衡量模型再尽量捕捉少数类的时候误差多数类的情况，这就是ROC曲线的平衡。
* ROC曲线：以假正率FPR为横坐标，不同阈值下的召回率为纵坐标的曲线

**基于概率！！！**

```python
#predict为1 的点 / 全部为1的点的比例
(y[y == clf.predict(x)]==1).sum()/(y==1).sum()

from sklearn.metrics import brier_score_loss as BS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X, y = digits.data, digits.target
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y
                        ,test_size=0.3,random_state=420)

#建模
gnb = GaussianNB().fit(Xtrain,Ytrain)
#查看分数
acc_score = gnb.score(Xtest,Ytest)
#查看预测结果
Y_pred = gnb.predict(Xtest)
#查看预测的概率结果
prob = gnb.predict_proba(Xtest)

prob.shape

#混淆矩阵和精确性可以帮助我们了解贝叶斯的分类结果
from sklearn.metrics import confusion_matrix as CM
CM(Ytest,Y_pred)

#使用布里尔系数进行评估
from sklearn.metrics import brier_score_loss
brier_score_loss(Ytest, prob[:,0], pos_label=0)
#我们的pos_label与prob中的索引一致，就可以查看这个类别下的


```

#### 概率与阈值

```python

from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt
class1 = 7
class2 = 4
centers = [[0.0,0.0],[1,1]]
clusters_std = [0.5,1]
X,y = make_blobs(n_samples=[class1,class2]
                 ,centers=centers
                 ,cluster_std=clusters_std
                 ,random_state=5)
plt.scatter(X[:,0],X[:,1],c=y,cmap="rainbow",s=30)
plt.show()
```

 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXQAAAD7CAYAAAB68m/qAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAU60lEQVR4nO3df5BdZZ3n8fc36U5IQpjApBXID6KSZQbYQaA3Bp2SrKILCIPj4Gzc4YeMZVb8UVI7fyyLW6hs7ay4s1qFqNmUZIQpgXL8waSsZJBx+DksDJ0YwBBhGwYhSyQtaEJIJOnku3+cm6Lpvp2+ndzu2/fp96vqVp/znOfe+z2czqcPz33OuZGZSJLa35RWFyBJag4DXZIKYaBLUiEMdEkqhIEuSYUw0CWpECMGekQcERH/HBGPRsSmiPhinT4RETdERG9EPBYRZ4xNuZKk4XQ00Oc14D2ZuTMiOoEHImJdZj40oM95wOLa4x3AN2s/JUnjZMRAz+rKo5211c7aY/DVSBcBt9T6PhQRcyLiuMzcOtzrzp07NxctWnRoVUvSJLV+/fpfZWZXvW2NnKETEVOB9cCJwNcz8+FBXeYBzw9Y31Jre0OgR8QKYAXAwoUL6enpaWgHJEmViPjFcNsa+lA0M/dl5tuB+cCSiDh18HvUe1qd11mVmd2Z2d3VVfcPjCTpEI1qlktm/ga4Bzh30KYtwIIB6/OBFw6nMEnS6DQyy6UrIubUlmcA5wA/H9RtDXBZbbbLUmD7wcbPJUnN18gY+nHAzbVx9CnAdzPzRxHxCYDMXAmsBc4HeoFdwBVjVK8kaRiNzHJ5DDi9TvvKAcsJfKq5pUmSRsMrRSVpPPT2wnveAx0dcMwxcN11sH9/U9+ioWmLkqTDsGsXvPOd8NJLVYj/+tdw/fWwbx98ccjF94fMM3RJGmt33AG7d7/xjHzXLvjqV6GJ3xpnoEvSWNu2DfbuHdq+c6eBLklt5ZxzYMqguI2AJUuGth8GA12Sxtqpp8JnPgMzZ0JnJ8yaBXPmwE03NfVt/FBUksbD9dfDn/0Z3HknzJ0LF18Ms2c39S0MdEkaL3/wB9VjjDjkIkmFMNAlqRAGuiQVwkCXpEIY6JJUCANdkgphoEtSIQx0SSqEgS5JhTDQJakQBrokFcJAl6RCGOiSVAgDXZIKYaBLUiFGDPSIWBARd0fE5ojYFBGfrdNnWURsj4iNtce1Y1OuJGk4jXzBRT/wF5m5ISJmA+sj4q7MfGJQv/sz84LmlyhJasSIZ+iZuTUzN9SWXwE2A/PGujBJ0uiMagw9IhYBpwMP19l8VkQ8GhHrIuKUYZ6/IiJ6IqKnr69v9NVKkobVcKBHxJHA94GrMnPHoM0bgBMy8zTga8Ad9V4jM1dlZndmdnd1dR1iyZKkehoK9IjopArz72TmDwZvz8wdmbmztrwW6IyIuU2tVJJ0UI3McgngJmBzZn5lmD7H1voREUtqr/tSMwuVJB1cI7Nc3gVcCjweERtrbdcACwEycyVwMXBlRPQDu4HlmZnNL1eSNJwRAz0zHwBihD43Ajc2qyhJ0uh5pagkFcJAl6RCGOiSVAgDXZIKYaBLUiEMdEkqhIEuSYUw0CWpEAa6JBXCQJekQhjoklQIA12SCmGgS1IhDHRJKoSBLkmFMNAlqRAGuiQVwkCXpEIY6JJUCANdkgphoEtSIQx0SSrEiIEeEQsi4u6I2BwRmyLis3X6RETcEBG9EfFYRJwxNuVKkobT0UCffuAvMnNDRMwG1kfEXZn5xIA+5wGLa493AN+s/ZQkjZMRz9Azc2tmbqgtvwJsBuYN6nYRcEtWHgLmRMRxTa9WkjSsUY2hR8Qi4HTg4UGb5gHPD1jfwtDQJyJWRERPRPT09fWNslRJ0sE0HOgRcSTwfeCqzNwxeHOdp+SQhsxVmdmdmd1dXV2jq1SSdFANBXpEdFKF+Xcy8wd1umwBFgxYnw+8cPjlSZIa1cgslwBuAjZn5leG6bYGuKw222UpsD0ztzaxTknSCBqZ5fIu4FLg8YjYWGu7BlgIkJkrgbXA+UAvsAu4oumVSpIOasRAz8wHqD9GPrBPAp9qVlGSpNHzSlFJKoSBLkmFMNAlqRAGuiQVwkCXpEIY6JJUCANdkgphoEtSIQx0SSqEgS5JhTDQJakQBrokFcJAl6RCTK5A37kTPv5xmDULZs6Eyy+HHYO/fEmS2lMj90Mvx4c/DHffDa+9Vq3ffjv8y7/Affe1ti5JaoLJc4b+3HNwzz2vhznAnj3Q0wNPPdWysiSpWSZPoPf1QWfn0PZp0+DFF8e/HklqsskT6KeeClHni5f6++HMM8e/HklqsskT6NOnw803w4wZcMQR1fqMGbBqVfUBqSS1ucn1oegHP1iNl3/ve7B/P3zoQ7BoUaurkqSmmFyBDjB/Plx1VaurkKSmmzxDLpJUuBEDPSJWR8S2iPjZMNuXRcT2iNhYe1zb/DIlSSNpZMjl28CNwC0H6XN/Zl7QlIokSYdkxDP0zLwPeHkcapEkHYZmjaGfFRGPRsS6iDhluE4RsSIieiKip6+vr0lvLUmC5gT6BuCEzDwN+Bpwx3AdM3NVZnZnZndXV1cT3lqSdMBhB3pm7sjMnbXltUBnRMw97MokSaNy2IEeEcdGVNfUR8SS2mu+dLivK0kanRFnuUTEbcAyYG5EbAE+D3QCZOZK4GLgyojoB3YDyzMzx6xiSVJdIwZ6Zn5khO03Uk1rlCS1kFeKSlIhDHRJKoSBLkmFMNAlqRAGuiQVwkCXpEIY6JJUCANdkgphoEtSIQx0SSqEgS5JhTDQJakQBrokFcJAl6RCGOiSVAgDXZIKYaBLUiEMdEkqhIEuSYUw0CWpEAa6JBXCQJekQhjoklSIEQM9IlZHxLaI+Nkw2yMiboiI3oh4LCLOaH6ZkqSRNHKG/m3g3INsPw9YXHusAL55+GVJkkZrxEDPzPuAlw/S5SLglqw8BMyJiOOaVaAkqTHNGEOfBzw/YH1LrW2IiFgRET0R0dPX19eEt5YkHdCMQI86bVmvY2auyszuzOzu6upqwltLkg5oRqBvARYMWJ8PvNCE15UkjUIzAn0NcFlttstSYHtmbm3C60qSRqFjpA4RcRuwDJgbEVuAzwOdAJm5ElgLnA/0AruAK8aqWEnS8EYM9Mz8yAjbE/hU0yqSJB0SrxSVpEIY6JJUCANdkgox4hi62t9rO2DzD2D3y3DiedD1+62uSNJYMNAL9+Jj8Nfvhv39sH8v/ON/hT/8z3D251tdmaRmc8ilcHdcDq9th72vwr490L8bHvgSvNzb6sokNZuBXrD+31Zn6EMEPPMP416OpDFmoBdsSid0HFGnvQNmeisdqTgGesGmTIXuK6Fz5uttMaVa/1cXtK4uSWPDQC/cOV+Cd3wWps2uwvyEs+FjD0LH9FZXJqnZnOVSuCkd8N6/rB6ZEPVudiypCJ6hTyKGuVQ2A12SCmGgS1IhDHRJKoSBLkmFMNAlqRAGuiQVwkCXpEIY6JJUCANdkgphoEtSIRoK9Ig4NyKejIjeiLi6zvZlEbE9IjbWHtc2v1RJ0sGMeHOuiJgKfB14H7AFeCQi1mTmE4O63p+Z3pRVklqkkTP0JUBvZj6TmXuA24GLxrYsSdJoNRLo84DnB6xvqbUNdlZEPBoR6yLilHovFBErIqInInr6+voOoVxJ0nAaCfR6N13NQesbgBMy8zTga8Ad9V4oM1dlZndmdnd1+R1oktRMjQT6FmDBgPX5wAsDO2TmjszcWVteC3RGxNymVSlJGlEjgf4IsDgi3hIR04DlwJqBHSLi2Ijq6xMiYkntdV9qdrGSpOGNOMslM/sj4tPAncBUYHVmboqIT9S2rwQuBq6MiH5gN7A8MwcPy0iSxlC0Kne7u7uzp6enJe8tSe0qItZnZne9bV4pKkmFMNAlqRAGuiQVor0CfetWuOYauPBC+Ku/gh07Wl3RYdm3Bx78X/C/T4e/fjds/mGrK5LUzkac5TJhPPMMnHkm7NoFe/bAT34C3/gGbNwIRx3V6uoOyW0Xwi/uh/7d1frW9fCb/wZn/afW1iWpPbXPGfq111Zn5Hv2VOu7d8Mvfwnf+lZr6zpEW38Kzz3wepgD7N0F93yhOnOXpNFqn0B/8EHYv/+Nbbt3w733tqaew/SrzRBTh7bv74ddvxr/eiS1v/YJ9JNPHto2fTqcdtr419IEx55ehfdgHdNh1pvGvx5J7a99Av2662DmTIjavcKmTq3WP/nJ1tZ1iLp+H37vj6FzVq0hoHMmvP+rMKV9PtmQNIG0T6CfcQbcfz9ccAG87W1wySWwYQMce2yrKztkH/obuHAVnHgenPKncMmdcPpHW12VpHblpf+S1Ea89F+SJgEDXZIKYaCX6L77YNkyWLgQLrsMnnuu1RVJGgfOpyjN3XdXHxzv2lWt33orrFsHTz4JxxzT2tokjSnP0Evzuc+9HuYA+/bBq6/C6tWtq0nSuDDQS/P000Pbdu+GTZvGvxZJ48pAL83Spa9ffHXArFnVmLqkohnopfnyl6u7T06fXq3PnAknngjLl7e2Lkljzg9FS3PSSfDEE7ByJfz853DOOdVMlwMBL6lYBnqJjj++uveNpEnFIRdJKkRDgR4R50bEkxHRGxFX19keEXFDbftjEXFG80uVJB3MiIEeEVOBrwPnAScDH4mIwTcnPw9YXHusAL7Z5DolSSNo5Ax9CdCbmc9k5h7gduCiQX0uAm7JykPAnIg4rsm1SpIOopFAnwc8P2B9S61ttH2IiBUR0RMRPX19faOtVZJ0EI0EetRpG3wT9Ub6kJmrMrM7M7u7uroaqU+S1KBGAn0LsGDA+nzghUPoI0kaQ40E+iPA4oh4S0RMA5YDawb1WQNcVpvtshTYnplbm1zriF7uhX++ER6/Ffa8Ot7vLkmtNeKFRZnZHxGfBu4EpgKrM3NTRHyitn0lsBY4H+gFdgFXjF3J9T3wJbj3i9XylA5Y9xm44n7oGjwfR5IKVcR3iv76GfjGKdD/2wGNAcefCR9/pClvIbWVF3pg7afhlz+F2cfDe/8HnOrtfIpwsO8ULeLS/6fvYujgUcLWDdD/GnR4GxNNItufg5v/LezZWa3/5llY8zHonAkn/VFLS9MYKyLQZxwDU6YObZ86vRp+0Rvt3QWb/hZeegrmvwMWf6D+f7/BXt0Gm75bPf+kP4K5vzf2tWr01q+CfXve2LZ3F9x7nYFeuiLi7qQLoeOI2hlJbQSpYyac+fHGgmoyeXUbrOqG3S/D3ldh2pFw7Nvhsp/A1GnDP+/Ze+HWD0Duh/39cM8X4L1/CUuvGqfC1bDtzw0NdICdvxz/WjS+irg5V8cR8Of/BAveCTEFOmZA93+E9/3PVlc28dx9bfUPe29tFtCenbD1p/D4bcM/JxN+eEn1nP7dsH9v9fMn/wV2vjg+datxiz8AnbPe2DalA078d62pR+OniDN0gN9dDH/+QHX2GFOHfmmPKr13VoE80N5X4akfwdsvr/+cV/4f7HppaPvUafDc/XDyxc2vU4fu5D+Bn66GLQ9Wf7CnHQlHzIH3/PdWV6axVkygH+CY+cEdNQ+2P/vGtimdcPSi4Z8z/XeqoZbBcj/MenMzq1MzTOmAS9ZVkwW2/B84+m3VH93OGa2uTGOtiCEXNe7sa6vZDgNNnQbdnxz+OdNnw7/+D9VQ1gFTOuCo+bDwD8emTh2emFINsSz7Apx2qWE+WRjok8zb3g8f+g787klVQM8/Cz56Dxz9loM/74KV8G8+BdOPqmYPnXQRfPReh7akiaSIC4skabIo/sKiZti/D57/J3jtFVh0dvVBkiS1EwOd6tYB314Gv/1NNYSwvx/+5PZqfrsktQvH0IG//ffV1Lw9r8BrO6qr6r63HH67vdWVSVLjJn2g734Ztj02dFrelKnw9I9bU5MkHYpJH+jDzVuPqK5AlaR2MekDffpR8Nb3Db2PSUytpvhJUruY9IEO8Md/AyecXc2v7pgBR78VLv9Hb7srqb04ywWYcTRc+mN4ta/6QPR3FnrBjKT2Y6APMKur1RVI0qFzyEWSCmGgS1IhDHRJKoSBLkmFMNAlqRAtu31uRPQBv2jJm79uLvCrFtcwFtyv9uJ+tZdW79cJmVl3Tl7LAn0iiIie4e4r3M7cr/bifrWXibxfDrlIUiEMdEkqxGQP9FWtLmCMuF/txf1qLxN2vyb1GLoklWSyn6FLUjEMdEkqxKQK9Ig4JiLuioj/W/t59DD9no2IxyNiY0T0jHedjYqIcyPiyYjojYir62yPiLihtv2xiDijFXWOVgP7tSwitteOz8aIuLYVdY5GRKyOiG0R8bNhtrflsYKG9q0dj9eCiLg7IjZHxKaI+GydPhPvmGXmpHkAXwauri1fDVw/TL9ngbmtrneEfZkKPA28FZgGPAqcPKjP+cA6IIClwMOtrrtJ+7UM+FGrax3lfr0bOAP42TDb2+5YjWLf2vF4HQecUVueDTzVDv++JtUZOnARcHNt+Wbgg60r5bAtAXoz85nM3APcTrV/A10E3JKVh4A5EXHceBc6So3sV9vJzPuAlw/SpR2PFdDQvrWdzNyamRtqy68Am4F5g7pNuGM22QL9zZm5FaoDBrxpmH4J/Dgi1kfEinGrbnTmAc8PWN/C0F+4RvpMNI3WfFZEPBoR6yLilPEpbUy147EajbY9XhGxCDgdeHjQpgl3zIr7xqKI+Afg2DqbPjeKl3lXZr4QEW8C7oqIn9fOQiaSel+SN3gOaiN9JppGat5AdT+LnRFxPnAHsHisCxtj7XisGtW2xysijgS+D1yVmTsGb67zlJYes+LO0DPznMw8tc7j74AXD/wvUe3ntmFe44Xaz23AD6mGASaaLcCCAevzgRcOoc9EM2LNmbkjM3fWltcCnRExd/xKHBPteKwa0q7HKyI6qcL8O5n5gzpdJtwxKy7QR7AGuLy2fDnwd4M7RMSsiJh9YBl4P1D30/sWewRYHBFviYhpwHKq/RtoDXBZ7dP4pcD2A0NOE9iI+xURx0ZUX+MdEUuofo9fGvdKm6sdj1VD2vF41eq9CdicmV8ZptuEO2bFDbmM4EvAdyPiY8BzwIcBIuJ44FuZeT7wZuCHtd+/DuDWzPz7FtU7rMzsj4hPA3dSzQxZnZmbIuITte0rgbVUn8T3AruAK1pVb6Ma3K+LgSsjoh/YDSzP2rSDiSoibqOa7TE3IrYAnwc6oX2P1QEN7FvbHS/gXcClwOMRsbHWdg2wECbuMfPSf0kqxGQbcpGkYhnoklQIA12SCmGgS1IhDHRJKoSBLkmFMNAlqRD/HyuExRPQW6j7AAAAAElFTkSuQmCC) 

```python
from sklearn.linear_model import LogisticRegression as LogiR
clf = LogiR().fit(X,y)
prob = clf.predict_proba(X)
prob

import pandas as pd
prob = pd.DataFrame(prob)
prob.columns=["0","1"]
prob

#手动调节阈值
for i in range(prob.shape[0]):
    if prob.loc[i,"1"] >0.5:
        prob.loc[i,"pred"] =1
    else:
        prob.loc[i,"pred"] =0
        
prob["y_true"] =y
prob = prob.sort_values(by="1",ascending=False)
#根据1，排列

from sklearn.metrics import confusion_matrix as CM ,precision_score as P ,recall_score as R

#混淆矩阵
#参数(真实值，预测值，[少数类,多数类])
CM(prob.loc[:,"y_true"],prob.loc[:,"pred"],labels=[1,0])


#手动计算Precision 是(1,1)/第一列的总和
2/2
#手动计算Recall是（1，1）/第一行的总和
2/4

#使用类计算
P(prob.loc[:,"y_true"],prob.loc[:,"pred"],labels=[1,0])

# 使用类计算
R(prob.loc[:,"y_true"],prob.loc[:,"pred"],labels=[1,0])

```

#### 画ROC曲线

```python
cm = CM(prob.loc[:,"y_true"],prob.loc[:,"pred"],labels=[1,0])
cm

#FPR:假正率（预测错误的0/所有真实的0）= 1-特异度  特异度：预测的0/真实的0
cm[1,0]/cm[1,:].sum()

#召回率
cm[0,0]/cm[0,:].sum()

import numpy as np
probrange = np.linspace(clf_proba(X)[:,1].min(),clf_proba(X)[:,1].max())

recall = []
FPR = []

for i in probrange:
    y_predict = []
    for j in range(X.shape[0]):
        if clf.predict_proba(X)[j,1] > i:
            y_predict.append(1)
        else:
            y_predict.append(0)
    cm = CM(y,y_predict,labels=[1,0])
    recall.append(cm[0,0]/cm[0,:].sum())
    FPR.append(cm[1,0]/cm[1,:].sum())
    
recall.sort()
FPR.sort()

plt.plot(FPR ,recall,c='red')
plt.plot(probrange+0.05,probrange+0.05,c='black',linkstyle='---')
plt.show()

```





#### 实战代码

```python
#predict为1 的点 / 全部为1的点的比例
(y[y == clf.predict(x)]==1).sum()/(y==1).sum()

from sklearn.metrics import brier_score_loss as BS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X, y = digits.data, digits.target
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y
                        ,test_size=0.3,random_state=420)

#建模
gnb = GaussianNB().fit(Xtrain,Ytrain)
#查看分数
acc_score = gnb.score(Xtest,Ytest)
#查看预测结果
Y_pred = gnb.predict(Xtest)
#查看预测的概率结果
prob = gnb.predict_proba(Xtest)

prob.shape

#混淆矩阵和精确性可以帮助我们了解贝叶斯的分类结果
from sklearn.metrics import confusion_matrix as CM
CM(Ytest,Y_pred)

#使用布里尔系数进行评估
from sklearn.metrics import brier_score_loss
brier_score_loss(Ytest, prob[:,0], pos_label=0)
#我们的pos_label与prob中的索引一致，就可以查看这个类别
```



##  支持向量机

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C21.png)

**SVM无论在线性还是非线性分类中都是非常好的**

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C22.png)

月亮型、环形、对半型 ，SVM都表现的非常好

* 超平面：
  在几何中，超平面是一个空间的子空间，它是维度比所在空间小一维的空间。 如果数据空间本身是三维的，
  则其超平面是二维平面，而如果数据空间本身是二维的，则其超平面是一维的直线。在二分类问题中，如果一个超平面能够将数据划分为两个集合，其中每个集合中包含单独的一个类别，我们就
  说这个超平面是数据的“决策边界“

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C23.png)

注意：除了特别表明是线性的两个类LinearSVC和LinearSVR之外，其他的所有类都是同时支持线性和非线性的。NuSVC和NuSVC可以手动调节支持向量的数目，其他参数都与最常用的SVC和SVR一致。注意OneClassSVM是无监
督的类

* Libsvm是台湾大学林智仁(Lin Chih-Jen)教授等人开发设计的一个简单、易于使用和快速有效的英文的SVM库，它提供了大量SVM的底层计算和参数选择

### svm.SVC

svc:支持向量机的分类

> 参数列表：
> class sklearn.svm.SVC (C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True,
> probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
> decision_function_shape=’ovr’, random_state=None)

### SVC的参数列表：

svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, 
shrinking=True, probability=False, tol=0.001, cache_size=200,
 class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

* C：惩罚系数
* Kernal：核函数
* gamma ：：核函数系数，默认"auto"
* degree ：针对ploy核函数的参数，默认为3
* coef0: 核函数的常数项，只对'poly','sigmoid'有用
* cache_size :
* class_weight:"banlanced"  样本间分配权重
* probability:默认为False,决定最后是否按概率输出每种可能的概率，但需注意最后的预测函数应改为clf.predict_proba。
* .decision_function_shape:决定了分类时，是一对多的方式来构建超平面，还是一对一。'ovo'（一对一）还是'ovr'（一对多）。


### 常用函数：

* predict：返回一个数组表示个测试样本的类别。
* predict_probe：返回一个数组表示测试样本属于每种类型的概率。
* decision_function：返回一个数组表示测试样本到对应类型的超平面距离。
* get_params：获取当前svm函数的各项参数值。
* score：获取预测结果准确率。
* set_params：设置SVC函数的参数
* clf.n_support_:各类的支持向量的个数
* clf.support_：各类的支持向量在训练样本中的索引
* clf.support_vectors_：全部支持向量

------------------------------------------------

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C32.png)

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C31.png)



### 调参：

- 采用网格搜索，寻找最佳的coef和gamma

  ![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C33.png)

 StratifiedSShuffSplit(n_split,test_size,random_state) 把数据集分成几份，每一份里面再分测试集和验证集 

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C34.png)

调C：

```
#调C： rbf核函数进一步细化C

score = []
C_range = np.linspace(3,4,50)
for i in C_range:
    clf = SVC(kernel="rbf",C=i,cache_size=5000).fit(xtrain,ytrain)
    score.append(clf.score(xtest,ytest))
    
print(max(score),C_range[score.index(max(score))])
plt.plot(C_range,score)
plt.show()
```



### 硬间隔和软间隔

> 硬间隔：数据集线性可分，能找到超平面，把数据完全的分开

---

> 软间隔：数据不是完全线性可分，肯定有那么几个分错的，所以我们允许一定的容许出错，即加入一个松弛系数，**找出最大边界和出错数量之间的平衡**
> 这个系数的意义就是，让决策边界向上向下多平移一点


* 更新损失函数:

  ![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C35.png)

* 拉格朗日函数的变化：加入了C表示对出错的点的惩罚力度

  ![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C36.png)

* 参数C:

  ![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C39.png)

**参数C用于权衡将训练样本的正确分类与决策函数的边际化的效果，默认为1**



* matplotlib.axes.Axes.contour([X,Y],Z,[level])

Contour是我们专门拿来绘制等高线的函数。x,y表示坐标轴，Z表示高度

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C25.png)



首先，我们需要获取样本构成的平面，作为一个对象，也就是先要有一个散点图

有了这个平面，我们需要在平面上制作一个足够细的网格，来代表我们平面上的所有点



### 制作网格

meshgrid

```python
#首先要有散点图
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
ax = plt.gca() #获取当前的子图，如果不存在，则创建新的子图

#获取坐标轴的最大、最小值 xy轴互相匹配构成网格的坐标点
xlim = ax.get_xlim()
ylim = ax.get_ylim() 
#默认创建(0.0, 1.0)范围内的横纵坐标

#在最大值和最小值之间形成30个规律的数据，给x，y轴
axisx = np.linspace(xlim[0],xlim[1],30)
axisy = np.linspace(ylim[0],ylim[1],30)

#要画决策边界，必须要有网格
#将30个样本点进行复制，形成30行30列的矩阵
axisy,axisx = np.meshgrid(axisy,axisx) 

#ravel()把矩阵进行拉平，30x30的二维矩阵变成900的一位数组
xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
#获取y.shape * x.shape这么多个坐标点
#其中ravel()是降维函数，
#vstack能够将多个结构一致的一维数组按行堆叠起来
#xy就是已经形成的网络，它是遍布在整个画布上的密集的点

plt.scatter(xy[:,0],xy[:,1],s=1,cmap="rainow")
#画出900个点
```

 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXwAAAD4CAYAAADvsV2wAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAA/60lEQVR4nO2dd3hUVfrHPye9ETooEKQIIk2aKBYERBBwZVGavYJYdtVVd9XV1Z8NXV17AQuKayUqihVBFxERBQWkQ4JAkJ4AIXVSzu+Pk5BM5s4kk0y5M/N+nuc+Kffke9+bm7z33nO+5z1Ka40gCIIQ/kQFOwBBEAQhMEjCFwRBiBAk4QuCIEQIkvAFQRAiBEn4giAIEUJMsAPwRIsWLXSHDh2CHYYgCELI8MsvvxzQWre02mfrhN+hQwdWrFgR7DAEQRBCBqXUdnf7pEtHEAQhQpCELwiCECFIwhcEQYgQJOELgiBECJLwBUEQIgRJ+IIgCBGCJHxBEIQIISwTfk6+g5nfZZKT7/BJO9EUTdEUTTtqektYJvz0FVlM/3Ij6SuyfNJONEVTNEXTjppeo7W27da/f39dH7LzivWMRRk6O6/YJ+1EUzRFUzTtqGkFsEK7yalK23jFqwEDBmgprSAIglB3lFK/aK0HWO0Lyy4dQRAEwRWfJHyl1Cyl1D6l1Fo3+5VS6lmlVIZS6jelVD9fHFcQBEGoO756wn8DONfD/lFAl4ptKvCSj44rCAFn90r4YDI81xXeGAIb5oKNe0YF4Sg+Sfha68VAjocmY4E3K8YUlgFNlFLH+uLYVoSKpSocNV/6NpP9h+wfZ812O5fBWyPh0SbwZFv437+g+Ihr27Xvw6wzYH065GyB7d/B3Mvg3akOZoTINRLN8NH0lkD14bcFqnuMdlZ8zwWl1FSl1Aql1Ir9+/fX62ChYqkKJ82sH+GOK7N47OuN3DAqi9nDYO9v9ovTqt3mz2D22ZD5NRQfhiO7YOnj8Nqp8N7SqrYlBTDvGigtAF1epVeSD59szOJRm18j0Qw/TW8J1AIoyuJ7li/BWuuXgZfBuHTqc7AJA9KcPja0nWh6bpu1FP57DrTSaQzoBcevSWNbIbx2Olz7I7TqaY84rdqVl1Ul8eqUFsHBbTBwbRp3jTJtt3wBys0jUqdf02jd277XSDTDU9NbfGbLVEp1AD7TWrv8eyulZgKLtNbvVny9CRiitd7tSVNsmaHBywNg9y8WOxR0GQ0XfxbwkOrMrhUwexg4jljvb90bpq02n//6Gnx1s3mit6LLeXDxp/6JUxDqih1smfOAyyvcOqcCh2tL9kJo4MiHvavd7NSmm8TOlBaBsnr/rLa/krRBzl051YlNgs4jfBubIPgan3TpKKXeBYYALZRSO4H7gFgArfUM4AtgNJABFABX+eK4QvDxlCxNg4CEUW+O7QflZdb7ouPghPOrvm7ZHdqfATu+d74RoCAmCU663K+hCkKD8UnC11pfVMt+Ddzoi2MJ9iI2CY7pC7uWW+xUcPzIgIfkFbFJcObd8P3DUFK9H1+Zfafe6tx+0kfw8VWw+VOISYAyBzQ7HiZ+AAmNAxq6IHhNWM60DRVLVbhojnrOJMeiRAdrBmZSlOgABXEpMPxR+8Tprt0Zd8GwhyChqTmP6HhoNwiu+RFKGju3jUuBielwyzaY/Alc9ytc/xuotva+RqIZnpreEpYJP1QsVeGi2e4UuGIRHBibxYqhG8k4KYvOI03CbNndPnG6a6eUeZK/Yx9cvxZuzYJrfoAW3dxrphwDHc4ybQIVZyhqag371xvbbvER+8YZqppe466qmh02qZYZepov/S9DHzhi/zhF0/+aO3/S+tnjtX44SevpqVo/lKj1B38zfyN2ijOUNa1AqmUKwabgAGz8GBx50P5MaNM/2BHZA62hOBdiE80gcbhw8HeY0dtc7+rEJkG/KXDu00EJKyLwZMsM1MQrIYL5+QVYcDuoaCgvhahoaDMALv7c9InXZOs3sGQ67N8AqW1h0G3QY2IdHEEhxq+vwaJ/Qf5+c25d/2TGQxr5qejIH8th0zxAm2O1Hei/3+mP/4HSYtfvlxTALzNhyP2Q0MQ/xxbcI0/4IURZCWR8BYd3QIsToOMw9zM/7cK27+Cd0TUcMJiB0W5jYfz7zt9f9jR8+0/n9rHJ0Oti+NPLfg83YPzwGHz3gPN5qhhIaQU3rPNtMiwvhTnjYesCKCkCtHnS7nAWTPoYomN9d6xKnj8Rsjda74tvbAa8O5xlvV+XQ+FB8zAQE+/72MIdO0y8EhrIH8vhyTbw0SWw4A54fxw80wmytwQ7Ms8secQ12QOUFZunzYIDVd/L3w/f3OXaviQf1rxtZsWGA45812QPoEtNovvlFd8eb8mjZgJcSQFQDmjzO/39fyYOfxCf6n6fLoP4Rhbf17D0CXi8lSlg92hj+PAS578RoWGEZcIPFUtVXdvu3u3gtttNO8cRKC00faOHd8Dsoc4Th+x27vvXW7ettHD+vqGq7caPTbePVbs85WD1m/6LM5Ca675z8NuACvtqDUoLYd17vo1z2TNGF5yts6WF8PPz7ks7N+TcB1xn3sxqUpToYO2gTOK6ump+fRssug8KsyE/ysGqkzL59VMHr54KJYX1O/f6tAslTW8Jy4QfKpaqurZ94Y0slg3ayJZeNdpVDPhlfGmPOK3apbazbrullzmnhQer2pbkm+6Hmu1WDN3I5h5ZFB/2X5yB1Pxqj5vrWUHlTc8XcZaXQWG1J+TK32flsR1Haswarsc5WbXrfZmZxRyb5HxeW/tl8UO/jXzwi7Nm/j5Y/lLVW09lnJtOzCJvD6x9z/tzr2+7UNL0lrActA2VSnd1bdt9axoDfoMua1zblRaap+iu5wU/Tqt2g26Dj690LTjWdX0aTTvBZfdWtT1usBnQrV7poPKcu29No/Mt/oszkJrXjE1j1TjoZHE9q5do8EWcUdGQ1KKqW6Ty91n5MTrOjKd4o1mXdtGxcPlCWPk6/DIDig7DcWfBxTen8c1hV83fvzU/U1bkGmdJIaybA33dFGQJleseVtUy/YEM2hq+e9BM/S+zcD3EpcCYl6D3pfXXL3OYJ+vqT2O+Qmv4/Ab47c2KAcNyE3NiMzMxq1Eb5/ZvDIU/ljk/daoY49a5aVP4DOL9+gp8dYtzP350HDQ+Dq5bCXEW3SH15ftHLEpHVBAVC/2uhTEv+u549WH9B/DxFdYxgnEUXftTYGMKVWTQNsTpe5VnN063cfXTzd4Cb4+GR5Jheio8380sBuJLlILzXoIrF8PAG83T65gZcNNm12QPprxwt3GmTk18qnn6PG6wuTmES7IH40W/8D1o1dt0dcQ1gn5TYcrPvk32AKf/HdqdZr2vvARWvR78wf9Owz330x+R2ro+ISy7dMKN1HZw3kz47DrjcChzmISoomDiR/VLEIez4NWB5lW7cima7E3wwSQY+7rxvfuSNv3rNtkqLgUufAcKc+DQNkg51n++9GBzwp/M5m+iYqDtyabbBIvyzuVlsHGuuTEEi/jGuFkSyZD7h7khxCYGLKSwRBJ+iHDSZaY07y8vQ85maN0H+k8xNV3qw/ePVMyCrPFPVlJguhq6jw+uxz+xmdkE31BWgmWyh6qHiGATHec+DhVlxiOEhhGWXTqhYqnyVlO3cjB8Okz8EM661zrZ11Vz5ZcOVveztgYW5zq/4tvh3EWzYZrHn2venpwqmlYQkwidLcpYBzJOpeCEsRzNSDUrr3Y6233piXC5RvXR9JawTPihYqkKpubGLs72PCe089N9uJ17JGp2HGbWFs7s63zdYxLNSl5tTw5+nOf8GxKbmEH6SltmxklZxDfyXHsnXK5RfTS9JSy7dELFUhVMzXE90nAsNguO1ySxuVnUww5xiqZvNJWCyxZC0q1psARO2JJGbBL0uRpGPG6POJt0gOtWweKHQH+WRuxPMKZbGmPehqadGn58u1+j+mh6i9gyI4ycjKoSBV/cBEWHTB9uJTGJMCEduo4JSnhCACgtgoJs488PJ+eTYJBqmQKOPEifCNv+Z7zXYOyAx50FO5eawbI2A2D4Y9BhSFBDFfxMTIKZ1yBEHpLwI4SPLjHJvrQIqDap6Y9lcP0aaNo5/MoPC4LgTFgO2obKCHvNtmUOUzPko0uN537HEjNTtaFxHs4y1RKrz16tdEHkRzv48UnPyT5Uf5+iKZrhruktYZnwQ2WEvXrb/P3wYg/4dIopBfzLK/DWufDBZJizvGFx7v3NtV5K9eJUO5d5r9mQdqIpmqLpG01vCcsunVAZYa/edt7FcGi7meoOHK1ZvuUzGDQ0jbtG1T/O5JbOA7PgXJwqZYj3mg1pJ5qiWV/NA5vgf/fAli/NW2mX82D4PWlQh/+PUD93XyAuHRtQeBD+c6x1cTSAFt3gxg3119canulg6ufXJDbZrDolrhzB7uxbB68NMg9CumLWsIoydYim/AzNuwY3PrsgxdNsTv4+z8vM5e1tmL5SFTV3GhmHRiWxydBjEnQZ3TB9QQgE828xbjNdrUSELjczwxfcEbSwQoqw7NIJNVLbOa9aVRNfPLm06Q9/2WwWmdi+CJJbQ//rKtbFFXeOYHPKSsySjJYF1jRs+cK8ycrfsmck4duAuGTocyWseqNqKbpKYpNg8D2+OU7KMTD0/3yjJQiBRFesxVvrfkn4HgnLLp1QsVRVbzvyKVMTPCbRdLvEJpuPg++FFkPtE6doimYwNGPiTYVYK4oSHeyYkMnBwuDHGWhNbwnLhB8qlqrqbWPi4aJ5MGU5nPO4KRZ1y3Y44057xSmaohkszZFPmiUga7K1XxYLO9gnzkBqeo3W2rZb//79dX3IzivWMxZl6Oy8Yp+0E03RFE17aGYu1Pr57lo/GGe2F3trvWqB/eIMlKYVwArtJqeKLVMQhJCj4ACgIKl5sCOxH1I8TRCEsCKpRbAjCE3Csg9fEARBcEUSviAIQoTgk4SvlDpXKbVJKZWhlLrTYv8QpdRhpdSqiu1fvjiuO0LFUiWaoimaotkQTW9pcMJXSkUDLwCjgO7ARUqp7hZNv9da96nYHmjocT0RKpYq0RRN0RTNhmh6jTv7Tl03YBAwv9rXdwF31WgzBPjMW22xZYqmaIqmaHoH/rRlKqXGA+dqra+t+Poy4BSt9U3V2gwBPgR2AruA27XW69zoTQWmArRv377/9u3bGxSfIAhCJOHvaplW1Stq3kV+BY7TWp8EPAd87E5Ma/2y1nqA1npAy5YtfRCeIAiCAL5J+DuB6pX622Ge4o+itc7VWudVfP4FEKuUEietIAhCAPFFwl8OdFFKdVRKxQGTgXnVGyiljlHKFC5VSg2sOG62D44tCIIg1JEGJ3ytdSlwEzAf2ADM0VqvU0pNU0pNq2g2HlirlFoNPAtM1g0dPPBAqFiqRFM0RTNwmtl5Dh5/O5PvX3ewbZHzQip2itPWtkww3TRa665a685a64crvjdDaz2j4vPntdY9tNYnaa1P1Vov9cVx3REqlirRFE3RDIxm7h9w2yVZvLBmI8+/nsW758NT7WHvb/aK01tNbwnLWjqhsgCxaIqmaPpfU5fDm8Ogza40BuyBTmvScBSC4wi8McSUIY9vFPw466PpLVItUxCEsGbrN/D+n816uDWJTTbrT5x8fcDD8huyiLkgCBHLnlVQ5qY7vCQfdi4LaDhBRRK+IAhhTXIriI6z3hcVC6ltAxtPMAnLPny7oDX8/i2smAF5e6DDWXDyDdCoTbAjE4TI4cRx8LmbLpuoGOh7Te0aeXvht//Cwa3Qqhf0vgTiU30bZyAIyyd8O1iqDuQ6mHs5vDcWNnwAWUtg6RPwfDf442f7xCmaohnumnEpMP49KGvmYN2gTIoSHahosz7u8OnQrLNnzQ0fwTMd4X/3woqXYMEd8FQa7Pwp+OfuLWGZ8INpqZpT0e6mP2Wx5i3TR1hJWbFxBrx/gXEOhIr1SzRFM9Q1u54HzV7N4ufBG8kel0Xfa+DaH+GUmz1rvvVtFh9dCqWFUFpk9pXkQ3EuvD0K3l8mtsygE0xLVctP0zhlGXT81X274lwzUBQq1i/RFM1w0Lx8RBqJzUy7Zsl102y3OI1VboyM5aXQZ18ad40SW6ZPCDVbZv5+eLIdlNfyJhafCuP+CyecH5i4BEGoHx9fAavftN4XFQtnPwKn3R7YmGpDbJkBYutCc9evjTIHtO7t/3gEQWgYLXtCTKL1vpgEaN41sPE0FEn4PmTXcsBDfQ6A6HjoNByadAhERIIgNIS+V4FykyVjk6DL6MDG01DCMuEHy11wYBMUJTpYM9A4AWqioqHjMLjw3eDGKZpB0vzqKxg4kJwmLZk57HJy7rwX8iymfwY7TtE82rYo0cFF84zTJy4FiIK4RpDUCi7/Bg4Vi0sn6ATLCRAdC1t6ZbFi6Ea29KrRTsGwR+CSLyr+cIIYp2gGQXPWLLjwQli+nPQTBjN94CTSv98Mp58ORUX2iTNMNbWGme+Zds+9mkVJQd01Ow6D2/bA6Bdg6AMw9nX4205o1SP45+417tY+tMMWamvarnlP63tbFOsLB2boOxOL9f3oo9uD8Vrn7bNHnKIZYM2CAq1TUrQ2eUdnJ6bqGQMv0NmJqVonJ2v92mv2iDNMNQ9naf1sF/O/OWFQhr63ZbF+JEXrLV/ZK876aFqBP9e09Seh5tIpK4FXToYDG43nvpLYZBj4FzPJQ4hAFiyA8eMhN9d6/xlnwPffBzamCEFreKmX+Z/UZc77YpPgxo3Q2Pfux6AiLp0AER0LVy+B/teZJI+C1DQY+aSxbwkRSkmJ5/0O3/fVCoY/foJD21yTPRhH3fIXAx5SUAnLiVfBJC4FRj0D5z5t/qCiY4MdkdBgcnPhiy8gP988jZ9wgnc/f9pp7pN6YiJMmNDwGAVL9m9wv6/MAXtWBi4WOyBP+H5CKUn2YcGsWXDMMTB1Kvz1r9CnD5x/PhQW1l2jSRO47TZIrjG9MzoaGjeGa6/1ZcRCNVLbubdVqmho2imw8QSbsEz4gbR+lRbBujnw41OQMR8O5NrDThbRmnnFsGsX7N/fMM0lS8i5425m9hpFTqmCggLjqFmwAG64wTvNBx+Ehx6CFi2MLXPQBHLOGwfLl5sbglWcR4qY+f4P5Py8Eko9z+gLuWsUIM2OwyCu4j5b0zIdHWeq19ohzvpqektYJvxAWb+2LYInWsO8a+GbOyF9PNwy0f4WtbDXvPAG6NwZ2rUzT+Q//QTr1sGYMRAbS/rwS0y7z5Z7Fnz4YdKPP53pQ68mvdfwqu8XFcF778HBg3WPUym45RbYs4f0/37N9MFXkH7royZGK+bOJX3cNKavPET67f82bxlvupnjX5fje9kuXDSjouHizyG+MWztZyzTGX2yiEmEcx6DVj3tEWd9Nb3GnX3HDpudbZl5e7V+OFk7WS/vR+u7kov1JSMz9IFce1rUwlrzvQ/1jNMnGbtjhQVSg9YJCVonJWmtlLMtssWxWq9a5V6wTRtnC2V1zdRUrVesMO0+/FBnDxhkjt35BK0fe0xrh6P+5/TNN1onJroeOylJ6/T0Bv2egn6NgqRZeEjrr58q1jdel6E/+kexzt5izzi91bQCsWX6nu+nw+IHqkqmVicuBSbNNSUUhAChNXTsCNu3e/dzw4bBN99Y7+vfH3791XpfQgJs2QL//a/pqimoMZMnJQWeeQauvBKivHyRHjjQdPVY0bmzOa5S3mkKEYPYMv3AnlXWyR6MO+fAxoCGI+zdazZv+e479/3jU6eaxF6T6GhzM0hMhAcecE32YEomTJ0KTZtCWhpcconpVtK6dpumu5sMmBuah3IMguAJSfj1pGknUx7ViqhYaBRB62TagthYKK+lcp0VlZ001SkvhzvugJtvdr0ZJCdDy5bw9tvw+ecQ48HZXFZmLJ07d5o+/z59zJN/fDwceyw8/bQ5lsMB778PF10EV1xhbijuUAri3CzQKgi1ID78etJ/Cvz0DJRbPKxFxUDXMYGPKaJp3hx69vT8dGxF797mZlGdhx+GF1+E4mLn70dFwfXXw333mcRdUuJ6s3BHebnZKm8ge/bAP/8Jv/0GP/5obgqVT+7ubiJRUXDOOeaGIQj1ICyf8ANhqWrayRRTikkw9i4wU7XLWjrg6UwOl9jPohb2ms+8yMwzJpOT1LjqmzExxvZYrWsmJzGVmQMvICcxFTZtMi6eShwOePzxo900Tm3Ly2H1apPsAc4+2zzF12xXW5yVbXUMzJ4NmZnO3TQVN4WclCZVmnFx5jyee65Bv6egXyPRFFumrwmUparvVXDjBjjt79DrUhj6ILSancVL6+1pUQt7zbIWTD/9UtIn3wyNGpn+82uuMX3nF1xwdKAzvdfwKqtlYSEMHw433mjabdvm9NTu1BaguomgQweYOBESElzbeYqzetvycrd9+ul9zzXtzhxvfP9r1kAn65lCIXONRFNsme42O9syRTPENI85xrpaZWUvflSU1omJxlIZH+++bYcOzrolJVrffbfOTmlibd+02NxaPWu2a9zCvr9P0bSFphWILVOIeJo1c5oo5ZbEROjRw4wF1BwETkw0/ff/+Ifrzx05AtOmwYcfmjEBXzhpevSAtWsbriNEFGLLFISzz66bH76szMxqremUSUw0A8N//GEcOtUXLdEa9u2DO+80g69Llxr/fW0kJUHfvuaj1b57761dQxC8QBK+EN6UlMCcOXDoUN0mKzkcZinCmv3qxcUmqT/3HFx3HRx3nBnwXbrUVM/s3dusXnX88TB3rtk8kZRkJmwtW2bGFxISzPeSk40L5447YNIk15/75BM46SQziNu8Ofz97+LLF+qMdOkI4cuRI3Dmmc4uGKXqbqWsjbZtTTdRzYlXMTFVNkwrEhKMM6h376rvbd0KCxea7qAxY6BVK9efe+EFk+CrHy8hAbp2hZ9/FrumAERgl06oWKpE08+ad94JGzc6PwFrXT8LpVXbPXuOlkl2alda6nESWM5VU5j59XpyHnvSlHXQ2rhvpk6Fq646muydzic/3zXZg+layswk5633qyqFLl0K06fDs8+aLihPvyNP5x6q1z2CNL3FJwlfKXWuUmqTUipDKXWnxX6llHq2Yv9vSql+vjiuO0LFUiWaXmr+/rvpRimzWL6oZtvycnjjDdfJU1hYLT0d31PbsrKjbwt11oyJIX31XqYfaET6xz/Cn/8M3bvD7t2ez2fRIvcTsvLzSf9mrWl79V0wYoTp///HP0wX00MPWWvWdu52ue6i6Tvc2XfqugHRQCbQCYgDVgPda7QZDXwJKOBU4Ke6aIstUzSz84r1jFnzdfaJvY1tMjlZ6xYttJ41y7NmYaGxWnpji4yONseoj4Wyju0s28bEaD1ggOfz+eQTU6HTneaQc/SMKf+nsxu3cN2flKT155/79xqFuGZ5mdZbv9F62bNar/9Q673Zzm0dBVqvfEPreVO1/vZfWudkBifOuoA/bZlKqUHA/VrrkRVf31VxI5lerc1MYJHW+t2KrzcBQ7TWro811ZA+fIFffoHBg127MpKSYMYMuOwy65/TGtq0Md0udaV5c9NFkp9ft/aVdW0s3iIsiY42cVl19yQlmX79nm4KtB88aM6nyKJiX1IS/OtfcP/91vtBFkr3wKHt8ObZkL/XFD6MijWrYV00D447E7I3w6wzoLQQHHlmf1Q0DHsEBt0a7Ohd8Xcfflug+rvHzorvedtGEFy55x7r5QQLCkwfvbu+cqXg7rutLY9KudbPAcjOrnvyBhg/3rh1rCpqWtGhg/t4o6NNd5U7mjY1yyTWPJ+YGHOjOuMMz0XVtmypW4wRhi43yf7Q7yaZlxaB4wgUH4J3RkPeXnjnPCg4YPaDqZ9VWgTf3gO7fglq+F7ji4Rv5XWr+dpQlzamoVJTlVIrlFIr9rtZok6IIJYsce+qycnx/AR/001mMlR8vKl/k5JikvOttxr3jpVNs5alBI8SEwPPP29KLdxzj9uSB0dJSIBTTnFvDT1ypPayyQ8+aAZjW7Y05xQXZ8YAli83Th1PN6u0NM/aEcq278yTvba4D5eXwXf/B0d2YZmtyopMAcVQwhcJfydQ/a+pHbCrHm0A0Fq/rLUeoLUe0LJlSx+EJ4Q0np5ay8s9P10rBf/5j6mP89RTZpbsmjXme5s3N8yeeeaZxk3TqJGpepmZCWPHum+fkAAjR1q/WVRSbXDVEqXMQup79hj3zeHD5i3mb3+DoUMhNdW6tHJysvH1Cy7sX28SuxWlhbD7V9N9Y4Uuh5wM/8XmD3yR8JcDXZRSHZVSccBkYF6NNvOAyyvcOqcCh2vrv28IoWKpEs06aF58hVnw28oW2bevKZngSbOkBB57zNS2v/9+6NmTnAsmMrPrsIbZMg8fdm73w8/M3B/vXvPQIfPG0aqVe83ff4cNGzyfD5gZw82bQ3q6mez13nvkbM1iZsczyYlPqbqpREebGcJXXAETJnjWtDr3cPtbsmiX2taUM69JUaKDtYMySTjRQbmblz4VA8m9I8yWqbUuBW4C5gMbgDla63VKqWlKqWkVzb4AtgIZwCuAm7XifUOoWKpEsw6ao640C373GVn1zeho82T98su1a156Kcycafr88/OhuJj0PTD95Imk9z6n9jjd2S3XrYNdu8wN5aKLSL/9caafeblnW2Z+PuTmuteMjTWzeT2dTyWHDpkZv4WFUF5epdnzbPMmMGmSear/6SczYauyUmioXPcAaXYZbZ3wt/TKYvngjez/UxbNT7BuEx0Lu4ZZH9uRD2vehZ+eg+3fm5dJO9gyfbIAitb6C0xSr/69GdU+18CNvjhWXZgwIM3pY13a5e6E/Rug0bHWK9nXR9MfcUac5lndzEe6QlY7k2BHjTLdKMcf71kzIwPmzXNxrkxYNR/Ky5jw28La41yz0OnjURISzOzYf/8bPvmECZgVt1zaVae8HI4cYcK6b6w1i4vhxBPdn091Pv7YqTaQU5xlZWYW8PTpLj8WMtc9QJrRccaN8/Yo07VTWmhcOCduTuPEC+CKkWnEngpvDIG83VBSBDHxpjtn7GvQZkwaya2dNTd/Bh9MNvfYshJzY2jSAc6blwajfHvu3hLxpRWKc+Gjy2Dr1xAdb0bgGx8HEz+Elic6ty0tNn8Q8Y3dj71pDbuWw+/fGr0TL4Amx/n1FAR3vPIK3HKL9ZqzDSUhAVatgn79vNePjXUdoE1IMAOw775bN42nnzYTqxxuXvsvvdQssC7UiYIDsHKWcd006WBWtGtW7XlCl8PWhaZPP6kFdB8PCU1cdXIyYUZvKKnxJxEVA8f0gSlu1qb3JZ5smRG/xOG7Y2Hnj1BWXLUo+YGN8PqZ8NdMSGgMR3bDl3+BzZ+a/YnNYcj90G+Kc+IvKTAWrj9+NjeHqGj49m445WY4e3rdancJPiQ+vm4VMr0lJsY4brT2vP6sO9q1gwMHzOdaG2fQ8OHw2mt11xg40Nw4rBJ+SooZxBXqTFILOP3v7verKOg8wmye+Pk581Rfk/JSM0C89zdo3dt1f6AIy1o6dWXvGtj1s0n2TmjzJL96NhQdhldOho2fQJnDbHm7Yf6t8P0jzj/25V/NzaMkH3Rp1U3k5+dhfXrATkuoZMyYutssK0lKMuMDjRoZh1BysknqlYk9JQXatzeLkrds6f4J2xPbthkP/wsvwEsvGefQp59azxlwx6BB0K2bq4spKsroTJ7sfVxCg9n9q/U612Amc+3fYL0vUIRlwq/rKPeGpQ5W982kKNG1XUkBbF0Av74KhTlQGOtgzcCqtiUFsOSRqskYjnxY8w7kKed2YG4ANW8O3sQZyi6IoGo2b2686zUSqVuXTFycmdz0zjtw9dXw8MPkrF7PzPteIees4eYpfOZMM0HqmGOM/pAhoJR3BdkSGjFzYx459z8Eo0e7jEXU6dyVgq+/NrOQExLIad2WmadPIqfPyaZ4mpubh+2uUZhpNu1o3gasKIx38FlhiLt07EhdR7kXl2Tx0xkb2dLLop2CpFawbo552t/SK4sVQ53bRsXAjiXm87zdpgvHqh2YmXz1jTOUXRBB17z9dvM0fvLJxqfesWPVWrE1XTJRUbB/v7EwzpoF999P+sW3Mj2/Fel5yaZ2/bRpZjJYJRdfbI5d03mjlNtyxem9hhvn0TF93C5KXqdzb9YMFiyADRtIf/AVpp9xGelPvg2dO9dfsx5tRbOq3ck3QYybqSFb+2YxY2MYuHTsRl1Hua+bnMa6cdBpjWu72ETody18fbv5uktFmy412qqKN/3kVlBW6r5dI4tCEnZzLISt5p/+ZLbKdgsXwQMzmbClInHHxZkumwsvhDffNK6eCmePk/ulMK9Kb8cOU+7gscdAa1c3j9bQooVx3lT211cev3rb/SvNhLCGnHuHDky4uA2syArdaxQmmm1PhsH3wHcPmr58XQqxScYNdM9DaXxXKC4dtwTCpbP+A/j4CjPIqitm3MUmw0mXw5gXYflLsOB211F3gLgUuGN/1R39g8mwca7p569ObDKMeg76XuXXUxG84cAB8xS/cqV5Ir7mGjjttLoVW0tKMpbHadOMu8bd/5BSZq7AlCnutTp1MrN0hbBi/wZY+Roc+QPanWbySULjwBxbXDoe6D4emneFpU8YS1ZqW+Oq6TLa7O9zhamXceh350QemwQjnnB+fTtvBhzYAAe3mr59FWX2d58Afa4M6GkJtdGihVlQpJL8fJcncbcUFMD69catEx3tfmA4NtbMhvXE+efX7ZhCSNHyRJMf7EbEJ3wwNqlxb1rvi02Ca3+CRffBqtfN4GzLHjDsQTihxv9qQhOY+itkzoeMryAmEXpOhmP7+v0UhIaSkGASdF1cPYmJplhZVJSpnzN3rmsVzOhoGDfOrGjlCU/1dwTBx0R8l44gHOXaa81kpdqslklJsH27eUvYsQP694fc3Kqfi4uDJk1MLf+hQ82MXyvi481s3TZtfHoaQmQja9o2sJ1oRojmE0+Y/vzk5Kq2ianMPKXCbpmYaJJ9erpJ9mA8+WvXknPzbcwceQ05XXuY8str1pgJVtdfb36u+vErNU8+zWOyD/nfp2j6XdNbwjLh28mmJZohpNmkiRnEnTHD+OPPO4/0B2YyfcjVpE+7zwzU7thh9lWndWvSx1zD9D7jSH/5E3j00aMLkXPjjab8QrWbSHrfc43mLa61boJ27qIZkppe427tQztssqataIaFpsOh9VtvaX3WWVqffLLOvu8hPePz1faLUzRDTtMK/LmmrT+RPnwh6GgNr78OjzxS1W9/881mRq6nxUwEIUhEXB++IPiM224zq0xlZhoHz549plzD2LENWzFLEIKAJHxBcMeOHaa4WX6+8/cLCmDxYvjuu+DEJQj1RBK+ILjj00/d17QuKIA5cwIbjyA0kLBM+KFiqRJNm2uWlZETn2JdBVNrJ79+2J27aIaEpreEZcIPFUuVaNpUU2t49VV48knSu55pXVkzJcWsUBXMOEUz4jW9xp19xw6b2DJFMyiaU6ZonZysNejsxFQ9Y+AFOjsxVWtzK9A6Pl7rfv20Li0NbpyiGfGaViC2TEGoIxs2mFIJhYXW++Pj4fLL4cknzVO+INgMsWUKQl2ZO9d1gfFKoqPNgiovv+ya7HNz4YEHoGNHsxrWxRebippCxFB0CLK3WJdStwtSLVMQquNwQFmZ9b6yMutqmrm5ZkWtHTuOLpzC++/DvHlmGcLTTvNfvELQyd8Pn14LGfMhOhbKy6DPVTDySYixXvQsaMgTviBUZ8QI94uJJye71tEBeOYZ52QPplxyfr5ZG1cIW8ocMOs02PIllBWbdTBKC00p9fSJwY7OlbBM+KFiqRJNG2oOGmSe1hPMyjZHFydv0hL69IEzz3T9mddfJ0fFWds3d+wwJZB9HWc924qmbzWXzXGw5NhMCmKc25UWwtYFZuUrf8bpLWGZ8EPFUiWaNtRUCr74wjyZJyWR3m+UsWVedrvpnrGaiFVY6LqIeSUxMWaSlq/jrGdb0fSt5rtLsvjptI1s6WXRTsPv3/o3Tm8Jyz58Oy1qLJohqJmYCC+8AP/5DxOy9sCOYiaceg4kxVm3Hz6cCR9/ZjQrFzGvJCrKrI7ljzhFM+iaZ5DGnkXQZY1rOxXtvASqP+L0FrFlNhBHvlm4PPcPaNkduoyCqLC8jQpu2bgRBgxwrbmTlAQPPWQWRBHCku2L4e3RUJLvui86Hm7dAcmtAhuTLGLuJ37/Ft77M6ChtMisYRufClf8D5p3CXZ0QsDo1s2sXXvVVbBtm7FvxsTAffeZUspC2NL+TDj+XMj40tmOGZsMg+8JfLKvDXnCryd5e+HZzhZ3dgWp7eCWbaDCcoRE8Mi2babPvksXqZdvI0qLzZv4vrXQqC30nAyJTX2jXV4Gv74Cy56CvH3QvCsM/ieccL5v9L1FnvD9wMrXQFvZtbWZgLF1IXQeEeiohKDToUOwIxBqsG8dzB5q3sIdRyAmCb6+DS58B7r9uW4apUVweAckNoek5s77oqJhwDSz2Z2wfAYNhE1r72rzR1CTokQHK3tlsnW9ve1koimakaBZXgZvjYCCA5Bb6mDNwEzytIPSQvjwYsjd6VnzQK6DhXfCv1vAy/3hybbw5tlwOMse5+4tYZnwA2HTatYFoi1MG1t6ZfHzmRtZFmdvO5loimYkaG5dCMVHAG3+N1cMrbJQ6nL45WXPmg/+M4ufnzNdt448M7lq23fw6kAozg3+uXtLWHbpBMKm1W8K/Pika7sua9KITYJpdwbHTja+fxoF2XB2szS0dr9+R6jY3kRTNBuieTATyiuqYVRaJys/lhXD/nXuNYtzYd9laS61cXSZSfarZsOEq4N77t7SoEFbpVQz4H2gA7ANmKi1PmjRbhtwBCgDSt0NKNTEzoO2AGvehnlTzJNCWTHEJhlL5mULoO3AwMez+TP4/AYozDZfJzSFUc/BieMCH4sgeGLz57D4QTiwARJbwCl/hYE3+t7SvOVL+GCS6buvSXQcnHYHDHvI+mfXvg+fTrH+WYCOw+Dyb3wXq6/w56DtncA3WutHlVJ3Vnz9Dzdth2qtDzTweLai1yVw3BBY9QYc+h2O6Qu9L4WExoGP5fdvIX0SlFZ7GikpgI8ugUlz4fiRgY9JEKxY9hR8e0+VjbE4F769GzK+gks+9627rfMIiEu2TtoqGvpPdf+zMQmAmzdkMNbLUKOhv9qxwOyKz2cDf26gXsiR2tZYsM5/1TyhBCPZA3x9h3Oyr6S0EBbcEfh4BMGKwoPwzd2uJYRLCiBrCWQu8O3xoqLh0q+NuyauoqJ1TKLZxv0XGrd3/7OdhoO2KI4KJtn3ucq3sQaChj7ht9Za7wbQWu9WSrmbZqCBr5VSGpiptXYzVAJKqanAVID27T1cDeEo5WWwZ5X7/fvXGR+y3Uq1CpFHxpfuu20cebD6Td+/jbbuBbdmwfoPYO9v0DgNel0MSS08/1xcMpz7LHz11xqTqpKg7SnB89k3hFqf8JVSC5VSay22sV4c53StdT9gFHCjUmqwu4Za65e11gO01gNatmzpxSGqCAc7mTftVJR5kgFjC10zMJOiRIfz/hr/ZOFy7qIZepqr+jj/fVanciKjr+OMTYS0Cxz8fl4mXa5x1JrsKzU7THZw0adw3GAzJta0Ewx7GC79yvzPBfv36S21Jnyt9XCtdU+L7RNgr1LqWICKj/vcaOyq+LgPmAv4dUgzHOxk3rRTCrqNNYm9pvVMRUGX0VU3hGDGKZqiubJFFj+dbl1dMjalaiJUsOOs3rbjMLjyO/hHDvw1E069xSx0Yoc4vaWhXTrzgCuARys+flKzgVIqGYjSWh+p+HwE8EADj+uRcLCTedvunCfg9//BCRuqrGdRsabfcuRT9olTNCNb8/IRaWTMh6YZzm2jYk0XS49J9ogzVDS9paG2zObAHKA9sAOYoLXOUUq1AV7VWo9WSnXCPNWDucG8o7V+uC76drdl2o3cP2DJo7DhQ9Da2DHPvNvU9hEEu1BWYgZuV7xkvi4vhc7nwJ9ehZTWwY0tHPBky5TiaYIgBIXSIjiyCxKbQUKTYEcTPkjxNEEQbEdMghkEFQJHWNbSCSXHgmiKpmiKZn01vSUsE36oOBbqo6nLYeXr8FJveKI1zB7m/3UzRVM0RdOemt4Sll06oTLC7q3m+P5pfDAZtnxeNREkfx/88ROc8zicfIM94hRN0RTNwGh6iwzahhDbFsE751mvnxmTALftlsEvQYh0PA3ahmWXTriyarZrDZJKomJMBUJBEAR3SMIPIRy5mKpEFuhy9zcDQRAEkIQfUnQZ474kq9am3ocgCII7wjLhh4qlylvNNuc7SGxq6nhXJyYROp0DLU6wR5zB0tye4eCx2Zls22zvOEVTNH2l6S1hmfBDxVLlrebH67K4Zhl0OAui4yE+1QzW9r4UJrxvnzgDrVlwAP47Au64IouXNmzk71dl8ebZxsFkpzhF0/+ajjwozDFvvHaO01ea3iK2zDq0s5NmarJZVi1vr0loTY4zid9ucQZKU5fDG2dB9hboHJNm6rKsSWN7Kbw+GG5Y51opNFzOXTSr2ubtgdIH0njse0CZ+lEjnnBe3tMOcfpa01vElikEjPx9xklUXmK6oJp2bLhmxnxIH28Wz6hJXCO48B3oel7DjyPYl5xMmNnXdRnD2CQY+zr0mBicuIKF1NIR/EJpsXl9TmpuFoT2xKL/gx8erRh/0ObJvPsE8w9Z8wncG3YssU72YBLA9u8l4Yc7ix+wdqiVFMD8W6H7eN+ukxvKSMKPABx5cGg7JLeEZHeLUHpBSQHMv80sR4c2C7D0vRbOeaxi4ecarH0flj5uqiNWZ8OH0KQDDG3A6gjxqeZmU2YxvhUVG7w1hoXAsfkL0GXW+4oOwaFtUqStErnvhTFlDvjiJni8Fbw2CJ5qD28ON3Xz64vW8NZIWP2GWTS9tNDcAH59Bd79k/Vg2eIHrGcHlxTAsqdNffT60nOS+6e3qGjoObn+2kJo4OkNUWv3a+hGImGZ8EPFUuVvzY8ugZWzTFJ2HIGyYlOe4dVTYO+B+mluX2wWTK/5tF5aCFtWOnj8bVfNnExXzcq1dwtiHBRm1//cG7eHs/5l+murr+cbmwxn3GX9ZGenaySaDdc8bqKDtYOs18lNbQepafaIU2yZfiJULFX+1Jw9P4vNn5lEXB1dZl5zX3i9fnFu+cJ9n/n647N4ca2rptWC0ZVr7246MYt4D90udTn3M+6CyfMg50KjmXNBFpPmmhtBfTW9bSuawdPcPSKL5YM3knGSc9vYJDhvhulytEOcYsv0E6FiqfKnZrfNafzg5lW3JB+OW5bGXX/1Ps5VC00Xii53bXfC+jSOP9dVc+BN8N0DzjefLmvSiIqG845PIzaxbsf2RKezYfqpafRbYdo2czMj2RtNb9qKZuA1j+yCwzvgz93TQEPzA2nsWGu6CNNOg7OnQ7tTgh+nPzW9RWyZYcqad+Gz61ytapX0ugQueMt73T2rYNbp1q6ImES4bqWZ8VudMge8PQb+WFb1dhCXYl63r14KiU29j0OIXAoOwEeXmu7JmATTvdhhqPl7Tmoe7OiCj9gyI5Auo83i0FbEJsNJV9RP95g+cOIFsOEj56Qfm2RuIjWTPRgXzWXzIXMBrHnb3AC6jTOTYmqzcwpCdapPtCsvMeNSANu+hdlDYNpqsWB6QhJ+mJLQGEb+B76+3TUxdxputvry59nQ7lRY+gQc2W2e1E//B/S71v3PqCg4fqTZBKG+ZMw33TjlNZxdZQ5jv9y6EDqPCEpoIYEk/DBmwPXQrCssfhD2rTUe/FNvgb7XVA1k1QcVBSffaDbBvhTmwMZPzJhN+zPhmJOCHVHD2b7Yw0S7PNj2nSR8T4Tly0+oWKoCodnpbLhyEfz9ANy4HvpPNb5lu8UZqpqHtsO6ObBinoOXvrVPnD+/CE+2hS//AgvugFmnwZtnw579wf19Zuc5eOHrTPYfrJ9mfKqZUFedSjuuI9XhcaKd3f+W6qPpLWGZ8EPFUiWaoatZWgzpE+H5bjDvWnjiP1k89vVGXnwz+HFu+w4W3mEGM0vyKz4WwI6l8NC9wft9rnkHbh6fxePfbuSGMVm8fwHk7vROs+ck14lWlRbfLT2zPNbNsevfUkM0vSUsu3RCxVIlmqGr+dl1sPkzKCsyW8flaZQWgSMjjT2nee4+8Xecn19g7aIqK4Km76dx2zuB/30ufwkW3A5pOo0BB+H4NWlsckDWD6aiac25Gu40m3Yy40VLH686xy5r0oiOhctGpNGkQ8PPx5u2wdb0FrFlCiFHYQ788gpkfGUGp/tda1xJgXJnFByAp9JcZxuDiaHHJFOlM1g82Q6OuCmfEd8YLl8IbSxNe/6htBgeb2ltEY5OgNP/DkP/zzvNjK/gh3/DwUxo2tncBMQQYBBbphA2HNhk+qNLCqsmcm1daFxHEz9sWOXNOsew0SxAY5XwdTn88ZP/Y/BEalv3Cb+sGFKODWw8u39xbxIoK4L1c7xP+MefazbBO8KyD18IXz68CAoPOs/aLck3Sf+3ekwkqw9JLVxtgdXxRUXShnDq36zXPo6KgTYnmxtCIFFR4KkfoeaSnYL/kIQvhAwHt5qna6vsUZIPPz8bmDhadIMmHQGLp9bYZBj418DE4Y4eE023UmxSVTdXXAqkHBOcrqZj+7t/84pJgN6XBTaeSCYsE36oWKpE0zvNggMQHeu+7YHcwMU5YQ4kNDHlJMBYA9efkcmxYxz0nFQ/TV/FqRSc/ypc/i30nwY9L4bRL8BftkBp08Bf9+hYGP2ca0XTqFhIbg0Dpvnu3H3RLpQ0vSUsE36oWKpE0zvN5l2tFzoBQMHOwYGLs2V3+MtmGHwPHDcYDk3O4qfTN1J8Q1atg8eB+H0qZQqHjXkBLnwbTrrcPE0H67r3usSMsewZaSyUW/tn0ecqmLrCepGaUPz7DIam12itbbv1799f14fsvGI9Y1GGzs4r9kk70bSP5rwpWj+UqPX9OG8PJ2m97gf7xCmaohkITSuAFdpNThVbphBSlBbDx1fCpo/NjEulzKpGY2eZtUsFIdIRW6YQNsTEw/h3TQGtrB9N/3DBAVO2ufiIGbCM81ALXxAimQYlfKXUBOB+4ERgoNba8nFcKXUu8AwQDbyqtX60IccVhMbt4XAWvDMaysuhJA9iU2D+LXDx59D+jGBHKAj2o6GDtmuBC4DF7hoopaKBF4BRQHfgIqVU9wYe1yOhMsIumvXXdOSZZF+ca5I9wJEyByu6ZfLqBQ63FRUDHadoiqY/Nb2lQQlfa71Ba72plmYDgQyt9VattQN4DxjbkOPWRqiMsItm/TXXzYHyMue2R9fJPSGLte/bI07RFE1/anpLIPrw2wLVI98JnOKmLUqpqcBUgPbt29frgKFS+Eg066+ZvdlMtqpOlzWmTac1aWSf5lkzbw90WpnGL5vhhPMhpbV/4oxUTa1h7XumyFluFjTrYhabnzDMXnGGuqa31OrSUUotBI6x2PVPrfUnFW0WAbdb9eFX9POP1FpfW/H1ZZj+/r/UFpy4dAR3/PoqfHWLa9IHM9t15FPQf4rrvtIimDMefv8W0FULsg++F868299RRw6fTTOlLqpfn9gkU+TsrH8FL65IwJNLp9YuHa31cK11T4vtkzoefydQ/VbVDthVx58VBEt6THJfkEsp6DnZet9Xt5pkX1pYVSe+tAi+fwQ2feq/eCOJPath9ZuuN+OSAlgyHXLdFHYT/E8gZtouB7oopToqpeKAycC8ABxXCGPiGxk3TlyjqkJhscmmZsxFn5n9NXHkw+rZzoXXKinJh+8f8m/MkcKadzzMiAY2fBi4WARnGmrLHAc8B7QEPldKrdJaj1RKtcHYL0drrUuVUjcB8zG2zFla63UNjlyIeI4bDH/bafqKszdD867myT4+1bp97k7P5ZOzt/gnzkijJB90mfW+slLrxVmEwNBQl85crXU7rXW81rq11npkxfd3aa1HV2v3hda6q9a6s9b64YYGXRuhYqkSzYZrxqeadXpHPAEdL3Hwxkr3bZNbQVmJcwGv6lgN3Nr53O2q2ekc86ZlRWljB0vb2iPOcND0FimeFmY2LdF03zaxqVkVKaNPxRqovaraxSaZOvJ2iNNTu/IyyN/n+pRspzi7joHUdhAd59wuOh52Dc/ixXX2iDMcNL0lLEsrhIqlSjQDr3n+a7B9SBpRS419E2WSfdcx0O8a+8RZs53WsOxJ+H56RZdJOXQZA2NeNHXu7RInmIVWrloCn1wNmfNNOYxSB3S/EK59Mo2TN9gjznDQ9BYpniZEHGUO2DAXMr6AmCQ46TJoN8i968cOLPgHLH/e+clexZhkf+N660FqO1CQDUd2QeM0s36A4H+keJogVCM6DnpOotaFSuxCYY5ZzavmGrq6FIpyjPNo4E3Bia02kpqbTbAHYdmHLwjhxPbFrv3hlZQUwDoPZSQEoTqS8AXB5kR5WNYRzGCoINSFsEz4oWKpEk3RrEu71IEOykut28Qmw0lX2CNO0Qy8preEZcIPFUuVaIpmXdp9sj6LEU8YN1F1YhLMZLMeE+0Rp2gGXtNr3K19aIdN1rQVTdGsarfpU61n9NX6wXit/91S64V3a12cZ784RTNwmlYga9oKgiBEBg2qlikIgiCEB5LwBUEQIgRJ+IIgCBFCWCb8ULFUiaZoiqZoNkTTW8Iy4YeKpUo0RVM0RbMhml7jzr5jh01smaIpmqIpmt6B2DIFQRAiA7FlCoIgCJLwBUEQIgVJ+IIgCBGCJHxBEIQIQRK+IAhChCAJXxAEIUKQhC8IghAh2NqHr5TaD2wPdhx1pAVwINhBBBE5fzl/OX97cJzWuqXVDlsn/FBCKbXC3WSHSEDOX85fzt/+5y9dOoIgCBGCJHxBEIQIQRK+73g52AEEGTn/yEbOPwSQPnxBEIQIQZ7wBUEQIgRJ+IIgCBGCJPx6opSaoJRap5QqV0q5tWMppc5VSm1SSmUope4MZIz+RCnVTCm1QCm1peJjUzfttiml1iilVimlQn5xg9qupzI8W7H/N6VUv2DE6S/qcP5DlFKHK673KqXUv4IRpz9QSs1SSu1TSq11s9/2114Sfv1ZC1wALHbXQCkVDbwAjAK6AxcppboHJjy/cyfwjda6C/BNxdfuGKq17hMKPmVP1PF6jgK6VGxTgZcCGqQf8eLv+fuK691Ha/1AQIP0L28A53rYb/trLwm/nmitN2itN9XSbCCQobXeqrV2AO8BY/0fXUAYC8yu+Hw28OfghRIw6nI9xwJvVqw2twxoopQ6NtCB+olw/nuuFa31YiDHQxPbX3tJ+P6lLVB9JeKdFd8LB1prrXcDVHxs5aadBr5WSv2ilJoasOj8Q12uZzhf87qe2yCl1Gql1JdKqR6BCc0W2P7axwQ7ADujlFoIHGOx659a60/qImHxvZDxwXo6fy9kTtda71JKtQIWKKU2VjwphSJ1uZ4hfc1roS7n9iumlkueUmo08DGmiyMSsP21l4TvAa318AZK7ATSqn3dDtjVQM2A4en8lVJ7lVLHaq13V7y27nOjsavi4z6l1FxMt0CoJvy6XM+Qvua1UOu5aa1zq33+hVLqRaVUC621XQqL+RPbX3vp0vEvy4EuSqmOSqk4YDIwL8gx+Yp5wBUVn18BuLzxKKWSlVKNKj8HRmAGu0OVulzPecDlFY6NU4HDlV1fYUCt56+UOkYppSo+H4jJMdkBjzQ42P7ayxN+PVFKjQOeA1oCnyulVmmtRyql2gCvaq1Ha61LlVI3AfOBaGCW1npdEMP2JY8Cc5RS1wA7gAkA1c8faA3Mrfj/jwHe0Vp/FaR4G4y766mUmlaxfwbwBTAayAAKgKuCFa+vqeP5jweuV0qVAoXAZB0m0/mVUu8CQ4AWSqmdwH1ALITOtZfSCoIgCBGCdOkIgiBECJLwBUEQIgRJ+IIgCBGCJHxBEIQIQRK+IAhChCAJXxAEIUKQhC8IghAh/D+aFbJfqawWjwAAAABJRU5ErkJggg==) 

#### 理解meshgrid和vstack的作用

* ravel():拉平
* vstack():拉成竖直
* meshgrid():两两组合

* 重要接口decision_function：
  返回每个输入样本所对应的到决策边界的距离



### 画图函数





```python
#将上述过程包装成函数：
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
ax = plt.gca()
def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    #组合x,y坐标
    Y,X = np.meshgrid(y,x)
    #拉平
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    #转化成一样的格式，P就是Z
    P = model.decision_function(xy).reshape(X.shape)
    #contour计算距离的函数
    ax.contour(X, Y, P,colors="k",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"]) 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

clf = SVC(kernel = "linear").fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
#画三条线
plot_svc_decision_function(clf)

#支持向量就是那三个边界点
```

 ![img](data:image/png;base64,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) 

### 三维可视化



```python
#定义一个由x计算出来的新维度r，X的平方的倒数取对数
r = np.exp(-(X**2).sum(1))

rlim = np.linspace(min(r),max(r),100)
rlim
```



```python
from mpl_toolkits import mplot3d
#定义一个绘制三维图像的函数
#elev表示上下旋转的角度
#azim表示平行旋转的角度

def plot_3D(elev=30,azim=30,X=X,y=y):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:,0],X[:,1],r,c=y,s=50,cmap='rainbow')
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    plt.show()
    
plot_3D()   

#加入一个维度后，就可以找出超平面了
```

 ![img](data:image/png;base64,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vPPOS1lxVk9PDzU1NWRmZqJpmik7D63Ara2t5XOf+xwPP/xwXMK8UwHf+c538Hq9eDweqqqq+Na3vjXZQ6bJYjoxTf0dTkkEg8EwT45169axZcsWPvjBDyZt8T8wMEBNTQ3V1dVkZmaahr6dnZ14vV7q6urIycnhZz/7GQ888ABr165N8bs6efD7/WzatImMjAxef/31VMSJ0rUhqcYFF1zAmjVrxvxEav3Hw2uvvcauXbt49tln+c1vfsMrr7wyhVc8M2C1WjnvvPO488472bt3L1/+8pfZsWMHH/jAB/jMZz7DY489xvDwcNzHGxwcpKamJqy1oc1mo7y8nOrqajZt2oTf7+f73/8+fr+f++67z3TLmiyee+45li9fzpIlS/jxj3885vWXXnqJvLw8qqurqa6u5vvf/35KzhsKt9vN0NAQg4ODeL3elB8/JqSUM/nnlMGyZctkW1ublFLKtrY2uWzZsgn3+d73vid/9rOfTfWlzVioqirfeust+V//9V9y/fr1csuWLfIPf/iDbG9vl8PDw1F/Ojo65N///nfpcrliblNbWyvXrVsnd+7cKT0ej3zqqadkf3//pK83GAzKRYsWyYaGBunz+eS6detkTU1N2DYvvviivOyyyyZ9rvFwxRVXyD//+c/y9ttvl1/+8pdTcci4xmN6ZpEiXHnlldx7770A3HvvvVHl5cPDwwwODpr/f/755ycbmDqloSgKmzZt4ic/+Qm7du3i1ltvpaGhgUsvvZSPfOQj3H///WFLuaGhIQ4cOMC6detiFny1trbyiU98gt/+9recfvrpZGRkcPnll5Obmzvp642ngnSqcd9992G1WvnEJz7BzTffzM6dO/nHP/4xPSePl1VO0s8pg+7ubnn++efLJUuWyPPPP1/29PRIKaVsbW2Vl1xyiZRSyoaGBrlu3Tq5bt06uWrVKnn77befzEuesdA0TR44cEDeeuutctOmTfKiiy6St956qzz//PPHnVEcOXJEbtiwQb788stTcl2PPvqovO6668zf77vvvjFP9hdffFEWFhbKdevWyQ996EPywIEDU3ItKUZc4/Fkk8GsIYupxLPPPiuXLVsmFy9eLH/0ox+NeV3TNPmVr3xFLl68WK5du1a+8847J+Eqpwaapslnn31WlpeXy3POOUd+4AMfkHfccYdsaGiQQ0NDJlEcPXpUbtiwQb7wwgtTdi2PPPLIGLK48cYbw7bp7++Xg4ODUkopn3nmGblkyZIpu54UIr0MmQ2Ix5Dn2Wefpb6+nvr6en73u99xww03nKSrTT2EELzxxhs8//zzvPLKK/z+97/H7/fz6U9/mksuuYS77rqL/fv385GPfIQf/ehHnH/++VN2LfEUGubm5pqp20svvdTM0swKxMsqJ+nnXY/XX39dXnTRRebvP/zhD+UPf/jDsG2+8IUvyAceeMD8PTTYOluhaZpsbm6Wd9xxh5w/f7783e9+N+XnDAQCcuHChfLo0aNmgDNymdHe3i41TZNSSvnmm2/KuXPnmr/PYMQ1HtNVpzMcs8lpKZUQQlBVVcVNN93ETTfdNC3njKeCdOvWrfy///f/sFqtZGZm8tBDD80aT5Q0WcxwSDl7nJZmAy699FIuvfTSsL996UtfMv9/4403cuONN073ZU0L0jGLGY7Z5LSUxqmNNFnMcIQa8vj9fh566CGuvPLKsG2uvPJK7rvvPqSU7NixY8Y6LaVxaiO9DJnhmE1OS2mc2kgXkqWRRhrpQrI0EsNMKJJKYwYj3hzrSfpJY5owU4qkpguf/exnZUlJiVy9enXU12ezKjYK0grONOLHTCiSmk5ce+21PPfcczFfn82q2GSRJos0gPi7sb3xxhusX7+eSy65hJqamum8xJTife97H4WFhTFfP1X6j04n0tmQNID4hF0bN26kqamJ7Oxstm/fzpYtW6ivr5+uS5xWvBtVsRMhPbNIA0gXSUUiHvJ8tyFNFicZ3/3ud/nlL39p/v7f//3f/OpXv5r264hH/NXR0WEOorfeegtN06a09d/JRFoVOxYzXWcx6yGEWAD8RUq5UQihAPXAGVLKnpNwLZcCdwAW4A9Syh8IIb4EIKW8WwhxI3ADEAQ8wNellK9P93WmCqOf/dNSyjF2ZUKIy4AbgUuBM4FfSSnPmN4rnFlIk8UMgBDib8B/AWXA9VLKq0/yJU0phBB/AC4HXDEGqgB+iT5QR4BrpZS7UnwNDwLnAcVAJ/A9wAYmMQrgTuBDo9fwWSnlqdkRKkVIk8UMgBDiY8B7gDnAvVLK7Sf5kqYUQoj3AUPAfTHI4lLgK5x4qv9SSnnm9F5lGpFIxyxmBh5Df4JtAv56kq9lyiGlfAVwj7PJZnQikVLKHUC+EOLdm4aYIUinTmcApJR+IcSLQJ+UUj3Z1zMDUAk0h/zeMvq3d7fQ4SQjTRYzAKOBzbOAj5zsa5khiJajTK+XTzLSy5CTDCHEKuAI8IKUcnYqnBJHCzA35PcqoO0kXUsao0jPLE4ypJS1wKKTfR0zDE8CNwohHkIPcPZLKdNLkJOMNFmkMe0ITVsKIVqISFsC29EzIUcYTVuenCtNIxTp1GkaaaQRF9IxizTSSCMupMkijTTSiAtpskgjjTTiQpos0kgjjbiQJos00kgjLqTJIo000ogLabJII4004kKaLNJII4248P8DgGb8DFxbArgAAAAASUVORK5CYII=) 

```python
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles

X,y = make_circles(100, factor=0.1, noise=.1)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")

def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y,X = np.meshgrid(y,x) 
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X, Y, P,colors="k",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
clf = SVC(kernel = "linear").fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plot_svc_decision_function(clf)


r = np.exp(-(X**2).sum(1))
rlim = np.linspace(min(r),max(r),100)

from mpl_toolkits import mplot3d
def plot_3D(elev=30,azim=30,X=X,y=y):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:,0],X[:,1],r,c=y,s=50,cmap='rainbow')
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    plt.show()
    
from ipywidgets import interact,fixed
#jupyter notebook独有，设置下拉菜单中的选项，即图的旋转角度
#更换X=fixed(X),y=fixed(y)

interact(plot_3D,elev=[0,30,60,90,120],azip=(-180,180),X=fixed(X),y=fixed(y))
plt.show() 
```



数据在三维空间中，我们的超平面就是一个二维平面。

#### 核函数

一种能够使用数据原始空间中的向量计算来表示升维后的空间中的点积的数学方式。

>我们刚才做的，计算r，并将r作为数据的第三维度来讲数据升维的过程，被称为“核变换”，即是将数据投影到高维空间中，以寻找能够将数据完美分割的超平面，而在高维空间中计算来找出超平面的函数就叫做核函数。

---

在SVM中，这个功能由参数“kernel”控制。之前我们一直使用这个参数，但是没有给大家解释，我们使用的是“linear"，线性核函数，只能用于线性的情况。

* 参数kernel：

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C26.png)



我们可能不知道什么样的数据应该使用什么类别的核函数来确保可以在变换的空间出找出决策边界

* plot_svc_decision_function(clf) :画出边界

刚才我们使用的计算r的方法，其实是高斯径向基
核函数，在参数”kernel“中输入”rbf“就可以使用。我们来看看模型找出的决策边界时什么样：

```python
#使用高斯径向基进行核变换，然后找出划分边界
clf = SVC(kernel = "rbf").fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plot_svc_decision_function(clf)
```

 ![img](data:image/png;base64,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) 

### 探索核函数在不同数据集上的表示

除了"linear"以外的核函数都能够处理非线性情况，那究竟什么时候选哪个呢？

```python
nrows = len(datasets)
ncols = len(Kernel)+1

fig,axes = plt.subplots(nrows,ncols,figsize=(20,16))
#子图循环，ds_cnt是行

for ds_cnt ,(X,Y) in enumerate(datasets):
    #一个索引对应一个二维数组 [(索引，array([特征矩阵X]，[标签Y]))]
    
#在图像的第一列，放置原始数据的分布,一列一种数据
    ax = axes[ds_cnt,0]
    if ds_cnt ==0 :#打一个标题
        ax.set_title("Input data")
    #横轴是第一个特征，纵轴是第二个图，zorder是让图像放在上面还是放在下面越大越往上
    ax.scatter(X[:,0],X[:,1],c=Y,zorder =10,cmap=plt.cm.Paired,edgecolors='k')
    #不显示横纵坐标
    ax.set_xticks(())
    ax.set_yticks(())
    
    #第二层循环：在不同的核函数中循环，从图像的第二列开始，一个个填充分类结果
    for est_idx ,kernel in enumerate(Kernel):
        #定义子图位置
        ax= axes[ds_cnt ,est_idx+1] #第一列放的是原始数据，所以跳过
        
        #建模
        clf = svm.SVC(kernel=kernel ,gamma=2).fit(X,Y)
        score = clf.score(X,Y)
        
        #绘制图像本身分布的散点图
        ax.scatter(X[:,0],X[:,1],c=Y
                ,zorder=10,cmap=plt.cm.Paired,edgecolors='k')
        
        #绘制支持向量
        ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1]
                  ,s=50,facecolors='none',zorder=10,edgecolors='k')
        #facecolors='none'是透明的意思，就不要覆盖掉原来的颜色
        
        #绘制决策边界
        #为了不要让图像的点落到图像的边界上，坐标轴的边界要宽出来一点
        x_min , x_max = X[:,0].min()- 0.5,X[:,0].max()+0.5
        y_min , y_max = X[:,1].min()- 0.5,X[:,1].max()+0.5
        
        #np.mgrid,合并了我们之前使用的np.linkspace和np.meshgrid的用法
        #一次性使用最大值和最小值来生成网络
        #表示为[起始值：结束值：步长]
        #如果步长是复数，则其整数部分就是起始值和结束值之间创建的点数量，并且结束值被包含在内
        XX,YY = np.mgrid[x_min:x_max:200j,y_min:y_max:200j]
        #形成XX和YY两个网格，步长为200j 生成200个 ，j的意思是包括结束值
        
        
        #np.c_ ,类似于np.vstack的功能--》竖直
        Z = clf.decision_function(np.c_[XX.ravel(),YY.ravel()]).reshape(XX.shape)
        
        #pcolormesh是拿来填充等高线不同区域的颜色 ，Z是距离，Z>0一个颜色，Z<0一个颜色
        ax.pcolormesh(XX,YY, Z>0 ,cmap=plt.cm.Paired,shading="auto")
    
        #绘制等高线，'k'是黑色
        ax.contour(XX,YY,Z,colors=['k','k','k'],linestyles=['--','-','--'],
                  levels=[-1,0,1])
        #不显示横纵坐标
        ax.set_xticks(())
        ax.set_yticks(())
        
        #将标题放在第一行的顶上,标题就是核函数的名字
        if ds_cnt ==0 :
                ax.set_title(kernel)
                
        #text就是再坐标上写文字，为每张图添加分类的分数，lstrip('0')是不显示0
        ax.text(0.95, 0.06,('%.2f'% score).lstrip('0')
               ,size=15
               ,bbox=dict(boxstyle='round',alpha=0.8,facecolor='white')
               #bbox就是为分数添加一个圆形白色的格子作为底色
                ,transform = ax.transAxes #确定文字所对应的坐标轴，就是ax子图的坐标轴本身
                ,horizontalalignment='right'  #位于坐标轴的上面方向
                )
#图像之间紧缩显示
plt.tight_layout()
plt.show()
        
```



可以观察到，线性核函数和多项式核函数在非线性数据上表现会浮动，如果数据相对线性可分，则表现不错，如果数据线性不可分，则表现非常糟糕

Sigmoid核函数，在非线性数据上强于两个线性的核函数，但效果明显不如rbf，在线性数据上又比不上线性核函数

rbf和高斯径向基核函数几乎在所有的数据集上表现都不错，属于比较万能的核函数。



### 支持向量机的参数列表：

* probability: 是否启用概率估计。

> 进行必须在调用fit之前，这个功能会减慢SVM的运算速度
> 设置为True就是启动，启用以后，predict_proba和predict_log_proba将生效

在二分类情况下，SVC将使用Platt缩放来生成概率

* predict_proba():生成的各类标签下的概率
* decision_function(): 标签点到决策边界的距离
* 具体运行代码：

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C40.png)

### SVM实现概率预测

decision_function就是到决策边界的距离：返回的值也被认为是置信度，不过置信度终究不是概率，它没有边界，可以无限大。
为了解决这个矛盾，SVC有重要参数probability

* probability：布尔值，是否使用概率估计，强行压缩成[0，1]之间的概率





##  XGBoost

XGBoost本身的核心是基于梯度提升树实现的集成算法。

整体来说可以有三个核心部分：
1.集成算法本身
2.用于集成的弱评估器
3.以及应用中的其他过程
`import xgboost as xgb`

**有两种方式使用xgboost库**
* 第一种，之间调用xgb
![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C61.png)
>1.必须使用xgb.DMatrix()来读取数据
>2.必须先设置参数
>3.必须先导出预测结果

* 使用xgboost库中的sklearn的API
这样使用的流程就是 实例化，fit，predict

xgboost.XGBRegressor

**两种方法参数有点不同**

### 梯度提升树GBDT

XGB是GBDT的改进
>class xgboost.XGBRegressor (max_depth=3, 
>learning_rate=0.1,
>n_estimators=100, 
>silent=True,
>objective='reg:linear',
>booster='gbtree',
>n_jobs=1,
>nthread=None,
>gamma=0, min_child_weight=1, max_delta_step=0,
>subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
>base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain', **kwargs

---
集成算法通过在数据上构建多个弱评估器，汇总所有弱评估器的建模结果，以获取比单个模型更好的回归或分类表现。弱评估器被定义为是表现至少比随机猜测更好的模型，即预测准确率不低于50%的任意模型

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C62.png)

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C63.png)

对于梯度提升回归树来说，每个样本的预测结果可以表示为所有树上的结果的加权求和
![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C64.png)

**对于XGB来说，每个叶子节点上会有一个预测分数（prediction score），也被称为叶子权重。这个叶子权重就是所有在这个叶子节点上的样本在这一棵树上的回归取值**

假设这个集成模型中总共有 棵决策树，则整个模型在这个样本 上给出的预测结果为：

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C65.png)
#### XGB和GBDT的区别：
>GBDT中预测值是由所有弱分类器上的预测结果的加权求和，其中每个样本上的预测结果就是样本所在的叶子节
>点的均值。而XGBT中的预测值是所有弱分类器上的叶子权重直接求和得到，计算叶子权重是一个复杂的过程

在集成中我们需要的考虑的第一件事是我们的超参数 ，究竟要建多少棵树呢？
![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C66.png)

**slient是监控建树的参数**



### 实战代码

```python
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime


data = load_boston()
#波士顿数据集非常简单，但它所涉及到的问题却很多
X = data.data
y = data.target

#划分数据集
Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)
#实例化
reg = XGBR(n_estimators=100).fit(Xtrain,Ytrain)
reg.predict(Xtest) #传统接口predict
reg.score(Xtest,Ytest) 

#你能想出这里应该返回什么模型评估指标么？
MSE(Ytest,reg.predict(Xtest))
reg.feature_importances_ #树模型的优势之一：能够查看模型的重要性分数，可以使用嵌入法进行特征选择

reg = XGBR(n_estimators=100)
#交叉验证中导入没有经过训练的模型

CVS(reg,Xtrain,Ytrain,cv=5)
CVS(reg,Xtrain,Ytrain,cv=5).mean()
#交叉验证返回的和score是一样的R平方

```

##### 交叉验证注意点：

通常的交叉验证：输入全数据集，直接分割数据集，训练集和测试集。

测试的数据包含训练的数据

严谨的交叉验证：分成K折，交叉验证



#### 学习曲线

**如果开启了参数slient，数据巨大，遇到算法运行会非常缓慢的时候可以使用这个参数**

slient = False ：开启

```python
#学习曲线函数

def plot_learning_curve(estimator,title,X,y,
                       ax=None,
                       ylim=None,
                       cv=None,
                       n_jobs=None):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    
    train_sizes ,train_scores ,test_scores = learning_curve(
                        estimator,X,y,shuffle=True
                        ,cv=cv
#                 ,random_state=420
                ,n_jobs=n_jobs)
    
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()
    ax.plot(train_sizes ,np.mean(train_scores,axis=1),'o-'
           ,color="r",label="Training score")
    ax.plot(train_sizes,np.mean(test_scores,axis=1),'o-'
           ,color='g',label="Test score")
    ax.legend(loc="best")
    
    return ax

#设置交叉验证的模式---采用k折交叉验证
cv = KFold(n_splits=5,shuffle=True ,random_state = 42) 
plot_learning_curve(XGBR(n_estimators=100,random_state=420),
                   "XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()

#典型过拟合！！！必然调参啦
```

 ![img](data:image/png;base64,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) 

### 调参

#### class xgboost.XGBRegressor

参数：

- n_estimators :集成算法中的弱分类器的数量
- learning_rate :集成中的学习率
- silent: 是否在运行集成时进行流程的打印
- subsample :从样本中进行采样的比例
- max_depth :弱分类器的最大树深度
- objective :指定学习目际函数与学习任务
- booster :指定要使用的弱分类器
- gamma :步长，在树的叶节点上进行进一步分枝所需的最小的目际函数的下降
- min_child_weight :一个叶节点上所需的最小样本权重
- max_delta_step :树的权重估计中允许的单次最大增量
- colsample_bytree : 构造一棵树时随机油样出的特征占所有特征的占比
- colsample_bylevel: 在树的一层进行分支时随机抽样出的特征占所有持征的比例
- reg_alpha：目标函数中使L1正则化时控制正则化强度
- reg_lambda :目标函数中使用L2 正则化时控制正则化强度
- n_jobs :用于运行xgboost 的并行线程数数量
- scale_pos_weight : 处理标签中的样本不平衡问题
- base_score :全局偏差
- random_state:随机数种子
- missing:需要作为缺失值存在的数中的值· 如果为None, 则默认为np 、nan.
- importance_type:feature importances 属性的特征重要性类型

```
1. booster [default: gbtree]
    a: 表示应用的弱学习器的类型, 推荐用默认参数
    b: 可选的有gbtree, dart, gblinear
        gblinear是线性模型，表现很差，接近一个LASSO
        dart是树模型的一种，思想是每次训练新树的时候，随机从前m轮的树中扔掉一些，来避免过拟合
        gbtree即是论文中主要讨论的树模型，推荐使用

2. silent [default: 0] [不推荐]
    a: 不推荐使用，推荐使用verbosity参数来代替，功能更强大

3. verbosity [default: 1]
    a: 训练过程中打印的日志等级，0 (silent), 1 (warning), 2 (info), 3 (debug)

4. nthread [default: 最大可用线程数][alias: n_jobs]
    a: 训练过程中的并行线程数
    b: 如果用的是sklearn的api，那么使用n_jobs来代替
    
 -------
    
1. eta [default: 0.3] [alias: learning_rate] [range: [0, 1]]
    a: 就是常说的学习速率，控制每一次学习的权重缩减，给后来的模型提供更多的学习空间

2. gamma [default: 0] [alias: min_split_loss] [range: [0, inf]]
    a: 叶子节点分裂时所需要的最小的损失减少量，这个值越大，叶子节点越难分裂，所以算法就越保守

3. max_depth [default: 6] [range: [0, inf]]
    a: 树的最大深度
    b: 这个值对结果的影响算是比较大的了，值越大，树的深度越深，模型的复杂度就越高，就越容易过拟合
    c: 注意如果这个值被设置的较大，会吃掉大量的内存
    d: 一般来说比价合适的取值区间为[3, 10]

4. min_child_weight [default: 1] [range: [0, inf]]
    a: 最小的叶子节点权重
    b: 在普通的GBM中，叶子节点样本没有权重的概念，其实就是等权重的，也就相当于叶子节点样本个数
    c: 越小越没有限制，容易过拟合，太高容易欠拟合
5. max_delta_step [default: 0] [range: [0, inf]]
    a: 适用于正负样本不均衡情况下，控制学习速率(类似eta)最大为某个值，不能超过这个阈值
    b: 首先我们有参数eta来控制学习速率，为了后面学习到更多，每一步在权重上乘上这个因子，降低速度
    c: 但是在正负样本不均衡的情况下eta不足够，因为此时由于二阶导接近于0的原因，权重会特别大
    d: 这个参数就是用来控制学习速率最大不能超过这个数值

6. sub_sample [default: 1] [range: (0, 1]]
    a: 样本抽样比例
    b: 在每次训练的随机选取sub_sample比例的样本来作为训练样本

7. colsample_by* [default: 1]
    a: 这里实际上有3个参数，借助了随机森林的特征抽样的思想，3个参数可以同时使用
    b: colsample_bytree   更常用，每棵树的特征抽样比例
    c: colsample_bylevel  每一层深度的树特征抽样比例
    d: colsample_bynode   每一个节点的特征抽样比例

8. lambda [default: 1] [alias: reg_lambda]
    a: 损失函数中的L2正则化项的系数，类似RidgeRegression，减轻过拟合

9. alpha [default: 0] [alias: reg_alpha]
    a: 损失函数中的L1正则化项的系数，类似LASSO，减轻过拟合

10. scale_pos_weight [default: 1]
    a: 在正负样本不均衡的情况下，此参数需要设置，通常为: sum(负样本) / sum(正样本)

学习目标参数：

1. objective [default: reg:squarederror(均方误差)]
    a: 目标函数的选择，默认为均方误差损失，当然还有很多其他的，这里列举几个主要的
    b: reg:squarederror       均方误差
    c: reg:logistic           对数几率损失，参考对数几率回归(逻辑回归)
    d: binary:logistic        二分类对数几率回归，输出概率值
    e: binary:hinge           二分类合页损失，此时不输出概率值，而是0或1
    f: multi:softmax          多分类softmax损失，此时需要设置num_class参数

2. eval_metric [default: 根据objective而定]
    a: 模型性能度量方法，主要根据objective而定，也可以自定义一些，下面列举一些常见的
    b: rmse : root mean square error     也就是平方误差和开根号
    c: mae  : mean absolute error        误差的绝对值再求平均
    d: auc  : area under curve           roc曲线下面积
    e: aucpr: area under the pr curve    pr曲线下面积
    
工具包参数：
1. num_boost_round
    a: 迭代次数，这货其实跟sklearn中的n_estimators是一样的
    b: sklearn的api中用n_estimators，原始xgb中用num_boost_round

2. evals
    a: 训练过程中通过计算验证集的指标，观察模型性能的数据集
    b: 指标就是通过eval_metric参数来制定的

3. early_stopping_rounds
    a: 在num_boost_round的轮训练中，如果过程中指标经过early_stopping_rounds轮还没有减少
那么就停止训练
    b: 指标是通过evals的验证集，计算eval_metric的指标
    
   early_stopping_rounds这个参数，首先触发这个参数（也就是确实提前停止了）的时候返回的变量会带有3个属性：best_score, best_iteration, best_ntree_limit ，这里best_ntree_limit 就是最好的模型的树的个数
```





#### n_estimators 调参

```python
#调一下n_estimator对模型的影响
axisx = range(10,1010,50)

rs=[]
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    rs.append(CVS(reg,Xtrain,Ytrain,cv=cv).mean())
    
print(axisx[rs.index(max(rs))],max(rs))#打印第几个是最大的

plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()
```

 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABJIAAAEvCAYAAAAegesTAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAqbUlEQVR4nO3dfbBddZ3n+/eXPBAkAXk4REiCiRAeIySQB1BntKGR2K3SXdVT4r3tUEzXpKBxLtjaA3Z39VSPM9VO2zr0EHooWrh4e6hGp43TaOFFyuuMQ9XeSQ4hEEJEYhByCEpIkEdDOOR7/1grYXNykrNOzslZ++H9qjq19vrttfb5ruCviB9+v++OzESSJEmSJEkayRF1FyBJkiRJkqTOYJAkSZIkSZKkSgySJEmSJEmSVIlBkiRJkiRJkioxSJIkSZIkSVIlBkmSJEmSJEmqZHLdBYzGiSeemHPnzq27DEmSJEmSpK7x0EMPvZCZfVWu7aggae7cufT399ddhiRJkiRJUteIiKerXuvWNkmSJEmSJFVikCRJkiRJkqRKDJIkSZIkSZJUSUf1SBrOm2++ycDAALt27aq7lMNi2rRpzJ49mylTptRdiiRJkiRJ6nEdHyQNDAwwY8YM5s6dS0TUXc64ykx27NjBwMAA8+bNq7scSZIkSZLU4zp+a9uuXbs44YQTui5EAogITjjhhK5dbSVJkiRJkjpLxwdJQFeGSHt187NJkiRJkqTO0hVBUp22bt3KvHnz2LlzJwAvvvgi8+bN4+mnn+bJJ5/k4x//OKeddhoXXnghv/Ebv8GPf/xjAO666y76+vpYuHAh5557Lr/3e7/H66+/XuejSJIkSZIkHVSlICkilkfEExGxOSJuGub9YyPiuxHxSERsjIiry/FpEbGmZfwvWu75SkT8JCIejYjvRMS7x+2pJtCcOXO49tpruemm4o/lpptuYsWKFcycOZPf/u3fZsWKFfzsZz/joYce4pZbbmHLli377v3Upz7F+vXr2bhxI1OnTuWb3/xmXY8hSZIkSZI0ohGbbUfEJOBW4DJgAFgbEfdm5uMtl10HPJ6Zn4iIPuCJiLgbeAO4JDNfjYgpwIMR8f3MbAIPAF/MzMGI+E/AF4Ebx/fxJsbnPvc5LrzwQm6++WYefPBBbrnlFv7+7/+eiy++mE9+8pP7rluwYAELFizY7/7BwUFee+01jjvuuIksW8PJhJdeguefL3527IA9e4rx1muGvp6IsfH8HEmSJEnS2BxxBPzBH9RdxYSr8q1tS4HNmbkFICLuAa4AWoOkBGZE0dBnOrATGMzMBF4tr5lS/iRAZv6g5f4m8HtjeI5aTZkyha985SssX76cH/zgB0ydOpWNGzdywQUXHPS+b37zmzz44IM899xznHHGGXziE5+YoIp7zK5dbwdDVX7efLPuiiVJkiRJ7W7qVIOkA5gFbG05HwCWDblmJXAvsA2YAXwqM/fAvhVNDwGnA7dm5uphfse/Asa+r+uGG2D9+jF/zDssXAg33zziZd///vc5+eSTeeyxx7jsssv2e/93f/d3efLJJznjjDNYtWoVUGxtW7lyJZnJddddx1e+8pV9W+R0EG+9BTt3Vg+GXn55+M+ZNg1mzoSTToJTTin+WZ900jt/TjgBJrdMk73Nz1uboE/k2Hh+jiRJkiRJo1QlSBru/3kO3SNzObAeuAQ4DXggIv53Zr6cmW8BC8seSN+JiAWZ+di+D4/4U2AQuHvYXx6xAlgBcOqpp1Yod+KtX7+eBx54gGazyYc+9CGuvPJKzj333H2NtQG+853v0N/fzxe+8IX97o8IPvGJT3DLLbf0ZpCUCa++Wj0YeuGFYrvZUEccAX19b4dAS5bsHwy1/hx9tMGKJEmSJEmjUCVIGgDmtJzPplh51Opq4MvlVrbNEfEUcBawZu8FmfmriPifwHLgMYCIuAr4OHBpee9+MvN24HaAxYsXH7zJS4WVQ+MtM7n22mu5+eabOfXUU/njP/5jvvCFL/D1r3+dv/zLv+Tee+/d1yfpYN/K9uCDD3LaaadNVNmH3+7dReCzN/z55S8PHg7t2jX85xx77NvBz/z58MEPHjgYOv74IkySJEmSJEmHRZUgaS0wPyLmAc8CVwL/x5BrngEuBf53RMwEzgS2lI233yxDpKOA3wT+ExTfBEfRXPvDmdmx33v/d3/3d5x66qn7trP94R/+IXfddRdr1qzhe9/7Hn/0R3/EDTfcwMyZM5kxYwZ/9md/tu/evT2S9uzZw+zZs7nrrrtqeopxcM898Ld/+3Yw9OKLw183derb28lOOgnOPffAwVBfHxx55MQ+hyRJkiRJOqA4wEKgd14U8VvAzcAk4M7M/I8RcQ1AZt4WEacAdwEnU2yF+3Jm/reIOA/4RnnfEcC3MvPfl5+5GTgS2FH+mmZmXnOwOhYvXpz9/f3vGNu0aRNnn312taftUB3xjIsXw8AAfPjDw4dCe8OjGTPcTiZJkiRJUhuJiIcyc3GVa6usSCIz7wPuGzJ2W8vrbcBHh7nvUWDRAT7z9Cq/Wx3g9dfhkUfgxhvhP/yHuquRJEmSJEmHiQ1lNHYPPQSDg3DRRXVXIkmSJEmSDiODJI1ds1kcly2rtw5JkiRJknRYdUWQVKXPU6fqiGdrNOD004vm2JIkSZIkqWt1fJA0bdo0duzY0RmByyhlJjt27GDatGl1l3JgmUWQ5LY2SZIkSZK6XqVm2+1s9uzZDAwMsH379rpLOSymTZvG7Nmz6y7jwJ55Bn7xC7j44rorkSRJkiRJh1nHB0lTpkxh3rx5dZfRu/b2R3JFkiRJkiRJXa/jt7apZs0mHHUUvP/9dVciSZIkSZIOM4MkjU2jAUuWwJQpdVciSZIkSZIOM4MkHbo33oCHH3ZbmyRJkiRJPcIgSYfu4Ydh924bbUuSJEmS1CMMknToGo3iuGxZvXVIkiRJkqQJYZCkQ9dswnvfCyefXHclkiRJkiRpAhgk6dA1Gm5rkyRJkiSphxgk6dA8+yxs3WqjbUmSJEmSeohBkg7N6tXF0RVJkiRJkiT1DIMkHZpGA448EhYurLsSSZIkSZI0QQySdGiaTbjgApg6te5KJEmSJEnSBDFI0ujt3g39/W5rkyRJkiSpxxgkafQefRR27bLRtiRJkiRJPcYgSaPXbBZHgyRJkiRJknqKQZJGr9GAWbNgzpy6K5EkSZIkSRPIIEmj12y6GkmSJEmSpB5kkKTRef552LLFRtuSJEmSJPUggySNjv2RJEmSJEnqWZWCpIhYHhFPRMTmiLhpmPePjYjvRsQjEbExIq4ux6dFxJqW8b9ouef4iHggIp4sj8eN32PpsGk2YfJkuOCCuiuRJEmSJEkTbMQgKSImAbcCHwPOAT4dEecMuew64PHMPB/4CPDViJgKvAFcUo4vBJZHxN6lLDcBP8zM+cAPy3O1u0YDFi2Co46quxJJkiRJkjTBqqxIWgpszswtmbkbuAe4Ysg1CcyIiACmAzuBwSy8Wl4zpfzJ8vwK4Bvl628Av3PIT6GJMTgIa9e6rU2SJEmSpB5VJUiaBWxtOR8ox1qtBM4GtgEbgOszcw8UK5oiYj3wPPBAZq4u75mZmc8BlMeThvvlEbEiIvojon/79u3VnkqHx8aN8NprNtqWJEmSJKlHVQmSYpixHHJ+ObAeOIViC9vKiDgGIDPfysyFwGxgaUQsGE2BmXl7Zi7OzMV9fX2juVXjrdEojq5IkiRJkiSpJ1UJkgaAOS3nsylWHrW6GlhVbmXbDDwFnNV6QWb+CvifwPJy6JcRcTJAeXx+tMVrgjWbcNJJMHdu3ZVIkiRJkqQaVAmS1gLzI2Je2UD7SuDeIdc8A1wKEBEzgTOBLRHRFxHvLsePAn4T+El5z73AVeXrq4B/GsNzaCI0GsW2thhukZokSZIkSep2IwZJmTkIfBa4H9gEfCszN0bENRFxTXnZl4APRMQGim9guzEzXwBOBn4UEY9SBFIPZOb3ynu+DFwWEU8Cl5Xnalc7dsBPf+q2NkmSJEmSetjkKhdl5n3AfUPGbmt5vQ346DD3PQosOsBn7qBcxaQOsGZNcTRIkiRJkiSpZ1XZ2iYV29qOOAKWLKm7EkmSJEmSVBODJFXTbMJ558HRR9ddiSRJkiRJqolBkka2Zw+sXl002pYkSZIkST3LIEkj27QJXn7Z/kiSJEmSJPU4gySNrNksjgZJkiRJkiT1NIMkjazRgOOPh/nz665EkiRJkiTVyCBJI2s2i9VIEXVXIkmSJEmSamSQpIN76SV4/HEbbUuSJEmSJIMkjWDNGsi0P5IkSZIkSTJI0giazWJL29KldVciSZIkSZJqZpCkg2s04Nxz4Zhj6q5EkiRJkiTVzCBJB5b5dqNtSZIkSZLU8wySdGA//Sm8+KJBkiRJkiRJAgySdDDNZnH0G9skSZIkSRIGSTqYZhOOPRbOOqvuSiRJkiRJUhswSNKBNRqwbBkc4f9MJEmSJEmSQZIO5NVXYcMG+yNJkiRJkqR9DJI0vP5+2LPHIEmSJEmSJO1jkKThNRrFcdmyeuuQJEmSJEltwyBJw2s24cwz4fjj665EkiRJkiS1CYMk7S+zWJF08cV1VyJJkiRJktqIQZL299RTsH27/ZEkSZIkSdI7GCRpf81mcTRIkiRJkiRJLSoFSRGxPCKeiIjNEXHTMO8fGxHfjYhHImJjRFxdjs+JiB9FxKZy/PqWexZGRDMi1kdEf0QsHb/H0pg0GnD00bBgQd2VSJIkSZKkNjJikBQRk4BbgY8B5wCfjohzhlx2HfB4Zp4PfAT4akRMBQaBz2fm2cBFwHUt9/4V8BeZuRD48/Jc7aDZhKVLYdKkuiuRJEmSJEltpMqKpKXA5szckpm7gXuAK4Zck8CMiAhgOrATGMzM5zJzHUBmvgJsAma13HNM+fpYYNuYnkTj49e/hvXrbbQtSZIkSZL2M7nCNbOArS3nA8CyIdesBO6lCINmAJ/KzD2tF0TEXGARsLocugG4PyL+miLQ+sAoa9fh8NBDMDhofyRJkiRJkrSfKiuSYpixHHJ+ObAeOAVYCKyMiL2rjYiI6cC3gRsy8+Vy+Frgc5k5B/gccMewvzxiRdlDqX/79u0VytWY7G20vWxoVihJkiRJknpdlSBpAJjTcj6b/behXQ2sysJm4CngLICImEIRIt2dmata7rkK2Hv+3ym20O0nM2/PzMWZubivr69CuRqTRgNOOw1OOqnuSiRJkiRJUpupEiStBeZHxLyygfaVFNvYWj0DXAoQETOBM4EtZc+kO4BNmfm1IfdsAz5cvr4EePLQHkHjJrMIktzWJkmSJEmShjFij6TMHIyIzwL3A5OAOzNzY0RcU75/G/Al4K6I2ECxFe7GzHwhIj4EfAbYEBHry4/8k8y8D/jXwN9ExGRgF7BinJ9No7V1Kzz3nEGSJEmSJEkaVpVm25TBz31Dxm5reb0N+Ogw9z3I8D2W9r534WiK1WG2tz+S39gmSZIkSZKGUWVrm3pFswlHHQXnnVd3JZIkSZIkqQ0ZJOltjQYsXgxTptRdiSRJkiRJakMGSSq88QasW2d/JEmSJEmSdEAGSSqsXw+7dxskSZIkSZKkAzJIUqHRKI4GSZIkSZIk6QAMklRoNuHUU+GUU+quRJIkSZIktSmDJBUaDbj44rqrkCRJkiRJbcwgSbBtGzzzjNvaJEmSJEnSQRkkCVavLo4GSZIkSZIk6SAMklRsa5s6FRYtqrsSSZIkSZLUxgySVDTavuACOPLIuiuRJEmSJEltzCCp1735JvT3u61NkiRJkiSNyCCp1z36KPz6135jmyRJkiRJGpFBUq9rNoujK5IkSZIkSdIIDJJ6XaMBp5wCc+bUXYkkSZIkSWpzBkm9rtksViNF1F2JJEmSJElqcwZJvez55+FnP3NbmyRJkiRJqsQgqZetXl0cbbQtSZIkSZIqMEjqZc0mTJ4MF15YdyWSJEmSJKkDGCT1skYDFi6Eo46quxJJkiRJktQBDJJ61VtvwZo19keSJEmSJEmVGST1qsceg9deM0iSJEmSJEmVGST1qmazONpoW5IkSZIkVWSQ1KuaTejrg3nz6q5EkiRJkiR1iEpBUkQsj4gnImJzRNw0zPvHRsR3I+KRiNgYEVeX43Mi4kcRsakcv37Iff+m/NyNEfFX4/NIqqTRKFYjRdRdiSRJkiRJ6hCTR7ogIiYBtwKXAQPA2oi4NzMfb7nsOuDxzPxERPQBT0TE3cAg8PnMXBcRM4CHIuKBzHw8In4DuAI4LzPfiIiTxvvhdAA7d8ITT8BVV9VdiSRJkiRJ6iBVViQtBTZn5pbM3A3cQxEAtUpgRkQEMB3YCQxm5nOZuQ4gM18BNgGzynuuBb6cmW+U7z8/5qdRNWvWFEcbbUuSJEmSpFGoEiTNAra2nA/wdhi010rgbGAbsAG4PjP3tF4QEXOBRcDqcugM4J9FxOqI+F8RsWT05euQNBpwxBGwxD9ySZIkSZJUXZUgabgmOjnk/HJgPXAKsBBYGRHH7PuAiOnAt4EbMvPlcngycBxwEfDHwLfKFU3v/OURKyKiPyL6t2/fXqFcjajZhPe/H6ZPr7sSSZIkSZLUQaoESQPAnJbz2RQrj1pdDazKwmbgKeAsgIiYQhEi3Z2Zq4Z87t571gB7gBOH/vLMvD0zF2fm4r6+vqrPpQPZswdWr3ZbmyRJkiRJGrUqQdJaYH5EzIuIqcCVwL1DrnkGuBQgImYCZwJbyhVGdwCbMvNrQ+75H8Al5T1nAFOBFw7xOVTVT34CL71UfGObJEmSJEnSKIz4rW2ZORgRnwXuByYBd2bmxoi4pnz/NuBLwF0RsYFiK9yNmflCRHwI+AywISLWlx/5J5l5H3AncGdEPAbsBq7KzKFb5jTems3i6IokSZIkSZI0SiMGSQBl8HPfkLHbWl5vAz46zH0PMnyPJcpvgPv90RSrcdBowHHHwRln1F2JJEmSJEnqMFW2tqmbNJvFaqT9+5pLkiRJkiQdlEFSL3npJdi40W1tkiRJkiTpkBgk9ZK1ayHTRtuSJEmSJOmQGCT1kmaz2NK2dGndlUiSJEmSpA5kkNRLGg045xw49ti6K5EkSZIkSR3IIKlXZL7daFuSJEmSJOkQGCT1iiefhJ07DZIkSZIkSdIhM0jqFc1mcbTRtiRJkiRJOkQGSb2i2YRjjoGzz667EkmSJEmS1KEMknpFo1F8W9sR/iOXJEmSJEmHxlShF7z2Gjz6qNvaJEmSJEnSmBgk9YL+ftizx0bbkiRJkiRpTAySekGjURyXLau3DkmSJEmS1NEMknpBswlnnAEnnFB3JZIkSZIkqYMZJHW7zGJFktvaJEmSJEnSGBkkdbuf/xyef95G25IkSZIkacwMkrpds1kcXZEkSZIkSZLGyCCp2zUacPTRsGBB3ZVIkiRJkqQOZ5DU7ZpNWLIEJk+uuxJJkiRJktThDJK62a9/DQ8/7LY2SZIkSZI0LgySutm6dTA4aKNtSZIkSZI0LgySupmNtiVJkiRJ0jgySOpmjQa8731w0kl1VyJJkiRJkrqAQVI3azZdjSRJkiRJksZNpSApIpZHxBMRsTkibhrm/WMj4rsR8UhEbIyIq8vxORHxo4jYVI5fP8y9X4iIjIgTx/442mfrVnj2WYMkSZIkSZI0bkYMkiJiEnAr8DHgHODTEXHOkMuuAx7PzPOBjwBfjYipwCDw+cw8G7gIuK713oiYA1wGPDMOz6JWe/sj2WhbkiRJkiSNkyorkpYCmzNzS2buBu4BrhhyTQIzIiKA6cBOYDAzn8vMdQCZ+QqwCZjVct9/Bv5teb/GU7MJ06bBeefVXYkkSZIkSeoSVYKkWcDWlvMB3hkGAawEzga2ARuA6zNzT+sFETEXWASsLs8/CTybmY8cUuU6uEYDLrwQpk6tuxJJkiRJktQlqgRJMczY0BVElwPrgVOAhcDKiDhm3wdETAe+DdyQmS9HxLuAPwX+fMRfHrEiIvojon/79u0VyhVvvAHr1rmtTZIkSZIkjasqQdIAMKflfDbFyqNWVwOrsrAZeAo4CyAiplCESHdn5qry+tOAecAjEfHz8jPXRcR7hv7yzLw9Mxdn5uK+vr7qT9bL1q8vwiQbbUuSJEmSpHFUJUhaC8yPiHllA+0rgXuHXPMMcClARMwEzgS2lD2T7gA2ZebX9l6cmRsy86TMnJuZcynCqgsy8xdjfiLZaFuSJEmSJB0WIwZJmTkIfBa4n6JZ9rcyc2NEXBMR15SXfQn4QERsAH4I3JiZLwAfBD4DXBIR68uf3zosT6K3NZswZw6cckrdlUiSJEmSpC4yucpFmXkfcN+QsdtaXm8DPjrMfQ8yfI+lodfNrVKHKmo03NYmSZIkSZLGXZWtbeokzz0HTz/ttjZJkiRJkjTuDJK6zerVxdEVSZIkSZIkaZwZJHWbRgOmToULLqi7EkmSJEmS1GUMkrpNswmLFsGRR9ZdiSRJkiRJ6jIGSd3kzTdh7Vq3tUmSJEmSpMPCIKmbbNgAv/61jbYlSZIkSdJhYZDUTZrN4uiKJEmSJEmSdBgYJHWTRgPe8x449dS6K5EkSZIkSV3IIKmbNJvFtraIuiuRJEmSJEldyCCpW2zfDps3u61NkiRJkiQdNgZJ3WL16uJoo21JkiRJknSYGCR1i2YTJk2CCy+suxJJkiRJktSlDJK6RaMB558P73pX3ZVIkiRJkqQuZZDUDd56C9ascVubJEmSJEk6rAySusHGjfDqqzbaliRJkiRJh5VBUjdoNoujK5IkSZIkSdJhZJDUDZpNOPFEeN/76q5EkiRJkiR1MYOkbtBoFNvaIuquRJIkSZIkdTGDpE734ovwk5+4rU2SJEmSJB12Bkmdbs2a4mijbUmSJEmSdJgZJHW6RgOOOAKWLKm7EkmSJEmS1OUMkjpdswkLFsCMGXVXIkmSJEmSupxBUifbs6cIktzWJkmSJEmSJoBBUid74gl46SUbbUuSJEmSpAlRKUiKiOUR8UREbI6Im4Z5/9iI+G5EPBIRGyPi6nJ8TkT8KCI2lePXt9zzlYj4SUQ8GhHfiYh3j9tT9Ypmszi6IkmSJEmSJE2AEYOkiJgE3Ap8DDgH+HREnDPksuuAxzPzfOAjwFcjYiowCHw+M88GLgKua7n3AWBBZp4H/BT44jg8T29pNODd74Yzzqi7EkmSJEmS1AOqrEhaCmzOzC2ZuRu4B7hiyDUJzIiIAKYDO4HBzHwuM9cBZOYrwCZgVnn+g8wcLO9vArPH/DS9Zm9/pCPcoShJkiRJkg6/KgnELGBry/lAOdZqJXA2sA3YAFyfmXtaL4iIucAiYPUwv+NfAd+vVrIAePlleOwxt7VJkiRJkqQJUyVIimHGcsj55cB64BRgIbAyIo7Z9wER04FvAzdk5svv+PCIP6XYAnf3sL88YkVE9EdE//bt2yuU2yPWroVMG21LkiRJkqQJUyVIGgDmtJzPplh51OpqYFUWNgNPAWcBRMQUihDp7sxc1XpTRFwFfBz4PzNzaDgFQGbenpmLM3NxX19flWfqDXsbbS9dWm8dkiRJkiSpZ1QJktYC8yNiXtlA+0rg3iHXPANcChARM4EzgS1lz6Q7gE2Z+bXWGyJiOXAj8MnMfH1sj9GDGg04++yi2bYkSZIkSdIEGDFIKhtifxa4n6JZ9rcyc2NEXBMR15SXfQn4QERsAH4I3JiZLwAfBD4DXBIR68uf3yrvWQnMAB4ox28b30frYpnFiiS3tUmSJEmSpAk0ucpFmXkfcN+QsdtaXm8DPjrMfQ8yfI8lMvP0UVWqt23eDDt22GhbkiRJkiRNKL83vhPt7Y/kiiRJkiRJkjSBDJI6UbMJM2YUPZIkSZIkSZImiEFSJ2o0im9rmzSp7kokSZIkSVIPMUjqNK+9Bo8+6rY2SZIkSZI04QySOk1/P7z1lo22JUmSJEnShDNI6jR7G20vW1ZvHZIkSZIkqecYJHWaZhPmz4cTT6y7EkmSJEmS1GMMkjpJZtFo221tkiRJkiSpBgZJneTpp+GXv7TRtiRJkiRJqoVBUifZ2x/JFUmSJEmSJKkGBkmdpNGAd70L3v/+uiuRJEmSJEk9yCCpkzSbsGQJTJ5cdyWSJEmSJKkHGSR1il274OGH3dYmSZIkSZJqY5DUKdatgzfftNG2JEmSJEmqjUFSp9jbaHvZsnrrkCRJkiRJPcsgqVM0GjB3LrznPXVXIkmSJEmSepRBUqdoNt3WJkmSJEmSamWQ1AkGBoofG21LkiRJkqQaGSR1gr39kVyRJEmSJEmSamSQ1AmaTTjySDj//LorkSRJkiRJPcwgqRM0GnDhhTB1at2VSJIkSZKkHmaQ1O5274aHHnJbmyRJkiRJqp1BUrtbvx7eeMNG25IkSZIkqXYGSe1ub6NtgyRJkiRJklSzSkFSRCyPiCciYnNE3DTM+8dGxHcj4pGI2BgRV5fjcyLiRxGxqRy/vuWe4yPigYh4sjweN36P1UWaTZg9u/iRJEmSJEmq0YhBUkRMAm4FPgacA3w6Is4Zctl1wOOZeT7wEeCrETEVGAQ+n5lnAxcB17XcexPww8ycD/ywPNdQjYarkSRJkiRJUluosiJpKbA5M7dk5m7gHuCKIdckMCMiApgO7AQGM/O5zFwHkJmvAJuAWeU9VwDfKF9/A/idsTxIV/rFL+DnP7fRtiRJkiRJagtVgqRZwNaW8wHeDoP2WgmcDWwDNgDXZ+ae1gsiYi6wCFhdDs3MzOcAyuNJw/3yiFgREf0R0b99+/YK5XaR1eUflSuSJEmSJElSG6gSJMUwYznk/HJgPXAKsBBYGRHH7PuAiOnAt4EbMvPl0RSYmbdn5uLMXNzX1zeaWztfowFTpsAFF9RdiSRJkiRJUqUgaQCY03I+m2LlUaurgVVZ2Aw8BZwFEBFTKEKkuzNzVcs9v4yIk8trTgaeP7RH6GLNJixaBNOm1V2JJEmSJElSpSBpLTA/IuaVDbSvBO4dcs0zwKUAETETOBPYUvZMugPYlJlfG3LPvcBV5eurgH86tEfoUoODsHat29okSZIkSVLbGDFIysxB4LPA/RTNsr+VmRsj4pqIuKa87EvAByJiA8U3sN2YmS8AHwQ+A1wSEevLn98q7/kycFlEPAlcVp5rrw0b4PXXbbQtSZIkSZLaxuQqF2XmfcB9Q8Zua3m9DfjoMPc9yPA9lsjMHZSrmDSMZrM4uiJJkiRJkiS1iSpb21SHRgNmzoT3vrfuSiRJkiRJkgCDpPbVbBbb2mLYBV2SJEmSJEkTziCpHb3wAjz5pNvaJEmSJElSWzFIakerVxdHgyRJkiRJktRGDJLaUbMJkybB4sV1VyJJkiRJkrSPQVI7ajTgvPPg6KPrrkSSJEmSJGkfg6R289ZbsGZN0WhbkiRJkiSpjRgktZvHH4dXXrE/kiRJkiRJajsGSe2m2SyOBkmSJEmSJKnNGCS1m2YTTjgBTj+97kokSZIkSZLewSCp3TQaxWqkiLorkSRJkiRJegeDpHbyq1/Bpk022pYkSZIkSW3JIKmdrF5dHO2PJEmSJEmS2pBBUjtpNostbUuW1F2JJEmSJEnSfgyS2kmzCQsWwDHH1F2JJEmSJEnSfgyS2sWePUWQ5LY2SZIkSZLUpgyS2sVPf1o027bRtiRJkiRJalMGSe2i2SyOrkiSJEmSJEltyiCpXTQa8O53w5ln1l2JJEmSJEnSsAyS2kWzCcuWwRH+I5EkSZIkSe3J1KIdvPIKPPaY29okSZIkSVJbM0hqB2vXFt/aZpAkSZIkSZLamEFSO9jbaHvZsnrrkCRJkiRJOohKQVJELI+IJyJic0TcNMz7x0bEdyPikYjYGBFXt7x3Z0Q8HxGPDblnYUQ0I2J9RPRHxNKxP06HajTgrLPguOPqrkSSJEmSJOmARgySImIScCvwMeAc4NMRcc6Qy64DHs/M84GPAF+NiKnle3cBy4f56L8C/iIzFwJ/Xp73nsxiRdLFF9ddiSRJkiRJ0kFVWZG0FNicmVsyczdwD3DFkGsSmBERAUwHdgKDAJn54/J8qASOKV8fC2wbffld4Gc/gxdesD+SJEmSJElqe5MrXDML2NpyPgAMbeazEriXIgyaAXwqM/eM8Lk3APdHxF9TBFofqFJw19nbH8kgSZIkSZIktbkqK5JimLEccn45sB44BVgIrIyIYzi4a4HPZeYc4HPAHcP+8ogVZQ+l/u3bt1cot8M0mzB9Opx7bt2VSJIkSZIkHVSVIGkAmNNyPpv9t6FdDazKwmbgKeCsET73KmBV+fq/U2yh209m3p6ZizNzcV9fX4VyO0yjAUuXwqRJdVciSZIkSZJ0UFWCpLXA/IiYVzbQvpJiG1urZ4BLASJiJnAmsGWEz90GfLh8fQnwZNWiu8brr8Mjj9hoW5IkSZIkdYQReyRl5mBEfBa4H5gE3JmZGyPimvL924AvAXdFxAaKrXA3ZuYLABHxDxTf5HZiRAwA/y4z7wD+NfA3ETEZ2AWsGPena3f9/fDWW/ZHkiRJkiRJHaFKs20y8z7gviFjt7W83gZ89AD3fvoA4w8CF1autBvtbbS9bGjvckmSJEmSpPZTZWubDpdmE04/Hbqx95MkSZIkSeo6Bkl1ySwabbutTZIkSZIkdQiDpLo88wz84hcGSZIkSZIkqWMYJNVlb38kv7FNkiRJkiR1CIOkujQacNRR8P73112JJEmSJElSJQZJdWk2YckSmDKl7kokSZIkSZIqMUiqw65dsG6d/ZEkSZIkSVJHMUiqw8MPw5tvGiRJkiRJkqSOYpBUh72Ntg2SJEmSJElSBzFIqkOjAe99L5x8ct2VSJIkSZIkVWaQVIdmEy6+uO4qJEmSJEmSRsUgaaI9+yxs3eq2NkmSJEmS1HEm111Az3nPe2DDBjjxxLorkSRJkiRJGhWDpIk2aRIsWFB3FZIkSZIkSaPm1jZJkiRJkiRVYpAkSZIkSZKkSgySJEmSJEmSVIlBkiRJkiRJkioxSJIkSZIkSVIlBkmSJEmSJEmqxCBJkiRJkiRJlRgkSZIkSZIkqRKDJEmSJEmSJFVikCRJkiRJkqRKIjPrrqGyiNgOPF13HQdxIvBC3UVIXcZ5JY0v55Q0/pxX0vhyTknjb6R59d7M7KvyQR0VJLW7iOjPzMV11yF1E+eVNL6cU9L4c15J48s5JY2/8ZxXbm2TJEmSJElSJQZJkiRJkiRJqsQgaXzdXncBUhdyXknjyzkljT/nlTS+nFPS+Bu3eWWPJEmSJEmSJFXiiiRJkiRJkiRVYpA0DiJieUQ8ERGbI+KmuuuROkVEzImIH0XEpojYGBHXl+PHR8QDEfFkeTyu5Z4vlnPtiYi4vL7qpfYVEZMi4uGI+F557pySxiAi3h0R/xgRPyn/nXWx80o6dBHxufLvfo9FxD9ExDTnlDQ6EXFnRDwfEY+1jI16HkXEhRGxoXzvv0REjPS7DZLGKCImAbcCHwPOAT4dEefUW5XUMQaBz2fm2cBFwHXl/LkJ+GFmzgd+WJ5TvnclcC6wHPjbcg5KeqfrgU0t584paWz+Bvh/M/Ms4HyK+eW8kg5BRMwC/i9gcWYuACZRzBnnlDQ6d1HMiVaHMo/+K7ACmF/+DP3M/Rgkjd1SYHNmbsnM3cA9wBU11yR1hMx8LjPXla9fofiL+SyKOfSN8rJvAL9Tvr4CuCcz38jMp4DNFHNQUikiZgO/DXy9Zdg5JR2iiDgG+OfAHQCZuTszf4XzShqLycBRETEZeBewDeeUNCqZ+WNg55DhUc2jiDgZOCYzG1k00P5/Wu45IIOksZsFbG05HyjHJI1CRMwFFgGrgZmZ+RwUYRNwUnmZ800a2c3AvwX2tIw5p6RD9z5gO/B/l1tGvx4RR+O8kg5JZj4L/DXwDPAc8FJm/gDnlDQeRjuPZpWvh44flEHS2A23f9CvwpNGISKmA98GbsjMlw926TBjzjepFBEfB57PzIeq3jLMmHNKeqfJwAXAf83MRcBrlFsFDsB5JR1E2bPlCmAecApwdET8/sFuGWbMOSWNzoHm0SHNL4OksRsA5rScz6ZYmimpgoiYQhEi3Z2Zq8rhX5bLLCmPz5fjzjfp4D4IfDIifk6x1fqSiPhvOKeksRgABjJzdXn+jxTBkvNKOjS/CTyVmdsz801gFfABnFPSeBjtPBooXw8dPyiDpLFbC8yPiHkRMZWigdW9NdckdYTyGwHuADZl5tda3roXuKp8fRXwTy3jV0bEkRExj6IZ3JqJqldqd5n5xcycnZlzKf599P9l5u/jnJIOWWb+AtgaEWeWQ5cCj+O8kg7VM8BFEfGu8u+Cl1L0yXROSWM3qnlUbn97JSIuKufjv2y554Amj3/dvSUzByPis8D9FN84cGdmbqy5LKlTfBD4DLAhItaXY38CfBn4VkT8AcVfNv4FQGZujIhvUfwFfhC4LjPfmvCqpc7jnJLG5t8Ad5f/0XALcDXFf5B1XkmjlJmrI+IfgXUUc+Rh4HZgOs4pqbKI+AfgI8CJETEA/DsO7e9811J8A9xRwPfLn4P/7qIxtyRJkiRJknRwbm2TJEmSJElSJQZJkiRJkiRJqsQgSZIkSZIkSZUYJEmSJEmSJKkSgyRJkiRJkiRVYpAkSZIkSZKkSgySJEmSJEmSVIlBkiRJkiRJkir5/wG90P6ZgSObJgAAAABJRU5ErkJggg==) 

#### 进化学习曲线

泛化误差：衡量模型在未知数据上的准确率的指标，叫做泛化误差。
一个集成模型（f）在未知数据集上的泛化误差E(f)由方差、偏差和噪声共同决定。

泛化误差 = 方差+偏差+噪声

以前学习曲线直接获取曲线上分数的最高点 ， 即考虑偏差最小的点。但是现在数据量小，模型不稳定，因此我们应该适量的加入方差也纳入考虑范围。

```python
axisx = range(50,1050,50)
rs = []
var = []
ge =[]
for i in axisx:
    reg = XGBR(n_estimators=i ,random_state=420)
    cvresult = CVS(reg,Xtrain,Ytrain,cv=cv)
    #记录1 - 偏差
    rs.append(cvresult.mean())
    #记录方差
    var.append(cvresult.var())
    #使计算泛化误差
    ge.append((1-cvresult.mean())**2+cvresult.var())

#打印R2最高所对应的参数取值，并打印这个参数下的方差
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
#打印方差最低时对应的方差取值，并打印这个参数下的R2
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
      
#方差以及泛化误差的可控部分
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))

plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()

#添加方差线
plt.plot(axisx,rs+var,c="red",linestyle='-')
plt.plot(axisx,rs-var,c="red",linestyle='-')
plt.legend()
plt.show()
```

 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABJQAAAEvCAYAAAATn5tUAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAmM0lEQVR4nO3df9BddZ0n+PeH/CDhlyBEBIISlV8BIZAQsceasv3RpqdVxq62xO7p7WK2ltXGXbXHXnGnt7Ysa6p3y+ktesEZSx3Fdq1Gp8VatHWUctqymN57QgKBEAJNBIUISgBRfsiPkO/+cW/gSfJAnid5kvs8975eVbfOvd9zzr2fU/FbJG/P53uqtRYAAAAAmKpDhl0AAAAAAHOLQAkAAACAaREoAQAAADAtAiUAAAAApkWgBAAAAMC0CJQAAAAAmJb5wy5gJhx33HHtlFNOGXYZAAAAACNj/fr1D7XWlky2byQCpVNOOSXr1q0bdhkAAAAAI6Oqfvpi+7S8AQAAADAtAiUAAAAApmVKgVJVramqO6tqS1VdPsn+l1XVt6rqlqraVFWXDMYXVdXaCeOfnHDOp6vqjqq6taq+WVVHD8aPrap/qKrHq+qqGbpOAAAAAGbIXtdQqqp5ST6T5O1Jtia5saqua63dPuGwy5Lc3lp7V1UtSXJnVX01ydNJ3tJae7yqFiS5oaq+21rrJbk+ySdaa9ur6v9M8okkH0/yVJL/LcnZg9c+efbZZ7N169Y89dRT+/oVs96iRYuydOnSLFiwYNilAAAAAGNkKotyr06ypbV2d5JU1TVJLkoyMVBqSY6sqkpyRJJHkmxvrbUkjw+OWTB4tSRprX1/wvm9JH8wGH8i/eDpdft6UUmydevWHHnkkTnllFPSL2u0tNby8MMPZ+vWrVm2bNmwywEAAADGyFRa3k5Kct+Ez1sHYxNdleTMJPcn2Zjkw621HUn/Dqeq2pDkwSTXt9a6SX7jXyf57vRKf2lPPfVUjj322JEMk5KkqnLssceO9B1YAAAAwOw0lUBpskSm7fb5HUk2JDkxyYokV1XVUUnSWnuutbYiydIkq6tqlza2qvq3SbYn+ep0Cq+qS6tqXVWt27Zt24sdM52vnHNG/foAAACA2WkqgdLWJCdP+Lw0/TuRJrokybWtb0uSe5KcMfGA1tqjSX6YZM3Osar6kyTvTPJHg/a4KWutfa61tqq1tmrJkiXTOfWgue+++7Js2bI88sgjSZJf/vKXWbZsWX7605/mrrvuyjvf+c689rWvzcqVK/Pbv/3b+dGPfpQkufrqq7NkyZKsWLEiZ511Vv7gD/4gTz755DAvBQAAAOB5UwmUbkxyalUtq6qFSS5Oct1ux9yb5K1JUlXHJzk9yd1VtWTC09sWJ3lbkjsGn9ekvwj3u1trI5mWnHzyyfngBz+Yyy/vPxjv8ssvz6WXXprjjz8+v/d7v5dLL700P/7xj7N+/fpceeWVufvuu58/933ve182bNiQTZs2ZeHChfna1742rMsAAAAA2MVeF+UePIXtQ0m+l2Reki+21jZV1QcG+z+b5FNJrq6qjem3yH28tfZQVZ2T5MuDJ8UdkuTrrbVvD776qiSHJrl+0LrVa619IEmq6idJjkqysKr+ZZLf2e2pcnPGRz/60axcuTJXXHFFbrjhhlx55ZX5yle+kje+8Y1597vf/fxxZ599ds4+e8+H2m3fvj1PPPFEjjnmmINZ9sx59tnkm99MTjwxWb48efnLh10RAAAAsJ+m8pS3tNa+k+Q7u419dsL7+5P8ziTn3ZrkvBf5zhd9iltr7ZSp1DUXLFiwIJ/+9KezZs2afP/738/ChQuzadOmnH/++S953te+9rXccMMNeeCBB3LaaaflXe9610GqeIZ94xvJ+9//wufjj+8HSxNfZ52VzNK2RQAAAGBPUwqU5ryPfCTZsGFmv3PFiuSKK6Z06He/+92ccMIJue222/L2t799j/3vec97ctddd+W0007Ltddem6Tf8nbVVVeltZbLLrssn/70p59vnZtT/vEfk8MOS/7zf042b042bUpuvz35m79JHnvsheOOO27PoGn58uSVr0wsPg4AAACzyngESkO0YcOGXH/99en1ennTm96Uiy++OGedddbzC3AnyTe/+c2sW7cuH/vYx/Y4v6ryrne9K1deeeXcDJS6Llm1KvkX/6L/2qm15Gc/64dLE1/XXJM8+ugLxx1zzORB00knCZoAAABgSMYjUJrinUQzrbWWD37wg7niiivyqle9Kn/+53+ej33sY/nCF76Qv/zLv8x11133/DpKL/UUtxtuuCGvfe1rD1bZM+epp5Kbb04++tE991UlS5f2X78zoVuyteTnP98zaLr22uTzn3/huCOP3LVlbuf7k09ODpnKWvMA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) 

 ![img](data:image/png;base64,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f//b64pEREREREREUmnKm6Q5dgxefRXq1IH77nOrtn3+uRuddMcdeTJMio+P56233uLiiy/mrrvuIjAwkI8++oh169bRrVu33BcmxcVB375uWlv9+hAVpTBJREREREREch0FSgJHjrhl6C+8EPr1g6pV4bvv3FSrm292o2TymGPHjjFhwgRq165Nz549KVu2LF999RXR0dF06tSJgIAAr0s83W+/wZVXwmuvwaOPusbnNWt6XZWIiIiIiIjIafLekBPJPocOwYQJ8PLLcOAAtGsH06ZBq1ZgjNfVZcnhw4eZNGkSL774Irt37+aqq67i7bffpn379pjc/J4+/BB693Y9kr79Fq67zuuKRERERERERM5IgVJBtHcvjBsHEyfC4cNwww0wfDhccYXXlWXZwYMHmTBhAuPHj+fAgQN06NCB4cOHc/XVV3td2tnFxrqG22+/DS1bwkcfuRFiIiIiIiIiIrmYAqWC5K+/3NS21193vXpuuw2GDYNGjbyuLMv27t3LuHHjmDhxIocPH+bGG29k+PDhNGvWzOvSzm3dOtcrae1aGDoUnnkmT/apEhERERERkYJH314Lgq1b4bnn4J13ICkJ7rwThgyBevW8rizL/vrrL8aOHcvrr79OXFwct99+O8OGDeOyyy7zujTfTJkCDzwAxYrBjBkQEeF1RSIiIiIiIiI+U6CUn23YAKNHwwcfQEAA3HsvPPYY1KrldWVZ9scff/Dcc88xefJkkpKSuPPOOxkyZAj18ko4duSIC5Leew9at3Z/Nhdc4HVVIiIiIiIiIpmiQCk/io6GUaPgs88gJMT16HnkEahSxevKsmzjxo2MHj2a999/n4CAALp3787gwYOplZfCsdWr3RS3jRvhqafgiSdc0CciIiIiIiKSxyhQyk8WL4aRI90qYSVLur48AwZAaKjXlWXZqlWrGDVqFJ988gkhISH079+fRx99lCp5KRyzFt56ywV7pUvD7NnQpo3XVYmIiIiIiIhkmQKlvM5a+OknFyTNmQNly8Kzz0K/fi68yKOWLl3KyJEj+eqrryhRogSDBw/m4YcfpkKFCl6Xljn//AO9e7vV29q3h6lToWJFr6sSEREREREROS8KlPIqa+H7712QtGgRVKrkVnDr3RuKF/e6uixbsGABI0aMYNasWZQpU4ann36a/v37U6ZMGa9Ly7yoKDfFbcsWGDHCjRgrVMjrqkRERERERETOmwKlvCY5Gb780gVJUVFQvTq8+qpruB0S4nV1WWKtZfbs2YwYMYL58+dToUIFnnvuOfr06UOJEiW8Li/zrIX//hcGDnTTDX/6Ca66yuuqRERERERERLKNAqW8IjHRTZsaNQrWr4c6dWDyZLjzTggK8rq6LLHW8s033zBixAiWLl1K1apVeeWVV7jvvvsoWrSo1+VlzaFD0KMHfP45XHONW82tfHmvqxIRERERERHJVpp/k9sdPw5vvgkXXwx33eVWBZs2zYVK3bvnyTApKSmJTz75hLCwMDp27Mi+fft444032Lx5M/3798+bYVJcnBuVdOml8NVX8Pzzrjm6wiQRERERERHJhzRCKbeKjXUrgz3/PPz1F1x+Obz0EtxwQ57tw5OQkMCHH37I6NGj2bhxI/Xq1WPq1Kl07tyZwMA8+lfx6FF4/XXXv2rXLmjeHD79FK680uvKRERERERERHJMHv0Wn4/98w+89hq8+CLs3QtXX+2mtrVvD8Z4XV2WHD9+nClTpjBmzBj++OMPGjVqxKeffspNN91EQECA1+VlzT//uN5VL70E+/bB//0fvP++u86jf04iIiIiIiIivlKglJuMGQPPPef68EREwPDhebqZc2xsLG+88QYvvPACO3fu5IorrmDChAlcd911mLwauhw4AOPHwyuvuD+na65xf04tW3pdmYiIiIiIiIjfKFDKTXbuhNatYdgwN8Utj/rnn3/473//y0svvcTevXtp3bo17733Hm3atMm7QdLu3W400n//C0eOwE03uSCpSROvKxMRERERERHxOwVKucnLL+fZ/kgABw4cYPz48bzyyiscOnSIa665huHDh9MyL4/e2bEDXngB3ngD4uOhUycX+DVs6HVlIiIiIiIiIp5RoJSb5NEwaffu3bz00kv897//5ciRI9x0000MHz6cJnl59M6WLW764TvvgLVuhb0hQ6BuXa8rExEREREREfGcAiXJsu3bt/PCCy/w5ptvEh8fT6dOnRg2bBgN8/LonY0bYdQo+OADCAiAHj3gscegZk2vKxMRERERERHJNRQoSaZt2bKFMWPGMGXKFKy13H333QwZMoQ6dep4XVrWrVrlgqRPPoGQEOjfHx59FKpU8boyERERERERkVxHgZL4bP369YwePZoPP/yQwMBAevbsyWOPPUaN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oWZNeP116NYN8up7EhERERERERFAgVKu0qVLF26++Wa6du1KUFCQ1+VkTWIifPQRjBoF69dD3bowZQp07Qp59T2JiIiIiIiIyEkUKOUinTt39rqErIuPh/feg9GjYcsWaNjQBUu33goBAV5XJyIiIiIiIiLZqJDXBUged+wYTJwIF10EPXtC2bLwv/+5aW6dOilMEhEREREREcmHNEJJsubIEdcTaexY+PtvaNkS3nwTOnQAY7yuTkRERERERERykAIlyZyYGDciadw42L8f2raFadOgVSsFSSIiIiIiIiIFhAIl8c2+fTB+PEyY4EKl666D4cOheXOvKxMRERERERERP1OgJGf399/w4ovw2mtw9CjccosLksLDva5MRERERERERDyiQEkytn07PP+864uUkABdusDQodCggdeViYiIiIiIiIjHFCjJyX7/HcaMgXffBWuhWzcYMsSt4iYiIiIiIiIiggIlOWH9ehg9Gj78EAIDoVcvGDQIatTwujIRERERERERyWUUKBV0K1fCyJHw2WdQpAg89BA88ghccIHXlYmIiIiIiIhILqVAqaD69VcYMQK++QZKlHD9kQYMgNBQrysTERERERERkVxOgVJBM3++C5J++AHKloVnnoF+/aBMGa8rExEREREREZE8QoFSQWCtC5BGjIAFC6BCBbeC2/33u9FJIiIiIiIiIiKZoEApP7PWTWkbMQKWLoUqVeCVV6BHD9cvSUREREREREQkCwp5XYDkgKQk+PRTCAuDjh1h3z544w34/Xfo319hkoiIiIiIiIicFwVK+UliIkydCg0bwu23w/Hj8O67sHEj9OwJhQt7XaGIiIiIiIiI5AOa8pYfHD8O770HY8bAli1w6aXw8cdwyy0QEOB1dSIiIiIiIiKSz2iEUl527BhMmAAXXQS9erlV2776CqKj3QglhUkiIiIiIiIikgM0QikvOnIEJk2CsWNh927417/g7behfXswxuvqRERERERERCSfU6CUlxw6BBMnwrhxcOCAC5AefxyuvtrrykRERERERESkAFGglBfs2wcvv+ymt/3zD9xwAwwfDldc4XVlIiIiIiIiIlIAKVDKzXbtghdfhNdec/2SbrnFBUlhYV5XJiIiIiIiIiIFmAKl3GjbNnj+eXjrLUhIgK5dYehQuOQSrysTEREREREREVGglKts3gxjxsC777rm2t26weDBbhU3EREREREREZFcQoFSbtKzJ/zyC9x/PwwaBNWre12RiIiIiIiIiMhpFCjlJq++CmXKQOXKXlciIiIiIiIiInJGCpRyE/VIEhEREREREZE8oJDXBYiIiIiIiIiISN6iQElERERERERERDJFgZKIiIiIiIiIiGSKAiUREREREREREckUnwIlY0ykMWajMWazMWZIBvtLGWO+McasNMasNcZ0T9keYoz5Nd32p9M95raUbcnGmKanPN9lxphfUvavNsaEnO8bFRERERERERGR7HHOQMkYEwC8ClwDXAJ0McacuhzZA8A6a20joDXwojEmGDgOtEnZHgZEGmOuTHnMGuBmYP4prxcIvA/cb61tkPJ8CVl5cyIiIiIiIiIikv0CfTimGbDZWrsFwBjzEdARWJfuGAuUMMYYoDhwAEi01lrgSMoxQSkXC2CtXZ/yfKe+XgdglbV2Zcpx+zP/tkREREREREREJKf4MuWtCrA93f0dKdvSmwjUB3YCq4GHrLXJ4EY4GWOigT3AD9baJed4vbqANcbMNMasMMY85kONIiIiIiIiIiLiJ74ESqcNISJllFE6EUA0cAFuattEY0xJAGttkrU2DKgKNDPGNDzH6wUC/wLuSLm+yRjT9rSijOlljFlmjFm2d+9eH96GiIiIiIiIiIhkB18CpR1AtXT3q+JGIqXXHfjCOpuBrUC99AdYaw8BPwGRPrzePGvtPmttLPAd0PjUg6y1b1hrm1prm4aGhvrwNkREREREREREJDv40kNpKVDHGFML+AvoDHQ95ZhtQFtggTGmInAxsMUYEwokWGsPGWOKAO2A587xejOBx4wxRYF4oBUw7mwPWL58+T5jzJ8+vBcRfygP7PO6CJECTOegiPd0Hop4S+egiPfyy3lY40w7zhkoWWsTjTH9cEFPADDZWrvWGHN/yv5JwLPAFGPMatwUucHW2n3GmMuAd1NWiisEfGKt/RbAGHMTMAEIBaYbY6KttRHW2oPGmJdwQZYFvrPWTj9HjRqiJLmGMWaZtbap13WIFFQ6B0W8p/NQxFs6B0W8VxDOQ+MWYhOR7FIQ/uEQyc10Dop4T+ehiLd0Dop4ryCch770UBIREREREREREUmlQEkk+73hdQEiBZzOQRHv6TwU8ZbOQRHv5fvzUFPeREREREREREQkUzRCSUREREREREREMkWBksh5MMaUNsZ8ZozZYIxZb4xpbowpa4z5wRizKeW6jNd1iuRXxpiHjTFrjTFrjDHTjDEhOgdFcpYxZrIxZo8xZk26bWc874wxQ40xm40xG40xEd5ULZK/nOE8fCHlZ9JVxpgvjTGl0+3TeSiSjTI6B9Pte9QYY40x5dNty5fnoAIlkfMzHphhra0HNALWA0OAOdbaOsCclPsiks2MMVWAB4Gm1tqGQADQGZ2DIjltChB5yrYMzztjzCW487JBymP+a4wJ8F+pIvnWFE4/D38AGlprLwN+A4aCzkORHDKF089BjDHVgPbAtnTb8u05qEBJJIuMMSWBq4G3Aay18dbaQ0BH4N2Uw94F/u1FfSIFRCBQxBgTCBQFdqJzUCRHWWvnAwdO2Xym864j8JG19ri1diuwGWjmjzpF8rOMzkNr7SxrbWLK3cVA1ZTbOg9FstkZ/i8EGAc8BqRvVp1vz0EFSiJZdyGwF3jHGBNljHnLGFMMqGit3QWQcl3ByyJF8itr7V/AWNxvgHYBMdbaWegcFPHCmc67KsD2dMftSNkmIjnrXuD7lNs6D0X8wBhzI/CXtXblKbvy7TmoQEkk6wKBxsBr1tpw4CiaWiPiNyk9WjoCtYALgGLGmDu9rUpETmEy2KYlhkVykDFmOJAIfHBiUwaH6TwUyUbGmKLAcODJjHZnsC1fnIMKlESybgeww1q7JOX+Z7iAabcxpjJAyvUej+oTye/aAVuttXuttQnAF0ALdA6KeOFM590OoFq646ripqaKSA4wxnQDrgfusNae+MKq81Ak59XG/ZJzpTHmD9x5tsIYU4l8fA4qUBLJImvt38B2Y8zFKZvaAuuAr4FuKdu6AV95UJ5IQbANuNIYU9QYY3Dn4Hp0Dop44Uzn3ddAZ2NMYWNMLaAO8KsH9Ynke8aYSGAwcKO1NjbdLp2HIjnMWrvaWlvBWlvTWlsTFyI1TvnOmG/PwUCvCxDJ4/oDHxhjgoEtQHdcUPuJMeY+3Bfe2zysTyTfstYuMcZ8BqzADe2PAt4AiqNzUCTHGGOmAa2B8saYHcBTwBgyOO+stWuNMZ/gfuGSCDxgrU3ypHCRfOQM5+FQoDDwg/s9C4uttffrPBTJfhmdg9batzM6Nj+fgyZtJKSIiIiIiIiIiMi5acqbiIiIiIiIiIhkigIlERERERERERHJFAVKIiIiIiIiIiKSKQqUREREREREREQkUxQoiYiIiIiIiIhIpihQEhERERERERGRTFGgJCIiIiIiIiIimaJASUREREREREREMuX/AdccYmMYRSOyAAAAAElFTkSuQmCC) 

####  放回随机抽样思想

> 对于所有的提升集成算法，没构建一个评估器，集成模型的效果都会比之前更好。也就是随着迭代的进行，模型整体的效果必须要逐渐提升，最后要实现集成模型的效果最优。要实现这个目标，我们可以首先从**训练数据**上着手
>
> 树模型天生过拟合并且如果数据量太过巨大，树模型的计算会非常缓慢因此，我们要对我们的原始数据集进行有放回抽样（bootstrap）
>
> sklearn的随机森林类中也有名为boostrap的参数来帮助我们控制这种随机有放回抽样。
>
> 注意：每次抽取一个样本的过程是独立的，这一次被抽到的样本会被放回数据集中，下一次还可能被抽到，因此抽出的数据集中，可能有一些重复的数据
>
> *因此我们规定，在梯度提升树中，每构建一个评估器，都让模型更加集中于数据集中容易被判错的那些样本。*

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C67.png)

在每次抽样过程中加大第一棵树被判断错误的样本的权重，每一次建立新的决策树更加倾向于很容易被判断正确。

#### subsample调参

在sklearn中，我们使用参数subsample来控制我们的随机抽样。在xgb和sklearn中，这个参数都默认为1且不能取到0，**这个系数控制的是随机抽取的比例，不是是否随机又放回抽样**这说明我们无法控制模型是否进行随机有放回抽样，只能控制抽样抽出来的样本量大概是多少![image-2.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA34AAABtCAYAAAD+g7O8AAAgAElEQVR4Aex9h1tTSff/9y96399uKgQIihoFaaKCgGJD1wqKZRUVwYIua1kVdRUrrq4VEFdWUFARGwgsCIoioLSV3hKjwvv5PXNLchMCBEhCAsPz3Cfh5t4pnzlnZj5zzpz5P9A/igBFgCIwWgT+/RdoaLD8RdK1t7/mZsvXk8eus9P2te3utl59+HqN5LOlBfj+3fZ40BwpAhQBigBFgCIwThH4v3FaL1otigBFwJYIWJMMabW2rMngeXV0WI8kNTUB//vf4Plb69fPn61Xr5GQPv6dxkagp8databpUgQoAhQBigBFYEIhQInfhGpuWlmKgJUQaG+3HnFobbVSoYeZrEZjvToSgvPt2zALZMHHrVk3nsSN5pMQU7XaghWmSVEEKAIUAYoARWDiIUCJ38Rrc1pjioDlEejqsi4p6u21fJmHk2JfH2BNqxixJI71n7XcdUdD+IzfJe6f9mQBHus2o/lTBCgCFAGKAEVgGAhQ4jcMsOijFAGKwAAIEGuV8STdkv+PJTEi7pfEDdOS9RGmRciMPfzZ614/IVb8d0JSyWIDIeT0jyJAEaAIUAQoAhQBsxCgxM8smOhDFAGKwJAIWNNiRFwhx2qS39ZmPdJH6jXW1ky+YQm+PLFylE+CHwmIM5Zusjx+9JMiQBGgCFAEKAJ2jgAlfnbeQLR4FAGHQcCaVjFCRL5+tT0U1nbxJEFx7OnPmuTdWmSSlJlYhKkLqD1JEi0LRYAiQBGgCNghApT42WGj0CJRBBwSAWtGvCSkgVjebP1HAstYi7AQa5W9HVdgzSA91sJRmC7BdCzdgm0tnzQ/igBFgCJAEaAIDAMBSvyGARZ9lCJAERgEgS9frEeS+Mm9La1+1t7zZo/HFNiiDfm2tMYnsf5Rt89BlJT+RBGgCFAEKAITGQFK/CZy69O6UwQsiQBxiyQWF2tM6Pk0beUaSciDNesyFtZLc9qa7DfksXa0T0r6zGlh+gxFgCJAEaAITGAEKPGbwI1Pq04RsDgC1gyEwhMRYpWy5p+1o3iSYyHGKlCNObhZk/DybWjpT0r6zGlZ+gxFgCJAEaAITHAERkT8qj/Vg14UAyoDVAb6ycD7KnzKL7TaVVP21up9z8fScquV/1NBEaorq61eh37tMow+++M/pdarvxVk4+OrYlR/qLFrTEfTHvRd2s9SGaAyQGWAysBwZGAwbkuJ3zAmRMMBnT5LlXSiysDHwn8sThzI5L7m3QerT+5JHlYlrm/eWb0Oo5W7mtdvrIqBJfGlpI/2s6OVd/o+lSEqA1QGxpsMWI34abVa0ItiQGWAyoCBDLS0QFtdbbmrrg5atdr6fQ3J4+NHy5XbGIOGBuvXwRJ9club9TAwxmQ0/5O26u7WYcoP3AayaAk8aBo6jCm2E6+vp3o18dqc6rnjtjmvr5T40YGbDtxUBmwnAxoNtDU1liEPnz/brtz19ZYpsykyQ0gKwcUR5LCjw3o4mMJmJPeMSB/BlR/wHAJjR5ADWkbH0FcrtxPVK8clAbQvnHhtx+srJX5W7hipck085aJtPkSbNzaOjjwQ4tjebruJl6WtlMZkprPTdnUZbX9Hympcfnv63wTpI/rID3hUN4fQzdHKB33fcXTZAm1F9YrqE+1THUcGeH2lxM8CnR8VfMcRfNpWdtBWPT0jt/rV1kJL3reV3hJ3QUtZKE0RpOZm29XFEph1dY2O+H36xJJ2a2A6AOkjssIPeDaTG0tgTdNwLN2YgO1F9coOxtMJKHe0Hx+Z3PH6SokfVRo6uFIZsL0MEDdNU0RosHv//gvtly+2LSvZQzhYmUbzG0nb0WRvtMSPrzMh74TEjwY/4buDkD5K/EY2SXA42XQ0XRoH5eUnklRWqI5RGbB/GeD1lRK/cdD5UoWzf4WjbWTURiRYirlWH/IcCSpia11tbbUcMRGSFPKdEBVbBKWxNGbEAmpcl+H8T9x8+TIREk/I/HDeN/XsEKSP5McPeLq8+TLQT317UCwoFsOUAapXRuPaMPGj/RHFz5YywOsrJX5UUelgR2VgbGSgqWnoST+xCgmiM9qskyTBVgihMEU0LHHPlnsULSnfoyV+pgLykH2DI7X+mUH6iMzwA57N5MeSmNO0xqZ/orgPiTvVK0pcaJ/qODLA6yslfrRzH7Jzp4rtOIrtUG01VIRPYh2ytWsn3x+MxBXVXELoKEc38FgIP0cb1ZMEyhGmJ/xOfjPXCkywNpP0kfz4AW/AvIXloN8HbiOKDcVGIANUr+jcgPapjiMDvL5S4ifoxKgAO44A07YaJ21lKmImmfwTN8ux0s3RBJ8ZivyRujmiiyffFqbaa6g6C38nxJFPy9QnWQwwx/1zGKSP5McPeIPmbao89N7g7UXxmdD4UL0aJ+Mw1eMJoce8vlLiRwV+Qgg8nfDZ8QAldPMjUR/HwrVT2A+YQzyEZGY43wezeAnLYK/fR2sJNbdtyXPE4mvKAjhM0keJnx3rvr3KOS2XWfMCfiJJx1eqY1QG7F8GeH2lxI928GZ18FSp7V+pHbaN+EiRxAVyrA8yH+0etsFIICG4jt7fkDYarI5D/TZc111iHSV7QXkCOALSRzDnBzyHx9/R5YeW3/H7AEEbUr2i8wLapzqODPD6SomfoBOjAuw4Akzbapy1lb0cYm5Nax8huI7e3xCL7FDkbqDfybsjrT8hjCQgjrkWQ6N8+AFvxPkbpUfTGWf9D23fEekm1SuqB7QvdBwZ4PWVEj/a4Y+ow6fK7jjKTtvKzLYi5IK3LA1EXkZ631Q0S0fre4g1dqT1J++NYVAbfsCjumCmLjiabNLyjsk4TvWK6hPtUx1HBnh9dTji113zFLdTM/AovwzVTV3QmN3ha9Dy8S3elL/Bh0bjlfduVOTcRMpfmXhS3mh+mu11ePemHOXl5XjzoRFdZpdFKChqNFc8xZPSJvPzHVE+wjzpd9pZURnoJwPWOrePuCcO18XRHnV8tBE9x5D88gNevza3R5xpmcaExFDZGP6YQPVq+JhROaOYjZUM8PrqQMRPjbonifhJJYVIJIJYGYhtV/Lx+v4xRK3djVtlpqIA9qD26Tls33gMec3tuL1eAZFIiqBjpej6mIY9GxOQWtoMTfdz7Pch6UoRfLwcPeYMvJ3luLx6KiQiEUQiCVTbMtFkznvkGbUaaubZFjzY5QtnsQiyoKMo6uYUQtOCt/n5eNfUTQdAczGlz1FZGa0M1NWNzqI1kDXMUc/sM8ZztIFdxjBSKz/gjdWAO3C+nWiqr0NdXT0aW+23v1e3N6GlU0P7GGOdmOD/U72iJGbgvo1iY2/Y8PrqGMSv9S0yDi3DdBkhWS7wjUzEg6ouaJqfI3GxB6SECLr4Y9PFfDRqBMKmqcfdn1WQiKTwjbuHK+s44nckC39GENImw+yE52gpOYYgqQgi6SwcfDX04KtpysfZ1dOZfAkJZS7xJCw6kmeYv9Gg0F3xNxK3L4O/agWSq9lBtOVBDGZIRBCJPRCV1sBa/ToyEe0hhkjqCt/d2Wg3SsfehImWRyBztK0cc3JoraAu4yGgCy/To9nfR0jxGO5x5Ac8u+ur2m4hQs4uHnrHPx2h14h1+5/m7F3wkYsgdp2PE0U9I9dvTQNKcjJxLzMX5U3tqHqZhczMLORXDz3m2l278ToxwT/Ho16pawvxIPMe7t3LRG55f0+s9nfPkXnvHu5l3serGhP60NWAisLneJyTg7z8Eryrb+/nzdVa8ZRNg0knE5lZD5CTl4/XNc39DQ+tFXjKlIctU2ZWNh4/L0JlP8816/YD40oHDfoiDTqrXiIrMxNZ+dXoHsc6zeurfRM/dRWyjm3GApUTS65EcqgWbkF8wn7E79mNuNgYbP/5J/i5inW/e65MxEOBMmpqrmK152xsSHqIZJ74HbyDazHBUM07gGctPSg/HsySOMlk+C8IQ1gYdy3egZsf1IKBrgOV9xOx1ocrj9gdCw9dw9n1XpAxBFCG6csOI7OyU/COXhl7in7DbEIwRU5YeqGKtfr1lOBYEGvFdAo9ifIeLbpf/AI/5jkZwpLecdZBfTrjSgHHsZLRdnIgmR2tNWsga19bm8m+wOFkg5xtOFAdzblP9k6Ooa7zA95YlsFk3hYgfprGImSmpeGBVbYLaFD7x3I4kfFNPAXb75se20zWzaC91Xh3eQUmiUVwCTuF190a1N7eiGkSEZyCfkN+hwP1FQb1mtjlHpd61ZSNWMYDTASpTxweNgvauKsIx4LZ+Z/EczsyG/VW8O7qHJzeGobpCgk3H+WNAs5QBUfj+lueJPbgdeI8Q8MBb0AQSTFp9jqcyK3XkcWe14mYx8wHufT4Z6VK+K44gIxKunAydP8jaEOtcV+khab2NjZOk0DkFITf8oc4a9aB9Z/XV/smftom/L2VWOyMBH6I/53nn0QpcZtsTkHUZFcoFC5wdXWFM6c8UifyvwsUCnes/iMXx+axxEtnvePTlwYjsawHWvVnvL53BrFLPeHM/6bwx+Y/i1BdnofsFy9xOzYQbmKunE4zsHhHEjJK6g1XcLtf4WAAR/LCklChJsKoRnVyOBRiGTzmxeGv2i4UHgxgOwXpPCS+5jsLoeDS78NTdIoXxWsIGbBGUBeyt8+BBwmDso/24Pb6+jHFgh/wDOpkD21jQPyejWC1WY03v4dCJpJizpGi/tYCS9Sx8wOe3EzG5TuFg3q0DIatpvEONk0hXiw+2JPHLYaoK5C00BkikQKLzrylC5yWaCsbpzE+9UqDxnvb4Ek8sURyBB4uQCeDqxqVycuhJPM8iQob0+t05Ky9+DxWctuQmHmk1AnuHh5QylkSKJm5G090ixtC4ueEyZ4+8PHxwjQ3/TxU7LER6RypFBI/p8me8JmpgruTnlzK5x5GQdcQ45uN5WKwvmCsfzPZF2nVqEhayMzvFYvO4C0zNx9/mPL6aufET4ueimxcv/kX7j96ihcFhcg99RMmu3kidP1JZFfWor6hEZ+bWtDa3oHWijTERcYijV8Bac3E/kWhCA2dh5lKXlHEcJkxF6Gh5P5C7Dy4nbWuSf2wNfkO7t5Nx8ElLsx+v4CYOyh6fBLhKmeIecInksJjfhxuFJXi8upJDEGTzj6Mou42lKfEIWSSXnnJnkHl3H3I0a0Y9aD0WBBH6nyx7xm7eqppKsfLojqWJHY+Qzy/2hR8HGU940/4xlrxaf5UpgxkgD9H0BzL1XCeaW4eU7JjUMfRDvyjdfMcYyz4AW90mPSg6mkabt68iZs37yK/TrDaX5mHVOb+LWS84lfru1DzIhVJB/cgZkcM9h5OwuUb5F1y3cL9183Q6oifFH4JL9BQ8heSDuzCztgEnEovxmfh1gUTbahueIjds8iYI8GMyBO4TtK+lY3yVi20PZV4knITN9OeolqtRWd1Hq4c3YOY2At40U76gB40lmXjRtIR7IuLwfaY3fj1xJ/IqTDcL99T+QQpTLoZeFXP1ZlPOzUPlT1q1L9KQ9KB3YiJ2Y2DZ+6g+LMeG7K4WZEUxlgNZSEnUa6bVGnQmLoe7mIRJN57kddJ+6XRyaft8bOMXmnRU/UUaYxe3MTdfD2h0nZXIi+V1ZlbGa9Qz+iD9fVKq6nDnU2c0cF1Mc6+6YGm/g42qcg8UgyPiBuo4eW4swCHA+Xclh83BMVcRX4dF0BQ04Ga/Du4mlUhWJQRED9pME6Uc4v73R/x11ZP1tAhmYG4x2waeuInRfAJNgaFpu0trkdNZ5+V+iEh39Dq11NfiDvnDmPvzhjE7D6IpNQX+GhMDrtq8CI1CQf3xGBHzF4cTrqMG1wb3Lx1H6+bteB1P+1pNdTaTlTnXcHRPTGIvfBCtwXJnLzUjSXISD6G/XExiNm1H7+dvIyMwlqOUBO5VaOxJAPJx/YjLiYGu/b/hpOXM1BYa+hhQNL5m0sndu9BnLr5FFVMX6aX/cHLPFBfpIWmMRXr3cUQSbyxN88wX0fTy4HKy+ur/RK/nhLcPLAXe/bsEVwxCPdiiZVEtQjRu4W/GX6PT8pGrUaLjuonOL91LrtKw5A3KbzDIxGx9STuV9Tg9kYPltRJ5+A3Zv9CI66skEEkckVUeju0PRW4uEwJsUgCV7/VSLhZiEbOUvf29/msi6fUB/FPWUFRN77Ctfjl8HYh7qdOCEksNrD6qT8kI9yFrCSJMWldCuoMBnYNGlKj4MFYDhUIv1hJV0FNTHYGEuoJdZ+E1if70ghpIWfgkYtEXSQXuT/WB6E7UruRA8KHQ+jMfZa4RzoSDgOVdbTRPAleY3xOIz/gjbY9mp/Ew5/xHJFAtekOOxHteYOzi13ZiZ/rIiS9JpOwZjw7Mp8bd8SQOsmNXLukCDn5FmoB8fMOX4UAZ6F3ixP8YzO5ya5+YsPXoacoCUs9hAuN3LvSMCRVqvWkUhqIg6lJWObBLX66RuFOhxYdeXvhbcqFzMkPO+5+1Fkz2m5FQE7GTok34p9yE1q+3BJPRMauwXSjdJz8d+EB7wanrsDv80k5pQj9vcJwTGtKw3olSZtMdMfnZItvr/H4aSm90jY/Qbw/P7fbhDvMAkMP3pxdDFdm3uaKRUmv0W0DveLbSVNzE+uIlVokxuSIZKTu9mV0WKxcjkvv9FuAWv/eys3ZJJgalYqPBnO6/npLFlx0rp5C4qcVbDsSkDlTxI+UsfnGalYvxR4CF2wNPucdwcJJvKGD70/EUAYn4CGvk83PcGQ+mdcSF24pnORG/Yg0BCffqsHrvjTwIFKTlsGDsYKK4Bp1Bx1a8/JSv7uGSKE1lGlPovMzsYchWGq8uxYJlVEfQiynkpl7uAUhDT4/PYbFJvo7J5+NuFquj9Q/aJkH64u0TUhbr2QX0eIeC0ipqTZ0zHu8vtov8eu4i01kQOCFZJif0nnH8CR1C2YwG+fFUIbsxs4lrG+2k8KZHYTlCp37JyF661Kboe1+gV98pRAJFK+74j5uZDzClbhwLF68WH8FTYezxAmTfYOxSHg/fCsu57/Gg+t3UWy0GkEmBJnbuZUkqS/25ApWV9tf4tfZfOe3FfeaHFO4+I6Tflqo/dRq9vBqYjUhZ6GZa30hrobkefLeGAbWsHs5IAFYzCVz5j43msPKByJgY3WfuGmaW++BnhvjhQh+wBu9LDbj8R4/zmvDG7HZn/Hhz9WM1YqMIezkVAt11QUsdSLjlxNCjuSjWdOFymuRmEIW9aTe2JZegWbizcETKGYC5oqAdQlITIzHipncPnKpH355rp/UCMvfU3oTCduXsMHByOQ0ZANi4+IQt+cMHpMJni5tMaQyKURiJ0zxC0LIiuN4SbZC9JTi1KoFiNydiEu37uDO1UNYzlg0RJAGHMQrLso0P5EySfyYcVmGGeGxOHLyKGIWTuG2ZjhhyfkqduGj8SpWESykvtj/3NAyodXU4OJSstAqxdwjxeNjoWSs9HQM8rWcXmnR/HgPF9tACu/YbHz+8CdWEwuMSATXRUl43W0bvdLrmBof/lzF7Esl8illiIkrFicJI7934/k+H64/IAYA07qqT5PMCQTET6LC8r3HcOzYERyIXYMAN5ZoKhcnoYTTv/7ET4OOmjwkhruzxE0RjuQqjoi2ZCPGk7VKkjnv1fuPcf/qfiwiwQJFEnjuzEarVo2qC0vZfbtOITiS3wxNVyWuRU5h0pN6b0N6BRtkRqf7YilkUhHETlPgFxSCFcdfotusvHpQ9NtsDh9/bLmQhvSbyTgeH4WwFSfYSPY9RfiNm/dK/bfgQlo6biYfR3xUGFacKGLd35m82Lmx2D0Ym389hqPxEfDn4ns4hRxHCecdN2iZB+uLtBrUXFzKGHOkc4+geBx62/H6ar/ET9OI8qdPkJubi/t/7kUYx/QlU39C4t3HeJx1E4lbw7Fy1yFs8mXN7C6+a3DwWhbzTt6rKjRXnMFCZyWCYi7j2pl9iJwtg9RVhdXnXuDllZ0I9ZyLsCB3SKVSSEQSeMc/Q2fNRSwlZNF1HVJ1LprkCIZKnFlotCoyEBmV+uNAgdEAJ+iU1RXnsJix+ongNO8IXjG+3x0oPBbC7SF0wWK652FiTwKItYjsq7LkEQOECJIgJsT6Mh7OlRPolOHAOgzCTUj1QGRlNPfH8My6EWNhCk9LRDu1g8im/IBnEWyaHmG3HzsWSH3DEDKFXV13XXSamZySPLoex7GEjCwgvmTHAk39ZSxnIlPLsfLqZ7Z/05EzEaQhJ/GGcx/rfJHAWRal8P81f8C+UFN7CeFMmib2+AnSFrkEI+FhrcDdzJSOdOP5ftaqIXJai5st7DO6iZQpi59IBKnfPjzn9i9pqs9jCVMeCVQxOUy5SbAyXzJpdlqFa5+N8+3C47gZDFl0WZc2YD0t0m6m5JveGxXmFtUrbRMe7eYXVXwRFsItIrguwmnGim47vdLJW08ZEoP18z5p0BEUGbhMdiA9imwNEkEkX4VrzGK9GhU3dyNy7Vqs1V3rcTKPJ4UC4tdvDimGa8A23Hyjt37riZ8IUhKfwkXgPUACDJ4oQBsnx823o6AkaRossvSgjAsmI1HF4mGXXuekfgnsIpBWg/rLy1kPNvlKXOX0VKf7IhFciMWwVu/FYl5e3chP4NvUC+uSX6KOI7Q6jLvzkcD3p17rkPyyrt9e56a09XAj9RK7I/IW70qvRsXpMM4bwQt7OHwHK/PgfZFAvlzWIa3NuK9y/P95fbVf4qfVoru+ADcPrIKPgrX8SaeFIfZkEg5uDYefkiijFB4LE5FXmopof3JUgwgi2RQErduPc3fyUfO5Gvl5xaip+RsxM8jg7ITFp4pRW0fOTapD7cd6NFc/Qsa1XzBHKoJ07m949OdqxrVAtuA0F3yFb+weNNW8Q0VFhYnrFVJi58KFUWIZPCMuoojssRhwUOlA/qFAVmBFcszam4N32XsQwFgn2ShnBbqNwIOlQ38bGGMHxIaQEGKds4YFypjEEBJI8hpjS8yYt58l3BiNsSX/j5donv/+O3piTFxpB+wLbaOn/IBnqXI05cSxZIbp88lCoX5ySvJQvz2JUM5CsOJyDeM22fp4F+taKZ6K7Q+4iZ2OnEmg2vlQvy2gMwvbmFV6EZwiUwfEzzziJ4b7xjtoMW6Dzmo8uXwYOyLDsSBoNvx9vaFixlUyiTUx+TNJ/MSYtDVT7xbVeQ9bGSuNGB7RmUy5O//ewlhExR7bkNVvH59+37ts8fkB62mpdqPpWFbfLK1X2qYcxBGPK16viBX9NHHxZMttK73i5aTj1RHMYyz37BxU7L4cyRV6N0+tlhA/bu4pX8MtlnSj4IA/a+XS1UOOyFQ+WqSA+Ek8ELohBjExO7A1MgxezBYhESRT1uBPzp1USPz0uLABZtZdfavba6fVduMVHxhQ5AR31XRMn85eKnfOg4ApoxpvT4ay5XNdgcs1ZD9uKx7v8mbuiaduxwNOT3UkSuyOjXdaBPppbl5adDxLwCxubkvcZhVeS7Aj6QEqdd5wHXiWMIubDxNyp4DXkh1IelDJ1Y3gOYstr2wxznFHoZE26s5P0FmJF539wJRv4DJrMXhfpEVPKXe0m2wxzgvy4eXB0T95fbVb4tfyKB7+zqyZX29m51w/5VMQFBGPC/ff4HNDCe4c34GYY8lIXOcHBadoJAxvVtZezOR8kg0UhldG6WKcIybyrjzsmSmBSDodPoyLjRTzEl8LVkc1aKl4huz793G/33UPl2OD4MrsyxPDfX48rmbex/0HT1DG+1MbD7jk/7YXOBTEKaPYFa6cwotcQpFYyHcQlu2kHV1ox235iRtmYyO01ogsaYqUCO+RPIl1arzsRzOla4PdI+RXiIelvo8H11qyEGEJmSRWw8HawAa/8QOepcqhafwLm0n4b368CTyCIiGp0dQjM9afdaeSecA/OBCejFuSGO5LklDCWw0ExM9rT56e+HXlYhfjsiWCfOXVAfEzj/gRbxajMwJ7ypG8fDLnlinHlNmLsSpyPVbM5vb9mE38JJi5V1DuzkxsncS6q/HEr+OvjcxqvXhaDLKFGDHtrsabkyHMpE4aljRgPS3VbjQdy84pLK1XWk0j/to8jZNLYrkKxJEivfVLayO9YuSkqwQn5rOkTua3EPOnsi6U7suTBUaBLjzZzQVkkQbhWAmxiKnx8eFFHD1yBIe3h3GuogMQP4M9flp8/nsrVMycVYqgY+wcVE/8pJgdn4ni/CuIYlyyJZi2IY2JZcHKdRfy9nix2IldofLzh7+/0TU3FhktWmjqMxHrz84/ZR7+CA70ZOewYncsSSrR9UM6EiVc9GH01vy8CCGtyjqGyAAlS96YPlMMt+CDyOO3M3WTo9siEcAvPJFnxG4IPpiHJm0X8niM5SvwZ6NehnVETSRF8PFyQ+LXr8xaDN4XaaF+cxIhZMGO3ydtg7HJln0Sr692S/y0XUU4EuQEJ89w7L1eiDd/J2Ddxj04eS0bha9fIfvGSexZFwKVExlkpHCfF4+s+ha8yTiGqCB/RN2qRdebm4jftBgzyWqD2BWzVmzCpk2bEBXGHb7uugbXG8hqRyfy9szUdzbyUJzkIy0xDd+DosOcnzJPGof6lKgQ+5A37esFVdjIna/PIpzzYWcmEOIpWHP1nYBwmn5PmAb97sAYEWuTJV05R0taCPkkk/1x1tkNWh+yB3K0uJl6fzwQaUtgYyd7HfkBb1BZMFfuNY3IipnJTmKkUkiZRT8XLDihnzCRfFoKSUAVKVw9AxESHIJFKzZi75n7qNCtdBvu8VNuuINWvgxNNxHBbQdQbs4YUCdHSvzISjkbpEYKv925aGLy7Ub+r5ylYhjEz4BUmiB+ndkxmEYwct3ABJUxbAO9dUT20+UB62n4jgP3+Xz7jpNPi+qVVoPGrBjMZCzlZE8du/DvsuCEfqHERnpF9uGVJy1iA8tIvRGb8xlVV1ezJE7sjuXJ+iBFjTcjuAA0UgQdKdKRJiKz3c/3w4epj3nET3/Wsy6O9nsAACAASURBVASeu58waQmJHxvVU43qq9y+YokKm+7wUVD1Lp0i6XycMrBMGutMCwpJsBapKzwDQxAcsggrNu7FmfsVAguiVhfcxWB/LyO7w8mLy1vThLK7iYia7cbuTRQ5IfyS4ZFHmqYy3E2MwmxmryNxDw/HpY9dKD/BnbVNyHWp3t2062EsR5SdsOJq45DEb/C+SIvuggNsvyj7CZcZbmCMm2P/z+ur/RI/ouAf3qLy3Ss8SL+Kc8f2Y8f6ZQj2doecGWiJUExF0No9OJdZjqaeJpRlnMaRy/lo0UVVIgqyhlFWsdsSJL6oR1drAY6FklUcKfzin+gG2q68PfDirINO4cmo0aVBGlpA/Jy8ELZG6Ltt+H3NUn/dOS+DEr/OSvx9YAmmMZ0CZ8kUSeAetB2XCxoNI5+Nk0GCDt5cp0GsQfZE+ITEhVh4iBVsouwBNDdQjhAjc747ugutpVxg7cDNk/Q7/IA3+j5IODmVY86BTNzkw767zMfxYs46oX6L3+fLIBIrsTq5HPVNzWhuaUFbp9HCis7iJ4LYIwLXPhDraDuKf1/MeZHIsehMJTOh6XqXiaRfE3Dser7+PL3PV7GCcaOSYMbObN1eHw0ZM3Rp97f4dT7YganMOCrDsj+4iWNnKY6H8i5h5rp6GqVtgvipy09wq+gLcLrfZLQVd6LcmMiJ02KyKfFzsLHecnpFwulnIWYm6+Ypn3MAmTc3cZN6F8w/Xsy6E1tBr7Rd75CZ9CsSjl1HPuel1VNxAeFKQjwlUP38FxrIfLDnDc4tYaP3Cl0+NZ/v4uepLEkVOc/G9utFaOQCg3Q+jR+C+M3BvgdlKC8vR8mzOzi2gj+32gnhF6sZF/H+xI+UpRxJi9i9hZLpW3CXO2al62k8vDmLYUD8EzQJ5rHd9SV4XtLIpKl++zvmy0QQK1cjubweTc3NaGlpQyd/RAUnhwNb/LQwLy8NGspLUC1Y7NLUXMBSbl9y4JESaDUNKC+pFhBODWousEFWGItvSY+BS2dAfB5bL3Ut7vzMHWkhn4/f37B962BlHrwv0qL1TtQg3gmOTfqE46BdEz9S0J6K85wCEqudHJN8QrF62wGcTctDRVM76kuycPnwViye6cpa7KSe2J75mQtF3YSXF7ZigYp3qXTGVJU7s1LrNOcXPOHMzJr6h0gI4VchiJnXG5tuVQhWbvTET6LaiRzeTcdEJ91TcgRzCZkbyOKnaUH53aOI9HPlVj0IgZ2GGcIwtVJ3zN1wDLeLGnS+7aOfsDi+0Do8BsSaRqxq5hCHsX6GEKL29vE/EbMWzo5sOSWk31KE2E5cXi01QRVOTqW+u/G4RQt19VWsZtwbRVCEHkcxcWfUNCIjeobAtYlf3CMWwIWIuVrCkjQdOWN/l7io4OvtASducVM8KQLXmT04bbizgRAkMj7NxmHm6CEttN3PsZ/fEyVxg+ecIPh7BuMgCSijS9uInJE9iFUXEc4dHSH1CMKazZuwYvYkyORc4AgLWvy0nTmIYVzTlIhKM9rvqSu/M1ZcqRv//Y2JOYMjj2uW0iuiL3orui92P26BVl2Nq6sncZErQ9lFFYvrlRZtdzawgUNEUsw+XMQE8ru8go2YKZ68Ftdr9Is1XUXHEMzs+RND7/KpRnXqJngxZIbVY6nCHZPdFZw3ALk3gMVvAM8xp4B4POLmpyaJn1aLzoLDCOQWfTy3ZrDkVF2Ja2vZ6JwkpoUqZBWiNqzD6sWzMdWJrV+PlpDsDETPEO6l5PonYgFcGIOrJW2MLg5GokjAw6HzIm6aXpC6TkdQeAQ2bd2CyBAVG0RGosKO+23MVqvdXlK4Tg9CeMQmbN0SiRAVifQrgkS1A/dJkBV1DVLW8y7AMrh7z8ZsT951VAqv7ffY+msHs1JqMWhfpNUHt3JeccXoqLXxMX/m9dXuiR/xD36TdRW3Mp+jvLoGFQUPkX4pEfFbViB4hqtgYHWCav4WJN4tQ7NulUODzsYKvLhzDGs8uQM2OUWTKv2xMv4m8nLPYr0vTwyVmOTOKYN0GsJ/u48qhuTpiZ940iqcvGdqrx97L/PSz+wGfiPi1/O5HNl/JCByNks8+b0hTjPX4vTzRnTXPcGJ1Z7snhC+MxA7Y0bYRiScv4si/kDQcTZwOPKgZ3bZyUSaWNEssV/KWkRloHRJKH9HJjGD6Quxyg1U79Hed2RXT0vte7QTN0+ip/yAZ7bOmpIb4eRUMgPRf7Or58Qj5M25JZy7lwKhiUVoq81F0tYQeBh4dHCTK9K/y4KRSFyWeHImDcC+lMv42V+/ICidPB/7M2s5748uPOMjbrotx6UP/GRUjaq0rfATnv8nm43DhYMTPxLMIf/3cAOPExe/jbj8PBWbydllliR+2lbc20omo2J4bMkwCDLTU3gIswlGzuG4yIekN4U9vWeXpNgiemXg4inBjOi/dRbtnjfnsMSV1RtFaCJeVVpar7ToerafC9TkhuWX3qP6RgR7Lp/YDeHnhQevk8l/G57u59yhDVw+e1CdnYiowMksqeHncORTosC02StwUncsiyC4i+A5iVQBD+8QRMRfQX4jr99aDET8yNFgj3ZzR0lIPBH9dwMrIy3FuBqzANMYUqjvcyQu3th4nZwNrUZtbhK2hngI5s/658jcVBaciNKeIUgU0ckh8+pB0YkFUBr1g2LFTKw49pg7B7UIJxbwJI4vhxiKmStw7DEfwZPkVYTL20MN6yWbhrBdKSgXBEMclKwO0hdpewpxiDlWwhnhF6vGpdcdr692Tfx63qYiYes6rFoaggBPDyiMhIe4a7rNDMP6+CT8VVgnsNCx7Lw5M5pzFeCESa7C4h17sTmYCLwYyqnToOCDv4jdEXb0KWpfncISbvXWNfQYXjEmaj3x4wnbkJ9C4qeuxOWV3MoVr+iyaVi4+yZKmskeQ341oQXl6Qexykc/+JN8JKpIXKsY+wAJ+nLy5aWfQ2JCXOYsZT0ZLRkZ6fuEsI6XKJU6XSPuMj3WI352YukaUj6FeJDvJBCLpRYo7MTNk2DAD3jDxsMYH3P+V1fiYjhxCZPCf3sqShu7GA8UTXcrPj4/hkVMlGpnrEszYVFXN6GyIA+PSTTqNuHYQFa9P+Pti6coqun/XnfjW+TnPsTDvHyU15kfHKz9YwmePnyIvIJ3aLLiuVUdz/cx0ffEyrW4UcvXqx25u8neejEmr08d8KB6m7SZOe1KnxHMVdixf1zolVaNz29f4GlRjcDdcKRzGzWaq0vwMu8RHuW9QOHrCtS26Pek2VKW1S1VKHmei4c5j/H01RvUc+RIXXkR4YRMS/2xPbUUjV1EHzXobv2I58cWsQESndchTeCeOVS5B8qLf6+rsRJFz3ORk/0QefllqDXu27RdaKwswvPcHGQ/zEN+WS3adAYcw7bo5tLKfVaEyqbhY2u6L9KiPXc3EwxSPHk9UjnXWb784+WT11e7Jn7azhdI8Nebo8UyN6hmhWHVz3tw9GI68sob+5E9gwbqeoFf/Jyg9FmIDQmX8KiSNV9rNZ9RlHYNOR/K8MeqKZA4+yEquQjNTMeuQVP+Gaz2C0NiER+cRUD8RrjHT1OXga2eUojkUxGyKREZb4ShcQ0FW9vTgMLUY4gO94XSNQi/Pm3u1+Ea1JMOSPaHD7HykWiZIyVb9vieHURntKjcW+KMuoHayRGJMrGAWnKRwo4sxfyAZ1H5Gajf7byP7cRqJnLF6qt84AXSx2vQ9OIQ6ypGDmZ/MYEW8zS1SI2aCrFIioBfX6CDuJtWJiPcTQSRSxh+L51AWAwkNw54n+qV0dzNAdqw8/52TGGCLa3G1Tp+EYa4pzfhxaFgxutM6vcLXhift+cAdTOrfzfRFxG31eRw4krvgrDfS8ftFiteX+2b+BGT9PM7SLv3GPmvq9E0yN66gRq8vbV1UJOt5vNrFLxt7UccNN3dguiaGjS8TMXF8+dx4Wouqo02wArz1jTmI/XieZy/cA1PBL7hzKD/+inyawShic1QpM729kHLL8ybfreTTpgQClucxTcQ6bDGfeL+Z4a8OtQz5CB7a2BF0iTn3zkaXpaI4snjSfay2lH9+QHPJmXS1CE1yoPdmySfirnhaxAZsQrhwZ5wZbxWZPDalDroOGKTctq4fTS1mfhtYwQiNiXiYWM7iv+IQWTEOuy+8WbwBVwbl3M8Ym+tOlG9spM5xzB0RFOXiijujFD51LkIXxOJiFXhCPbktk7JvLAptXpczzsN+yINOov/QExkBNbtvoE3I+AZ1tIvS6fL66udEz/HUypLNxRNz8FkoLXVcq5y/CR6rD8JIRjGwOIwzxJ3TGtha0f728xqD0vt6+PxtDNXV37AMwsLC8i6prkYN/ZHIsTHAy5SCSRSJ7h6eCNo2RYcvlmAhkEWD21VRpqPg40tFpBLS7c51StHlCENmotvYH9kCHw8XCCVSCB1coWHdxCWbTmMmwUN45r0WVoHHCk9Xl8p8bPDztSRBImWlev4iZucJS0m/AR6rD8JgXH0owkG0nFr7vEj7WZn5GdAXbXU0Q28rJKjSgbCfIzu8wOevZWLlscRJ860zLzcUr2issDLAv20f1ng9ZUSvzGaiFAlsX8lMbuNCIGw5N4ofgI91p8kyIejkJeR6DHZg2ZNjMkez5GUy5bvWDKYC48lIZK2rIMZefEDnr2Vi5ZnHI0DZsjheGtvqldUfsebTI/n+vD6SonfBOysx7Ng27xuZJ/Yx4/WJRD8hNrWny2DBCAaD3pjzeMcSFsR4mxHAU766QZZsLC07NqhtY/Umx/w+mEwHuSY1sHuFhomipxRvaLEb6LI+nioJ6+vlPjRQZMOmiOVARK50VKh721N6obKz86Cc1it07V2+9mr1c9aVmo7tRDzA57V5GikfQh9j44/DiwDVK8o8aN9quPIAK+vlPg5cKdLFW4MFY6cUTYUeXLU34nbKjmOYiLoBjmg3prtRIilvR2DYS3SZ8eRTPkBb0LI9ETQW1pHu+ifqV6N4RyE6oBd6IAjjSm8vlLiR5WHKs9wZYBYw6xJFsYybXskKsNtn+E8b4uzFsnRHvZCpAkJtbR7J5FXIjd27NbKD3iONEjTstJJtb3LANUrKqP2LqO0fHoZ5fWVEr/hTBLpsxObJJLJ+3iM3CkkmuQ4iokk56S+wvpb67s9WMNI0BVrkD6CmZ2f88gPeBNKtieSHtO6jkm/TfVKP6mmfQvFwt5lgNdXSvzogDEmA4a9K0i/8hHSZ223QGuRDnPTnSj7+oQ6T/akmYvPaJ8bS3JkTddkB3AN5ge8fnotlAX6nY4FVAaGJQNUryjZoX2q48gAr6+U+NGOflgd/YRU8olA+uzJHdHWOmntAC9CwkgImC3rRyKXWnvBgkS2tWWdRpAXP+DZezlp+RxnEkXbikbLpTJA9dWRZIAfBynxG8EkwpEampbVAh2Tvbl3EqJiSZc9kh4J+DFRdcHW7UvcPm2x548QMmufL2lrIjtCGeUHvAkr4yPEjeJlgfFjHGNP9YrKB+0jHEcGeH21GvHjM6Cf9bozpCgWjofFx5LX+JRfODZXQRFI/jVv36P6fRWqqz8ZytLHOvbehxrm95qyt/hYVDLsstZUVBqm+8nx2mk0ukXqb+s2Ju1UXVltHdyrPjJyY+06MXUgMjjB5IXWd2L1D7S9aXtTGaAyMJ5kgBI/OmmhE7cBZOBjSZnNCcEnQvb+KQVDxmpqR9Y21Z8YsvixuHTI8n8sLR9ZHgNg5pCdY03tkDhZi0TVvH6DakuRp+pPIO1prbIapFtQhGqy4DCe5IDWhbYnlQEqA1QGqAyMcxmwGvEbLGH6G0XA7hHo7AQaGmx7tbUB379bFpovX4DmZtP1IPf/9z/L5ueoqbW2msbIFjLQ2Ah0dABfvw4fPdJ+pI3b2wGSji3KS/Lo7h5+WcfwDZ6gjmERaNYUgXGHANWrcdektELjGAFz9PX/RlJ/cxIeSbr0HYqAzRAgE2lbTaBJPoSAjWTSPxxANBqgqUlfL0ISLE0yh1Mee3tWq9VjY8u2N87r82eALDqo1QApE2kjQu7I1dfH/k9khfxOyKotyR5fVrJA4WB/dFxysAajxXUIBKheOUQz0UJSBBgEzNFXSvyosEw8BHp7gX//tQ0JIPkQQmbLP96Saet8bVnHkeYlJMY8yaGfhrrgoFZicwa8kYoNfY8iMFERoHo1UVue1tsRETBHXynxc8SWpWUeHQIDuUVamgCQfAjJHIs/aukzjTqxolm6ncdTemShYqxk1nSLmX3XnAHP7MTogxQBigCDANUrKggUAcdBwBx9pcTPcdqTltQSCJB9VraYqJN86J/9IUDcKYmrpS1kwNHyIC6l377ZX5uZWSJzBjwzk6KPUQQoAhwCVK+oKFAEHAcBc/SVEj/HaU9a0tEiQFwfrT0ZJ5Nn6mI52pay7vu23t9pbZmzVPoOLrfmDHjWFSyaOkVg/CFA9Wr8tSmt0fhFwBx9pcRv/LY/rZkQAeL6aO0gGcRNzoEtJkK4xv33sYzwaSmiZsl0enocvsnNGfAcvpK0AhQBGyNA9crGgNPsKAKjQMAcfaXEbxQA01cdBAHi3mftoB7EfZDuq3MQgQDbVtZeCLAkMbNmWg52bMNAQmbOgDfQu/Q+RYAiYBoBqlemcaF3KQL2iIA5+kqJnz22HC2TZRHgo1xaa/JMSJ+DBsSwLNAOlhohPNaSCUdJd5yQPiJ55gx4DiahtLgUgTFHgOrVmDcBLQBFwGwEzNFXSvzMhtOOH6SkY+DGsbaLJyV9A2PvCL+Q8+ochaRZupzjwL1TKGLmDHjC5+l3igBFYGgEqF4NjRF9giJgLwiYo6+U+NlLa420HGSvEiUfA6PX0mK9iT3FfWDcHeUXW7gBW5qwWSI9cqzFOPszZ8AbZ1Wm1aEIWB0BqldWh5hmQBGwGALm6CslfhaDewwSEgaooCSkfwNYM3ojCeRC9/T1x9wR75B2JO1pCUJl72mQfY3jkPQRsTNnwHNE8aRlpgiMJQJUr8YSfZo3RWB4CJijr45D/L5poPk6GAB96GqswqeWL4M9NPhvmmbUVDWiq2+Ax3o+o7qmGZoBfh789ld8NVH+b18HOjerD1/UX2CyKL0a9NQ395+ojpT8fWnBp6patJooH7Td6NKaLMXg1RX8qmmuQVVjl+m6AOj5XI2a5pGhKsim/1eCh9FEvK+qEl/qGvrdR0MtNJVV6DN63vh95n8yedZqDfL7arpxMVDrGrzM/PMN7XWjlF+STm8nGj5UobHbvDbrrc1HVu47tJn1+Fe01VWhrt38WvWv5+B3vr7PxpWrj1H9XfhcH5pfpePK3RIzyyl818zvpP3Ge7AXUj+yGDJO/8wZ8MZp1Wm1BkDge0c9qmrbYGpoY17p+4ZvBn2NYULM2FXXPox+3PD98fAf1avx0IoWrsP3DtRX1aJtIMWiemVhwM1Pzhx9HXPip+1qQXNzs/5qb8SjI2uw+68mA6LQkxUNVcAhFA045+xGWoQC/geLBJ20Fp3CtHXf29BjIp2vT3ZhuvNapHSZBvlL1lZMmr4LTwYSdua1PjRcj4J/xGVU9/LpaFFycgE8F59AkTDtnnwkBPlg7ZUKQZm5d77kYZeXG4KPl/T7TX17E+ZJZ+FCzqf+BGYE5O9b0QH4ywNx/E3/EbA9LQJupM46ntOD97npSEtLM3HdRsarBuiqzVTlK57smg7ntSkQVp1HBviCrK2TMH3XExODcx9aXz9Ausm8SP63cSe/1ig/LmVi1ehH4qrwcpMH/PzjUf7RiPxV3kSCuxSr9uXge7/39M/2FZ3CFu+VuKxvXGhLTmKB52KcMGxc5CcEwWftFVSYkDV9/blvvdU4G+ZkJL/9nmJufK9Kx8GdR5FVZ4g0+bGv6U8slymwIcOcCf53vDkeBKfgE8gruIfbJnBOv1+KFp4Ufq/E6RAFFl+oNV0wC9xtv7UacpeNMCz+NxQd9GfKWdFfRAF1JXLTTcnjUPfSkV3Wpi81IfPjlfyRqLbj3EJtzoCnb2xbffuOnrZmtPWYElxblcFK+XzvQVtzG2xetd5GFGbcNjH+EH1PR86bDm7u0IvaC4uhmHcCb7/1GcwnGET6mvEgbhb8NqSg2uSY/hXP93pBvuCMYCy3EpZ2nCzVKxs3zljpFXrRWJhhch5A5nvpOW/Qwc0FemsvYLFiHk68/YY+fn7Aw0T1ikdiTD7N0dcxJn5fkLnZDSKRSHfJwi8iNzEILovOo/rbZ5Q+ykF2dhbORqqgXHYU97Kzkc1dOS8r0akTOhPE78tfiHLSp63LR+KDX/L79/SWIX7cYBNyGpWCsf579Q1ETJVBtT4FNcx9DYqOBMF50nIkv+tflq770ZgmD0RimRFraP2EvPXu8JyxE6XG5IUhLHXovHccv+7di3379gmuX3Gj1CgtTiwHJn5f8DhWBWVkGnTT475G3I4JQ0hIiNE1B9MVMvj+WmhEVEdD/L7j7fk1CPBUQipxw3T/WZg1i1y+mKoQw8nDG76+/pgzZ47hNXs2lnh7Y1ngLryq0ZM2vLuKeOWPWL73Ib4ZkTv1jfWY98MM/J750QRh1KfRW3YCixUhOG3YuLgRMRUy1XqksI0LTdERBDlPwvLkdyYIrYn+YBjE72v+fnjL5uLoa4GAcUkOi/j11uHSUgV8E17geeISBMzygYeTGIqpvhzOszB71VmU8dmYQ/zInjliPSOHgZOIkSSiKgmgQvZaEtdk8r29nb3aipBxYBd+2bMH++LjGVndtcIbUpk/1uwVym48Noe4QzolDNHxwvsJuFLQhd66ZCyRyTDJ0xe+vuZe3piikCHgcIlBY/TVl+DFzb/xqVLf5mj4hPYXWXiVlo7SZ2X9ZKf/AoPw3Wo0ZN3Cs5xSA7nqK3uMlzeu48l1ct3Ay9wyg98HT1OYvhnfCd6kXcb5nzkDns0h+FaIX31l8D8gXJC0eSmskuG3wl/hK/PHgYFXY62SL7Q52DFVhbCf4xAXtxWLVAr4r4pFXFwsIgMnIej4G7BdFjcWB/+G6wkhmLv5Gsq7hUXqQ+uTXzBXIYdfTCYa+q2jfcWzPTMgX3gONf1+E6bDfdfWofjRY7xuMudhE+/b6S2qV7ZtmDHTK2iRs2MqVGE/Iy4uDlsXqaDwX4XYuDjERgZiUtBx8LYBlvgF47frCQiZuxnXDBSL6pVtJcYwN3P01Q6InxIztv+FyuoKXFs/CR5bs9D57iRCnINxovQhEkICMMtnCpwlrlDpJv4sAZgTkYz3/KQUAxE/GZadfoWSkhLmKs6Kxxy57Ykf0IfmRyewP7kQLb1AT+FhBLr6YHvWZ4OVSE3pbZxIPIroEFdIVcuw+1giEhMTkXjiNsrrWtFXnoxY5//Cb1YEzuzdi3OC60JiGtBQh8ajwZgunoKFK1dhzZo1WLMqBNNlTliXbjqgw4DE72s+9vu4YsXVWrw4sQoRpweZvJBnvZ2w9FKDQX2A0RA/VqC1D2MwxWWD3grU+wnnFjoj8Hg52t5cRdQMOQLjruPOnTvMlXHjBu5s9ofX5M0oIMTvQzbu7duLM2tnwe//uWPj5j0cbvtwL/sDUFeO+z85Y7psFvbtNsT03L5EvCyuZSflLS1gOzwj4kcsbc2PcGJ/MgrZxsXhQFf4bM/CZ93CBHHDbMGH4iIUFZm4Xt1D/Fw5vKJvocDU70VFKK5oBLHjjYj4dRfj1MowbL3xFrwU9DVcwXIXb+x7yS08fCETqknYfE9wmHdfM+7FL8euO43oG4j4EaJHSB4hd8OxmH18hNsbV2HXypWIW7kSO1evxpZ5HpBJZ2DRhk3YtIm/NuKnADdI3AOxaiN/j3xuxcncdpb4Oc/BUeOFje9vkLxxOX7Tm6v1PWRfI/4IdzYkfr01uLbKA6rVf6Khnt3z1/f2b1xe5IGZ//kvvCQizPjPjwhZmIB/3tQNTdQ+vsKTHXMw+7//gd+qKwbPf83YiZXu7pjn7gK/H/6L4M23DX63CPEjbTHOInfqG7D/N3MGvP5vWfkOJX6WB5gQP9UCHEjPwLPqT0heOg3R9xtRnHUfN2J9ESwgfjXnFkIRchJv67IRH+gK98VnUGawzvoN1TciME3igpDf8tFpUNqvyNs1HfLFF1E7BJf7VpeNX8MmQyJyw+ZMc7wtDDKy63+oXtm2ecaW+Kmw4EA6Mp5V41PyUkyLvo/G4izcvxEL32AB8as5h4WKEJx8W4fs+EC4ui/GGQPFonplW6nR52aOvtoF8ZsZ/xJftc+wx8sdUentwPe3uLR9K5KLyBS1F7V/LMeUpRdNultoH++Ct9INbm5uUMhEkMgUzHf3kOMo7SYWPyesv6PviHs/nEaokyHx+5oXD1+ShkIGsUgGZzc3KD234+AuPyjd2LRJ+uR3kVgGhfCeMhCHyYrn9wpcj12PyMhIrJg7GVL32VgeGYmoIyk4F+HBlIlJg3tXIRNDJJHDhftfOfcQXn3rQ8v1FZDLJsNPaFHznYQp4iVIL/uET0eD4fOjGxbPmoVVgmuxyw/w9N6tI34zJ23Gy1rufLkv97DZ1QmRfxYg48olXLpkeCUfWI4psulY81syLl26jPTCZkaKtPn74eO0GBdr1cj6WQmPHQ/QXP8BlZWV/a+yC1ihcEPElTeorKxCY9cX5MX7MvgxdZU5w81NCc/tB7HLT2mAB2k3MdduPEbKwMM6t96Bid8bfO9rYKxW8068ZVd5iWXj31qU7ZwBnwW/o6WuAX2vT2Hzjz9g9nQhZioE/L8fseVUGb6/OIooyY8InCH8fRZWzXCB5w/eOJNThcqkjYiKiEDkirmYLHXH7OWRiIw6gpRzEfAQygP5zsiRBHIXTnaUc3Ho1TegMwVr5CYs0AKLt84qbeKeLPQ0PvSOgPipy3FptQpSxRzsyfnMucb2of7yciimxyGXc+P9XnYMgYqlSK4XsNW+egbfsLPVINDIBAAAIABJREFU6BUSP0L2OjosFhSlt6YGvQ0N6EhaBC/pCtyva2GthoylanBXT8bixxC/ZpTcvayX74sJWDZZhsAd5/X3Ll3CH7eeo/6bMfHrQ+vfP2OqcyhOEX9SckRKUyXy1k+Cp/sSXM98je8Ndeh6nIQ9037EvLVX0GZyryhrgftenIIzIUr4qsKx3feHfsRPR+w+ZuO05w+WJ37E5Zu00QT6M2fAszkclPhZHnKG+M1HTEwIQhIf4jwhfukp2Oi7GYdi/QyIX9WZBVDMZ/vNvuYXuPu41sgjhazateHp8VgkPuH7Rr7IWjzaORXypckQdon8r+zndzTmHkX4NBf4rotAkIwSP0N8rPQf1SsrAEssfirMj4lBSEgiHp4nxC8dKRt9sflQLPyExK/qDBYo5uM0mZD0NePF3ceoNXYoo3plhTYaOklzxkG7IX5qss/MbTVuNAsmnaSOXbnYNVOGqfOjsG3bNt0VczwbjX1AX2ctyktLUFJ4BVFTZPDaloqikhKUvmuEWmMe8evr/ITSoiLk/xEFD/liHH9WhKKS97i7UwWP9VfwT1kZysrK8E/yGiinbsC1Evb/suIzWK7gSGRvPfKuncPZs6cQHSSH1HM1Dp05i/N3i1Fd8Qr5+fnc9RCHF8gwOeICnuvu5aOg5BO6+lji5zQpGtm6PXXAlxtr4ffjQqQXpOLw1B8QujEFPQauitV4tt4NnrP2GxC/fGLtIhPAnrsM8YtI+gOb5/LukoLPmZMhEzvBw5vcm4O1FypIyBU8jFFBIl+DlM4vLPHbdh3nw2U6t9yBSYoca1M60PmpFEVF+fgjygPyxcfxrKgIJe/vYqfKA+uv/MNgWlb2D5LXKDF1wzWUcDgXn1kOhc8v4L1xGeInm4nwaL79oxA6VYZAZmWXdFZTMC3mIRjIiIthXRn+WiTFoh33GLc8hviJnBF/vVpvVam6gt0yQvwK8GLzVHh6bMTz94Zuc9pb6xH4wyxcfFKN+pw/cO7sWZyKDoJc6onVh87g7Pm7KK6uwCtBOz48vACyyRG48Jxv73zkF5TgExMxqBdfv2ig0RhfTXgQ4wUJIXuS6Yj+m8iu8TMafPnKmreHY/Hra3uF08unQOoahPjsRv1+yO/vcTpUBqnvryhkOuw+1P+xDE7O/viJxznmBHIaWXdQhvh9e4+zgXKsPfaPHkcDOTTET0duhnrmw9/43dcda36+gPclxajNL0XPh7doKS1FS1kZmj9+RH1lKYpfV6GR26fbJtikqyN+JZ9w95dI1spNLN2rF8JbIcW0YM7yTe6tWYOI6Mt4/dWI+PVW49wiJygjUsB3QX1N17FJKsXGxFcMKWXrU4f25J8QIJ6H6wWcJbhf/apRGueL0NB9eFX6Gn8t/NG2xI8Q8gng2mk8BJoz4Bm/M9j/31reIz/nb2Tcy0FeUSVahDz6axdamjugMbYE9X1h9pV38v03N0ElLsXfvrehMj8Hf2dkIa/sM2O975f/txa8Z565h5y8IlQaZMo+/aW5EsXPstl0XrNeAPp0yJ7CVnSRsn5rwbvn95GRkYWnb5t1Lue9XdUofHgPGZmP8U89b//nU2Df7ybvf+9AVcHA5R3QMqFtxlsm32y8fN/an2jxWY30kyF+YTiVtg8Byw7h0JJp2LBnCwI2pyNnnx+CD2XiMbPnNwVnN/lA5rcFF1L5/b7CPYBDFUCLrK1K8AtuJp/WFuG3IC8sP/EczS03sUpOiZ9JnIQ3B5Nxu9Ur4HtPG1pZxULLu+e4n5GBrKdv0cz3C71dqC58iHsZmXj8T73Os4atugX0Clo0v2XzzX75Hq3GREuI8Yi+s8Qv7FQa9gUsw6FDSzBtwx5sCdiM9Jx98As+hMzHbHyHlLOb4CPzw5YLqbq9tsI9gINnT/VqcHxG96s546CdEL+nKDkeCJdll/D+/UvkkD18D/9Bw3ctSo+HwEnkhKDokzh9+jRzndg8G04hJ/FO5+YJfCs+hFlSqWFwDGaP39AWPx5mwz1+WuTGqvSEAkC/4C4kfZ746RIhLo8SyIz2+PE/o68V11c5YVpsLktUdD+QLyaIX2srvlxZBb8fQ3F8xyz4yOcjJd94wlmNvEgFPGcf6E/8yKT041X87CIflqtnX+MNrFaKIBISvx05UH/9Ak13EZKiVuNwtmmCotF8wVfdhMjI1VObi1jVNMQ85GdG/YO7fPkrqj/xkwdiywm2/U+fOoS1XnKO+PWi+lwYFAvOoIrkSfaRVd/CAXcnxF56xxCUwYjfxu07sEoix6bj+YLJPUtgtDciMfuH2UguZi2gxG01f783JLL+rp5sM/ah9foqOE2L1VnRDJp3gH96Cg4icHoYFvnLMCMsDL5zEpBvsBfF8EWG+EmdMMXXcG9j4KozKG3gg7tooPmQjpg5LpBOWYYT+W0GLrjEmusrFemJX99nXF/pgqlh0Yjftw/x0WGYIl+ApMpa/LHYGcsTS9Bb9xyXZkmw7rcCyxK/hk/4fHs/tkz6AX6ea/H38xo0Hw+F53/+AxV3efznP5D/8AO36CBlXDT5cU9H/IxdPRn3Y4VpuTdy9eytPoMwmSsi0/Q4abO3YbIsFOdKOw1cWPtKT+FnkQTbzrwZGIePH6D51ADUlSI11EbEjwRwmWBWPqFmmDPgCZ8f+PsXvP0zCj4KGZTT/eDvrYJSJoZs+nZktbNvkbFAKVuIc590HR3zQ1/rdaySyRGRxikwQ/yIK/oJxMxVwm2aL/x9pkAhcYJXxAWUCPT8y9s/EeWjgEw5HX7+3lApZRDLpmM7n2lfK7J2+cNVKoObyhcB/iq4SuXwXHcN73llIPn5KbD8yEVE+yuhVPnAR+UGmViBgNi7eJmyHXOUCkz28oHXZCeIZTMQcfW9jhSCed8Fq5NSsTd4ElyneMHH0x1ysRxea8/jH0F5+xO/PnQWncNaL2fIlDPg56uCq8wVAdtS8YGfHA8Muvm/cMTvTGU72js4V8+sVrR39iB/vx/mxZ9AXBi7Bz1A5QyJmxcCdR40C7Dl+gfUpUZjHr83fF40Uskqcr+/HtyOdILYYwdy+OGq3zPAN42G8Tbpa/4TyynxM4GQ/tZQMm63eoVvKPzVD4rlR3Ax2h9KpQo+Piq4ycRQBMTi7ssUbJ+jhGKyF3y8JsNJLMOMiKt4z8v9qPSKGDiKcG6tF5xlSszw84XKVQbXgG1ItaxiMRa/sDOVaG/v4Fw9s9Da3ome/P3wmxePE3FcfIcAFZwlbvAK1Md6WLDlGl7eonqll/ax+WbOOGgnxO8BUqKUmPbzXRScXwXf6e6QOa3F9fxTCFO6Y5JSaeA3r05fBxcD4qfFy30+kIik8I5KwMnrr8HsUhoD4ve1IIGdUPtHIO7gDbzXAr2tH1Cs27eVi6Nhckxel4x83b1ivG8iPYQR8ePO6WOJXxjSnmfj8Y0U5PySgNxi4R6jKjxaI4fnvKM64uflthwpGZkozcxEacav2CgRY31qFykMynILDTay99/j9wVFBwMgc3GFwklg8duRw5DVb/8cQoDMBztSC0zuV/unWj957rfHb6TEz+QeP3YT/9enu+Gp3ICMjl5mIv4tOw5LpKFI4TAajPj9fPI53ly9geeXf8Gla8UGxzl8+XMN/H4MRvInblb1tQAJvlKIpP6IiDuIG2zjGuzbyz0aBvnkdUjO1+/jK37fpJ9YGfUFfR1PsX+2B5Yn5+I0iep5IAs310/H3IQXRvtN9C+yFj8VfkpIwtmzZ3XXuevPUN9IiJ8TFmyNQYhSAregXbhbbTRr6WvErYhJcJk0CS5+rMWvr+EqVihnIv45+6w2Nw4zXNch82MZbgRLsPbgC/R+empx4qfOy8DrUnYhg7hHXth2FOWVtQzxm+m2FvcLivG5WH/9W5KGPd5yhJ6u1FkvhcSvM+cgflq6FEvJtWQeZjhL4BGwhP2fuf8TjuSqASPi13prDZxli3BeN5HvQ33yEsiI3JGTRkhEzGbuCJWa2zjs/gOW7soemPjxVkBbED+yl6/LdMxcvdSM/2/mDHhmodCejvVKJZZfeK87uqev+xNeFbwHz3uGN0ElfYYnNtx4z45L6EN70WksmySDf0I+Z/lrR/p6JZTLL+A9f7JNXzc+vSrAez5T9KIh/x4ev+/gZL8Xzdk74S2bipiH3HYGhmhKIZL5YvvdGi7tLvxzfAEUYilkqpU4V9LOLgJ9q8OdzdMh89iGbN7wx70vkXlj002+vN/QmJuAEBdheQFj4tfXkoXo6TLM3JyKD0wd+tBemIgwNwUWnK7gAq6Y1QKDP0SIn4cMrpM94OExGa4yCZyU5LsHlAq5ILjLV7yI94LM/6Bu2wCfcPebLFy7cgWXD/2EKfL+BJ55rreS8YoQDbjQx6fGfjKBtSjxMwTF4L+hZdxu9Yohfr6QimTw3X4XNZy6df1zHAsUYkhlKqw8V4J2Zv3gG+rubMZ0mQe28Yo1Cr1CXwuyoqdDNnMzUlnFQl97IRLD3KBYcBomI10b4G7uP8Ti5wGZ62RGlya7yiBxUjLfPZQKyAXBXb6+iIeXzB8HjQI7Ub0yF2vrPWfOOGgnxO8l1FV/YPlkX+x+0oVvb44jULEKR48sQciBazgQKINMYbjXTiYgfn2tGdis8oK3pwzT5i+Bv2IaViW/gYZz9YwaYo8f3wRfn8RBRSIKxqxD2MpEnNkxXIufGrlx/ggM8odsRjDmTXWG7/Z7+HB7A9wVCii4Sy4VQSST6/5XKJRYcYUEReH2+CnXI+V5AaqfPmWud4lL4EtcPcvq0Pc2Hcc8RZgVegQfdBErq5D9kwSeoSd0xE9oLWGtJj9ie3I1vjzbCx/ZdGy7zw3+xCPI6DiH3rqrWKmcgvUHYzFLYUz8OvBg+3TIZE5s+eVSiERSyAX184jOFLgxfcWTOBVk/msQsy4MKxPPYMdILH6DED903sUGdxX2ZDSBRGCs+mUWfGb9gipicWng9/jJEXPysQ7T6keJ2CZl9/j11b1FQbQnZjqF4vqTGt1k/ssfP2GyeDH+bGJXg9W5cfAPDIK/bAaC502Fs+923PtwGxvc9W2rYPCQGeChXHEFDaYWlHvrcWezF5RLzqHii/44hy/VV7DCw5tx+TT12tCuniKIxG6Yu/2GURQ7VtI1+b/AT+6P+BNbofInxO87Kk6GwDngMP4hHLdXi6rf5mCm7z68rym0HvGrzUdKqBye7mFI/qtUYHGtY4mf+0Y8rzZyH61Jw8FJIqxM/qizYAqJn7YqF6m3buEWua4fwQoPGUJ2XWP/Z+6n4mn1NyPi9w1FB/wgmxqDRzqO/B3vToZANiUGOuM0gY8cE1KXjd9n/ICQzek6WRnQrdWaxI8QPhI1tV88bb5Hm1if5gx45iDS13QVPzkpsfp6vW5xwfi94U5QZWFnjfaof2WOKCEyx3C2viZc/ckJytXXUW9oRDTO2vB/7UPETJHrj1jhJpjyZZcN+hxGR2RGHjEAtI93YposBL/zUdK492WLLxgFNNHg6R4vRkf0HFMY1bMXtZfC4ey2Fim8rzRT0h48iPaALOAwSnirpGENhv9fbyOK7+ehkqzu9taxwV2YvRF9aHmdg9wK/jgHNTI2uELkvArXDcqkz5Ihr04DED/NfURPlkIiUWDNrRZdf6N/2/AbJX6GePT7zwwZt1u94omffBkuCwdzIn9LZJAaLy5oH2PnNBlCfn/PLniMWK+A3tpLCHd2w9qUZgMZ7HkQDQ9ZAA5bTrHQWHwfeaxioY4J7pLNLPb3tbxGTm6F7jgHdcYGuIqcseq6YZn4Nqd6xSNh+09zxkG7IX5fQQbCWcwem8YyQvzWIqWuCc09ZUgMVGLdjU9obGxkruorq5hIXayr5zeUnwiBy7yDOLyanOP3ClXpm+HrG4uchjREOrli873Bg7t0/3MN8VFh8FGS4C1u8F2+DUfTCpGxUwWF30+I5vYWRi/3gVzhjxX8HqitS+El0weK6Wu4jjWeEfj9KIkkdgrFL35DiOtkrE8TnElohqvn1P/+V+fixru6qX5giR+ZYKozd2OFWIzlO//m9vpVIjP8R3gtPqMjfiS4C7PHj1geqi5jt0yEHRcqgYZCXA53gXzecbzmBmJj4ofO+9iz8SIqig4jwIj4dZUeR4hiJnY9YZehNXfWw3lSNB7oJsy8oHfjn2vxiArzgZIEb3HzxfJtR5FWmIGdKgX8form9mtGY7mPHAr/FTqcty71gsx4j59EAY8ZMzCDuVRQyvk9fkxITaREKBH0czo0n57g4iwJ1vz6TEckGIvfD3q3QR2m//mBCe7CHNpemYkzfmJ4++zEP9xeP/WlnyAnrq4dJI8GXF/jiYjfjzLRrE4Vv8BvIa6YvD4NHC9kLLZmu3r2dSD/aCiUHitx5cM3wOA4h17U3FiLqcoFOFbAT2J4XM0J7iJH2O/66J36N9lv2hf7sGDzX6h7uJMhfq8+ZyJ6uhuWHfsDR9ftRvbzl0gJlWDepjRo66xI/AgpL8/ElUXu8BTNwNFU1jWXRKYlrp4zTRE/RpYliEn+wEYSZeZ+yVjiPAdHXpThofA8v5RjWDVFjpA9N3X7EJjziB5WoKtXuMdPiwfRkyCbfQR6b9HveG+K+BEIvxbhqI8Y87b8NTbE799/KeEzFmoA5gx4Jl7rf4s5h8ofzrKpmB99AulFxvvoWLd/8109ZVDF5faz+qszNsJVNh9JjI96H5ofxMHfmexnj8aJ9CI06octfRm/teBtbhqSfz+CX/ftwa6Y1QhQyED24DJ/zARTBs+9zw3za7uBVXI51qQYxa3M/wU+MhIRl9s3wb0/Y/dTw/fJru+/ouCiK6+xxe8LsrYoIZ25AWdS+P105DMFp6O8IFVE4S/ekqmvzci+df7/9q7ErYmjD/9JrSJsEsBgUe6CgIoCAooClQoKKopFVFRUWlFRqaWKN7QeaBFBqRceqCCHyFGRK6B8gNxy5Xje75nNweYANpJgkNnnyZPNZnd25p357ew7v6sYZxPV/t5bsNpJBK9w9XyyC0dyG5Uv27J3SPcnc/piJDwwfPPJXlClr1PgJfRHbKwXFsXmosfQKhynBZT4ccAwuDv1GDeO+M2gXKmIH+N2AC/U5ptsG3twLVIAwcYcbQudsRIc9mSwPK2KQ/wYGC9X5FmzA2IbD2w5k6M1j+VkxMLdxhaxphMsFJ9N1MTR2LLaCSKvcM172a4juWhkHxMyvEv3B2O1AIsTHmisIrhdTuWKi8bM7vOZBy2I+AGjDxOw2DUJT96oiB+xXpIR4mcL302HkJKSwn4ORi3V+PjJm7IQLhZj47X3yNEkcB/D0JAUivYshAudsffJuKQaiur5uSARK8N2IjVpHRyE6gTuSh8/e/94pJF0CidPIi3ODyJxAH5JU6VYOL4FvprUEIN4cdAbTnH5qD63Frasj58MHyor0MZd6ZyC+HWfXQN3YSQK39Sgt0b56coMVwZ3qVabdzai5oAPfpzngqM5tVB8rEXemvnwCL/Ag/h9xOd/tsHDZhFic5WrNXrEDwpIpTJI3+gSv/toyE9C6La/NavBPdciIXLeCw7EqlH+GQWJKxG2MxVJ6xwgVCdwZ0097eEfn6ZMU3EyDXF+IogDftHgfHyLLwS6xM82HBmvVOaTZYU45CdU+fiR2ynQc2szltmF4/bl7Qi2CcA1TuANlvjNF2LfhTcaTHvfZGI3G9ylWmPeOXL/ACIXzMOauBz0tX1E75UICISxuE1S0r04CG+nOORXn9Pk8ZN9qESFdufy9PEbRFVmOBwFXkhUp/PQIn5KMvtwrzeEP4Tq+edNrfGbIoG7rAPtnXKMPibELwX559fD2WMvyqoe4dYGF3gKRPCZ74Bfb5BUF+YlfqymrKUUj349hYqyavSzkTInJn6yooPYMN8Rx/NUmtmeHshbL7DE72jhNSSuCUJQkOqzehmchdZYtHS15pifqy2Y0Itok3GJH8knKgbjNx6umoyp9sthYOxicUdtBqd+fo89Q5ILA9/USmXkUZLGQm3aqfttSo0fuQ8JXjQHA7eooZ/sm8+EN9n1Wv8p+vFfwWnsCnWHnTWDxQE7cb6kS6MBnOwFdYOejx8Dr5TXekFOlEGruGlIFOj/rwCnd4XC3c4azOIA7Dxfgi6VBnC0/hrivO1g574GW5NSkZ6RiYsXk7FmoT7x08sbqCJ+uil9xjTET70KqMw76JWim48VIPV1ZEmi8lz25U6Tx68PN34WYIGWP924/09AcAoe6sqRFuBG/BhpRsm9fOTnk89fSFwmxvpjt1W/C/C4ppvVjCi6cxC10AshwS5w3/fcgE+9irxyNX6jg2AD27AL0d4Q+KSi5MkBeIg34rpWbh79+lLip4+J/pHJx7jFypWa+Hn/pmM2rCJ+m3O1g7lwiB8rLaoFFePlCui78TMEC7T96cbzKAcjxXSCheaSeyo5ysdficsgXn8Mt1k5y0fB4xp0k8UPRTdyohbCKyQYLu778Fxv0Z/Klf64n7kjfOZBiyJ+n+/GYZHzXjwyQPyWbUnF8ePH2U9qjLeG+A3c2wGXVSfwZlQ/jx9LaIQB+J0TBUaL+Mk/oPJlPXpVE+u0grsMFSLeaTmOVg6j9bya+CnQ30wiW477e5WXG/LxI/9XoOG/D+j+Iwgeoig85Zi4KX38xjV+7Etmw11krPTFkYsvIWurQE7gfHj/nM2L+EHyFBdXe2Dz+XLWJFOf+CkHqR7x++U6bh0I5/hLrUPwUjGsRa5YFaryq2L9qMKx56ZENdJNFNxlMlNPcqf/PcQpl/nwtLFGYOx1DHBewCfz8SPpHFiNH3t+A97sXYmft15ES9tHtF8Jg0C8HYUjQyiMd8Lyo5UYbj2vIX6K/mY2Giy3fw35+JH/NX5+il6U/REGRxtHhF+oGV8t0yV+pE2j9ciOcgZj74c9txs05rNfSvwUn3vRx1mEIC9yXj/uR+W756h6/E6JQ2sFctcI4Oq0A2WNJDjJDBA/gn1bBe6sd0Bg5AX8r5VL/JpQvN0NQe4+iFq5HKEkbYnrLlQ0jJuAyt+cwjrhMv08fnrBXYhJpzdso3LQr6Pxe0LMkTmLDSz8RXtYM7iM95woUiwnv4oNQiEir3erxrjKB9BQLsPpED9iykn8fEkuPuJjSLdJEeAz4U1agME/ZeipyUNqOJHD9bjIaueA0QmCu5BFiLUGiN+S3UV6xGMwNwa2TDAym/RtO2U9NchLDYczY4/1FxshV3Ti2s/2YJYfQZnG5488I+5jp4PpiZ+h+vbf2qRVX23ip4z8zHXBMAinyQ7KMTbUj+72SqQHOWJj5nM8L7qPgts5uJZXzkbm7cvfhh+cEnAndzscXffgMSc9qboaXM2EtPU+UoL9cYjkNe1/gF0uAvidqIaM9e0WYeXJaj3yri6HfFPix0Vj6n29MU6Gs6XKlYmIn/FyRTR+2yFmtN9jp0b3y8+Qjw2hv7sdlelBcNyYiefPi3C/4DZyruWhnJhM9+Vj2w9OSLiTi+2OrthjQLCoXH05/tO9ks88aBHEzzl4C3Zsi8ZqFwHEG6+hTW3qqdH4LdIK7jJ4K5oN7sK6JYzUo+odMePQJX4KfCTh6RftxL8cKw8t4jdSiB1ie2xlozcA0yJ+GMTLgkfoVMg5xG8URXtdIWQYMNyPNQnbr31siRWDrSdK+RM/9mVZ9fLb+hLZy+fDL/YGP+KnvpakepDL9Xz81ANPj/gl3EZp7jlNMJGzZ44h2sOGNaUJO8gNNJKJm2WfVMWYiPjZRuJ8eQmK7ubgyp+HEe5iAzu3ZfB1d0ZsTi/wvwa82OYIl++IpoqYtI4TA/7Ej1yj0qr297DRQgVLdrN+X4MvC/CoU6GVwH20aC9chdr9yDDWsLKy1u5vhoFd+GV8GG1FYbI/xNZihJx6itLCrPH8chdPIeZHBksifsNFTZ7FLDx8W4Ur0a5grB0QkHQHxD3NIPFT9ON9XSuGO9VRPfXtxAZuRmOhUyxySAS7z58xmrMFq1x34U3LOFbyklPYKrDB5qMqU1k+xE9Sh/biYnyoGfeP5OLPZ38odwdC5qmjq3KJXxs+3TqO33+Jx/Ed8fj90J+oqGgBWgpxcf16nMupgvx1GqKtPZFWqbP0qCZ+OeW4/UcGzp1Lw2ZPIbwOl0KqFdxFjvcZgWBso3GTmPWqt748bBHbIuxyq0bTQzSB3bmxcCAmbySHkaGNaORGRpSmqN31uBFkhaWRZFFmHGfNviaPX64yeExvr5LojY5SzZ4hbCc5xmfCm+Tyyf/quYGNIhFicpVqKxL8yJlov95qE/LBwp1wXKAb1dMGjH+6ThCGITxJcgPjth/F4wYpOnXowY2NIohicjFENAZLGb1o0FKSe5MxNfEzVN8B3P/FSau+2sRPjjaSF1S0FucMEFmdhrE/ZX0f0NDQip4J22/oKgXa/9oAe5sFbIRfkv9VvNgVnj5+CAhZj7AN0diRWgCJtBM50Q5YsvNfDPTfw47FDth4vV3LR4qUrnxBDULqtTSEO4vgGnkaL7tGUJ0eAJF4I66xDpdyNGSuga34J2Rpgj/p140SP31Mpj7CGeOE+FmqXJmE+H2JXBE31isItxVh7bkmzjw0MbJfJlfE7f0vbLC3wQKSVorkqxYvhqunD/wCQrA+bAOid6SiQCJFZ040HJbsxL8D/bi3YzEcNl5nU6txa0TliovGzO7zmQctgvi5xx5Ecmwsth84h2ftUvQ/TISrSOVbxZp6EuLXj/JLe5CwKx4/edvDfuN1Tb4tJaw6xE/xAVnhtrDbeEPrPC7xU3y6ig0iDxwkK3zEdedpElxE2qaemvxwfNM5kITzGo2fgQ43ZOqpit5JSAdvjR/3JVLyGJme87E6Pl9D/Nxt3BC/OQYpMTFI2RyI4HlqHz+dl8+BAR7Ebwzl5+Pxy+UqrRXP/pcp8BW5IzDAGUs2/s2SEv0WT4f49aPBozA/AAAVkklEQVQwaTlcxALlw4ixh5PXSqzZsAGrFjNw3ZSB/KLXaOoeQ0f2ZoQscECkzw/wdN2KZ1Vqs1hVcJcJ8/hxNX4cbKTDqDnpB4HHQU0+QdI2OUfjp9/WydM5yBquYbO7M9b//hq98h7cS16rMUEMCloFd3triFxWYLXaVDEoFEceDwNjzShI/hlxl6rZqIBaxE/WharcNGzxWwTRuktoY6N6GjL1lKPxzyAsFEShsLGDJSGjN2K1iZ/kJW6F2MLNNQEV79VYvMe77HQ8ePgfFBNE9ZQWJCBknh2S/9Im3Bpywx2rhvZbnuDycmt4LN2PWlbTzSV+6nrofL+/jL3MfGw7rSR+MQs8kFH0UTvQiZr43SzHxdggBPgHIDT2JB63k6SzXFNPYPTpXrgwbtinZbcyjNLffCF03ojz5Z8ggwL9dTew3ZOBY8yt8clu6F/s9XbHikNPDAyJTmSHC7Aw7p7yPxKIhSSHJxo88j1aiWM+Nqzvh/7F9IgxCPCZ8PiUp+gpQ35eCZr7VaROMYSm3Hh4CHxxlI1+BCg6ruFnewZeu/IhYddY5Bh4fwd7/ezBWOkTPxL1LyKjBB3sVDOC1gcH4W8rgn/6W6UvnaIHZfl5KGnuV0W/VGCoKRfxHgL4Hq1kFyr+irAF43sAT1jbTwWGmu/j11APkOh7uj5+0zP1tIEN4zReX8UQGm4nwkfAqa+aNGlMPQFFbxGSvBiIVx9EXm0PO1coRnshKS/E1fuqIBeaDpCiNMULjONO/GtAE6c5zcCOorcVdfUSdPQOQSpXQD5Yi7wr/6KpMx9xvpHIapRi7O1JrBJ54eAr0jnDeJnsCcZrP57pBL8debwbTgussMBmMdalPUWHDBiuTEegnS0Cf68e93McfI4DpG2hZ1Gjv6bG1pISPwOdxT001RgnWlNLlSsTEb8vkSsoelGU5AVGvBoH82rRQ6x2FKPolZSj8Op9qOMyKaH+crki92mtq4ekoxdDUjkU8kHU5l3Bv02dyI/zRWRWI6Rjb3FylQheB1+xFkjDL5PhyXhhv45gUbniDvyZ3eczD1oE8fNIfqV5wCp6n2C/tz3s7Ozhl/wI7WMfUHhiP7KrRtB89wSSkpKw/0gmHjXrrO7raPyGX/8GX8YBsbna0bjkjX9itcAN+4vHlIRHOB7C/asQPw3pIy+2SuLn/r0IwZ5eiPBSfsKXCOGiiupp8GW67jKSRPMQcfCxhvi5Wbtga8QGJG3YgKQIPwQaIn4k0TOZwIkTu2AlTtVqr16Pa/x0By4J752GUEcRlu59iK4PudjqYgvvHddRxzVDYi+bDvGToenJdeTce4ry+o/oV5spyiXIDBFp8vi139uHCIZB5L4CDLwrwBkfBku94/GsXMISHKWP3/fwdfLUYBrhuQQ+36mieuqSEZIPDX3I37YIzMp0cGGZDvEjcMiIpk0XTvLbkKmnofPIAkXxfrgx7ohKTkKElx2shS4Ijj+Ffyo6Mdb1FyIYETbn6nTESC+ebRHDzWMfalUaPi3i1/YfKpJ8sNTKA+l31EFWdMiW5BkbOEc7j18rWlOXw00UiYfvdM7XxdXgbwmajgdg6bzFSM35T6UVa8On9EB42G3GM47JM3fsyx4nI0Ll60c0foT4nXnQotSaqX3gZHW4sn0D0p4ZQFyH+OHzAyQsIb5YpZpnEQv/yH+4ut0X9tY2ENqKwCwQwCUsDU85UQKlVcexjLFH1A11vkdOxyl0iB/nL3ZXWoVjvkqnf92/6G/jEOAz4fEpUdGZi61ODKysGNg5OmOJWAAbW09EnylVhWsnpShz/XmRUO62DnD8wQ5CsS92ZP+NJA9b7Tx+3q5IvFmII2uWQEjOdbCFjbUYfon/oFE9NBWdyN3qBMbKCoydI5yXiCGwsYVn9BmUKmPE43PFGYQ7M1ggFMPJSQyReAUSc9/gerTYxMRPAL/9mUhR11csgrVufQ0QPxaVhlzsDybtWABGJASzgESwdkRAmo5cyZuRGcxgUdxd6HCxqbtI3oiz67yxL/8W4r3jcKerEsdWuGHXg27UnY1G1Nm7yAwTQ/xTtsYXXS7JQvhCBt77n3CCtIyhPNUHjN0K7LnTxD6XpZJcbP+RELwzqFH3DVsjBXqK9mEpw8At5i/UGfBXpMRviq7jMcYtVq5MQvy+XK4w0oDc/cFwIoHyGBGEDNF4M3AMSEMpV2M+HbmCHI1n18F7Xz5uxXsj7k4XKo+tgNuuB+iuO4voqLO4mxkGsfgnZLeqrF3kEmSFLwTjvR9PNNGPqFxNIQlm/ZvPPGhZxE+1srEwMB3lVVcQ7WoLR78o7P7tJP44k4nMTOXn7JkM/PF7OtIzH2Lc8mJc4zfW+wK/+onALD+Kct3Vub48xJL8Zj8GwN+drKAeQ/mHGjwvfoWHaWthS6KPsQ91ZXAXgbMfQlV5wUJXLAEjcMFKtT9b6DI4cqJ6KnvTCI2fFunjED9rHxw6lIJLKcrPhSgveGiInwSSy/twPGE3/tx3AOeTd+PQCjFcv3PE0VsNGuI3cVTPj0BDER6dz8K9x8UoKX2GG7uXgWHW4eIH7bBlesRPMYTWkptIi10BMeOAgAN3IVE9dAbKMxC+hIHIbR12n85B0bsudNY8R/Grh0hbawvb2Dyl8zMb3EUAZz91brVQrFjCQOCychznZY56/lZakqImfideoORCLJbbCBG2/Rq62MAgHyEvu4zDLlZwWxSEzGulGH77B+LmWyNy0yENppcORSHCapz4yV5expkdCTidtA9//vorUhNC4GRjDfd9z8b98Kap8dNqg+4P3sRPGf3WxsoK1gu9EX3kOl63c578I4+QuMQaC/22ITU9HaePH0dmSgoyf1mPtdbfw39rDoZVBExD/JpqUHEoAMu/t0Nc2hOMcAla2zt8eKjMB1mZk4LtIits+72KY7ZYh3thDHwiLqKfex3PfdnLk9gq+A4rws9r+o8QvOHrMfD7XoTIqD04f/Cg9udAApK8BHC134hHdR9ZU8+YeTYI9FmFWH9/RK1ereWHyub0U8nw+H4wvB0YNhG8sitG8DrFG4zzLjzQDnxIeh0DrW9R8uIVKhq7tbTexPSz62okROLNuMUhg7rdO+FvSvwmhMbYP/hMeLzLlPahteY1Xjx9iuLXtfg4aNisV9YnQVXJcxSX1uLjkPbzU/9eo+iqL8eL4hLUfBzSMzsEpOhrrcHrF0/xtPg1aj8O6pt2jXSgrvQ5nr2sgkStkdS/0ZcfUQWh8D5SDimmqu9Et5Git7kKJc+forj0LSS96hU7zvl9uYhZ6IAteVzbas7/k+yOVR7FcsdY5HY1ITPkB2y61YX6M2FYcfg5hhQDKDsRCHv7YPxZx3kuYgzVp4Nga70YP12oHg/E8bkWxWXKqNuD1dnY4imEyHsPColVgN42jJoLEVhsvQBi/1QU6z0n9C74Jg6YVK74jHGyQErlyuDYkfY2s8+bp8WleCvp1ZmLyHr1l8sVxipxdLkjG/SvKTMEP2y6ha76MwhbcRjPhxQYKDuBQHt7BP9Zp7U4OlZ9GkG21lj80wVUqxdEqFwZ7L+ZOMhHXi2K+Cm687DFeQVSy5SjR979FncykhG/KRyhQQFY5bcCy5f5wtfXF8uW+yHk1yJNQt1xH78yNP0dCQc7fxx/raP1YFGX4sPTizh68ACSf8tAwbvPrJ3/UpJbz0oIz4R8KIN3KYmfOCQZl7OykGXoc2kP/IXj6RyUncqP+PnG5WHUwMtx74UohHgm4JUmR99HjN7YgVDnKPxbqySHn85GItjBAavYzyIE/RiIlBP32EiU7Evzi1w8+PsBPqmIED62YKiuFkPqMjvv49CPZEWbtJnYc9vBZ++9cdM11ejUIn4kiWiCG2yshHAJScT55x+0hJ9cMtb2BBnbA7DY1gtJjztQ9utSEIJiJfREQn6H8kWHJX5ihCRfNoxpVhYu7fGHUCfQhqpKKoiJxk8I94AALGYc8fOhu5y2KrVOstfZOOrtjPiTzzBacwHJzp44dYvjg9Z0A8fdnJF8sVapFaw6i31OSkyXOjrC9ccVCEu8jHLVarv6/tPV+KnL0fvmTfwAaeN1JO06gxftBl6oIEPbw1PYGbEWGwP8scnPD5v8VmLzqkAkxp3CG01kWDKulKaelf89wOUAV2xNuauKqsnR3LUW45LvPE16EbeFa5DzQplwndXANefi6A8CJJ6r4QTJ4VxvYIxzNXfyqgJcjdmGvFecMsk1rRV4kRKDhMAAbPH3ZwkdIXXsJyAQOzbsxM27b9l7sho/KzG27jisIfZnT5xARkbGJJ9j2ORFzOjeaLpC0XEbW53sEJzxTn9C1ZyluzOKoj0ucEl8yHkW6Z6j/1vWVIg/jqYi9cguBDtSjZ8+QsYf4TPhGV/qHLtCTfx+I8TPfBtrtSCOwT8aLQHfe43g5WEfeCU9wRBkaLp/GTcrPkGhIFlwAXnzJYQvdERkVoN+/cfqcSXSDzFX3xuwuhjB28wwePgfQGGbtuWLds0+o/bqXiSeqxonj9onfHO/qFyZoEstXq6AkZeH4eOVhCdDgKzpPi7frMAnIldKwcKl8IVwjMwCyT6lvY2h/kok/GKu4r2WlpycReVKGyvz/+Ijr1+Z+MnR/OQqbpa0q1Y2FehqbWX9mIyHR4GB9kZIPo2w4Wbf13fqr5ZOVKh8CD0dHejqG9G6ZqizGU2d6iUMAxcrBtHeKEGXjlZxsPousvOrNMkuta8cQUNOBvJyX2vyzHFfhM2+rzLvJFqMkf4utLe24kP3sIHVZ0DRVYbc7IeoVy2cyjveoLiqU4/wabcPkA0Ps+fIh3rQ0dGFvhHu6ukQOpubMDms7WiUdGmiWOqWDwyg+m428sobUFfditGuLo4GikM6WpohnYJ4GMSbJOmeaBusxt3sfFT1GV7dH6m/j6zrxWjjNnmisrSOS9Hbphq/WseN/EHMHElSbxIRcqq2v6+CpKRKqeFraZkQK+n7GnRXVeFTVS2GWzn4kvKbSlCR/Q+aVLkPp7znVHX6kv/rH+NR+nlUvR3362TbP2kkzCHUFWYjt4xrnqlA58MT2J5wHmW8/Y4GUXPvKoqaJntZ1O9DWU02EmM2YdMm8onBnr/r9E+iR4xCgM+EZ1SBc/HkGXpBHZWUoPD5e6MWS8a7YxQjei+Y6n9l6HjfOMHcS1RJUpUPpfp87rcUY3ovtdz/5+Y+lSsT9PuskCtgdGLBgqzjPRoneO8BZJBOOAVSuTLBCOJdBB95/crEj3dbvp0T9cw7dV6kv+TFl+81GtL37cDJtoREJ+WLAZ/zSETG2bgRwkqSe/Np41w4h/hqqv39ZmN/0jobhQCfCc+oAufiyTP0gjoXoZ2tbaZyZYKeo3JlAhBpEXwQ4COvlPjxQdJU51DSZyoktcsxNdkhSbJn00ZC/xOSMxfInLFt/FYXO2bT+JyhuvKZ8GaoKvQ2FIFvBgEqV99MV9KGzAEE+MgrJX4zNRAo6TMf0nzMGo0hDJOZepqvFcaXTEwZv+a4MgbTr3UuGRuTmnwaDzu9wjIR4DPhWWbNaa0oApaLAJUry+0bWjOKgC4CfOSVEj9d1Mzx+2u+nM8FjYepScVn3g5e5hgtU5dpjB+fqbGZjeX19EyNKT1j1iPAZ8Kb9Y2kDaAIzDACVK5mGHB6O4rANBDgI6+U+E0DYF6XUtLHC6YvPomQIFOTERIcxVI3QkpNbdpqavwssTwpjdpgqUPaVPXiM+GZ6l60HIrAXEGAytVc6Wnazm8BAT7ySomfOXuakj5zojtetqlNPdkE7uPFW8Qe9eObHsEnski3bxoBPhPeNw0AbRxFwAwIULkyA6i0SIqAmRDgI6+U+JkJ/K/qezUXzDu5/WYODZjc6HwM3BqZbp/68U2P8HG1j2PchM6m6yJakmUgwGfCs4ya0lpQBGYPAlSuZk9f0ZpSBPjIKyV+5hgnVNNnDlQnLtMcES2/tp8fyZpKCLyptZlcIjTX9ufagsjEEvNN/sNnwvsmG04bRREwIwJUrswILi2aImBiBPjIKyV+JgadavpMDSiP8j59Mp1WSE2GSG7Ar5UDbnCQ+vGp+8GU30QzTLdvFgE+E94323jaMIqAmRCgcmUmYGmxFAEzIMBHXinxMyXwVNNnSjT5l0U0OaYkCOqyBgb418EUZ5LE8aZORq9uy0x9Ew0lkQOiMSWmlcRklmgvyTf5kCArhNgSsv41tJkEY7p9kwjwmfC+yYbTRlEEzIgAlSszgkuLpgiYGAE+8kqJn6lAp6TPVEgaXw7Ju2cOYkOIyUzkgCNkyBxaS3NgMlGZhLASQkdIHt9Nbc46UZnmON7by7d29LxZhgCfCW+WNYlWlyLw1RGgcvXVu4BWgCLAGwE+8jot4qe+wVz/bnnzFpKSsq/yaX5bi7mOf9P7JrNh31JWiabmVvNg3NiCljfVZqv7TIzJlsoqNNU3Tg+fhma0VFTNDA6l5WhqaZtefSUf6PUUAzoG6BigY4COAToG6BiwyDEwGVOkxG+ag5aSPgt4CW5pMytpaCl/g6YmiemEu0mClqoaSErLzVpvcxI/QtSmTfh0ZI/FZAYWUExd7zm/8KLTjxQPC3gm0j4x3fOaYkmxpGOAjoFZNgZMTvwmK3BO/UfNOy2nu7u6zGPuqTY3JIFBhoen117i4zbbI3USnElOQXNt5vLXVPcj+Z5p301zYUXLpQhQBCgCFAGKAEWAImAEAl+k8TOi/G/3VEr6LKtviX8Z9+XeXPskdYQxBJBEBiXnz3YfPtLumQqM0tNj3r4kfUE3igBFgCJAEaAIUAQoAnMMAUr8vqTDKen7EtTMew0JkGIusmeoXBLMhGinSGAZogEj9ydRLMk+iWhJgoiYI7+gobqY85ixRNcUvUzIsjmjm5KgPXSjCFAEKAIUAYoARYAiMMcQoMTP2A6npM9YxGbufHOSBXOSK0ssm5i2fs0k9kS7aE5ciNkt3SgCFAGKAEWAIkARoAjMIQQo8TOmsynpMwatmT93psw9zUlILKFsosk0Ji2DuXranCafM2W2ai5saLkUAYoARYAiQBGgCFAEjESAEj++gFHSxxepr3ceIStfIym4JZA1U9SBBG4hJquWshGtnCnaZagMYqJLN4oARYAiQBGgCFAEKAJzCAFK/Ph0NiV9fFCyjHNmIiqkISIxm48RE1ljAtbMZE+bKyjO1zRjnUn86L0oAhQBigBFgCJAEaAIqBCgxG+qoUBJ31QIWdb/Mpn5tESzmdwZqjvRjvb3AySYiqVuhKAZqvt0j9GUDpba47ReFAGKAEWAIkARoAiYCQFK/CYDlpK+ydCx3P+o1m9qskTGNiHJlr6Zy9yTEF66UQQoAhQBigBFgCJAEZhDCPwfNZcQmuOtmXgAAAAASUVORK5CYII=)

subsample参数通常是在样本量本身很大的时候来调整和使用

波士顿数据集的规模太小，我们的模型现在正处于样本量过少并且过拟合的状态，所以现在使用subsample参数模型效果不会有效提高。

**正常来说样本量越大，模型才不容易过拟合，现在展现出来的效果，是由于我们的样本量太小造成的一个巧合**

---

调参代码：

```python
import numpy as np
axisx = np.linspace(0,1,20)
rs = []
for i in axisx:
    reg = XGBR(n_estimators=180,subsample=i,random_state=420)
    rs.append(CVS(reg,Xtrain,Ytrain,cv=cv).mean())
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="green",label="XGB")
plt.legend()
plt.show()    
```

 ![img](data:image/png;base64,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pxLczASc8FpHT02artOzf7SAXtCTAdBCrtTptlprPvKRj+Qzn/lMNm/enN///d/Pxz/+8Xz+85/PH/3RH+UrX/nKbB+gE93l6/bbb89ll122XGVDXzk0fSgHDh2YvTR1rikUr/1r/ErYf6gemg1NTjak6XbQMjo0mlUjqzI2PJZVw6uyamRVVg0fWT+yvGpkVdauWnvs/iPrM8szbzBOJow5lfWZNzB+eQJYmUophwOUDGc0o22XA0tq5qoSv7fQi3oyAGrDn//5n2fz5s2z075++7d/O7fddlvuuOOOfPWrX83v/d7v5WMf+1jOP//8rF27Nn/wB38w+9iZHkDT09PZuHFjbrvttpa+CuiOmek9rxx8Jfum9uWVqVdOuPzK1JH1OZZPuO814yzX5fDdVFKO+Wv+a//iPzw0nLHhsTlDlzNOO+OYYOWY9eOENHOFMic6dmx4zC8jAAAwoMrRczCXy/j4eN21a9ertu3evTtXXXXVsteynAbha2RuM40ej3eJ8on2HX11yWv37T+0/7jhynGDlnlCmJnlE81hns/Y8FjWjKzJmtE1WTOyJqtHVs+5vGZ0TVYPv3rfqpFVrwpS5gpXFhK4LNd+l38DAAC9opRyV611fK59rgCiL72w/4Xs3rs7uyd354HJB/LIs4/k4PTBecOWUw1i5tvXVvPDknLCEOac1efkorUXHV4fOU5AM7PvBMtHP271yGpXmQAAAPQYARAr2jOvPDMb8jww+UAe2PtAdk/uTudnndljxobH8rpzXpc1I2uOab45sz46NJqhkePfbnTOJp4LOeZkxjuFY+e70mZseMwVKgAAAAiA6H211jz10lM/D3kmH8juvYdDnydfenL2uNNGT8tV66/K2y99e7au35qtGw5/bDlnS0aG/KgDAAAwuHrqXXGttW+vVmij19JKU2vNnp/teVXAM/Px7L5nZ487c9WZ2bpha95z+XuydcPWXLXhqmzdsDWbz9ps6hEAAADMoWcCoNWrV+fpp5/OunXr+i4EqrXm6aefzurVq9supSccmj6UHz/342NCnt17d+fFAy/OHrf+tPXZumFrPnT1h2ZDnq0btubCMy7su58RAAAAWEo9EwBt3Lgxe/bsyeTkZNulLInVq1dn48aNbZexrA4eOphHnn3kmGlbD+59MPum9s0ed9Hai7J1w9b85rW/ORvyXLX+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) 

细化：

```python
axisx = np.linspace(0.05,1,20)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=180,subsample=i,random_state=420)
    cvresult = CVS(reg,Xtrain,Ytrain,cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean())**2+cvresult.var())
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
rs = np.array(rs)
var = np.array(var)
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="black",label="XGB")
plt.plot(axisx,rs+var,c="red",linestyle='-.')
plt.plot(axisx,rs-var,c="red",linestyle='-.')
plt.legend()
plt.show()
```

 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABIEAAAEvCAYAAADSGNH4AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABxUklEQVR4nO3dd3hU5dbG4d8OSeghQOi9hE7oIEhRAUUUEStib+hBj91jbx/2iigWsKCHo6KIFBuKhSK9hg6hJvRACAmE1P39sZgUSCBl0p/7uubKlL33vBMyIfuZ9a7XcV0XEREREREREREp2XwKewAiIiIiIiIiIpL/FAKJiIiIiIiIiJQCCoFEREREREREREoBhUAiIiIiIiIiIqWAQiARERERERERkVJAIZCIiIiIiIiISCngW1hPHBQU5DZu3Liwnl5EREREREREpMRZvnx5pOu6NTJ7rNBCoMaNG7Ns2bLCenoRERERERERkRLHcZydWT2m6WAiIiIiIiIiIqWAQiARERERERERkVJAIZCIiIiIiIiISClQaD2BMpOYmEhERAQnTpwo7KHkm3LlylG/fn38/PwKeygiIiIiIiIiUooUqRAoIiKCypUr07hxYxzHKezheJ3ruhw6dIiIiAiaNGlS2MMRERERERERkVKkSE0HO3HiBNWrVy+RARCA4zhUr169RFc6iYiIiIiIiEjRVKRCIKDEBkAeJf31iYiIiIiIiEjRVORCoMIUHh5OkyZNOHz4MABRUVE0adKEnTt3smXLFi699FKaNWtGly5dOP/885k7dy4AEydOpEaNGnTs2JG2bdty1VVXcfz48cJ8KSIiIiIiIiIiGSgESqdBgwb861//4vHHHwfg8ccfZ+TIkdSqVYtLLrmEkSNHsnXrVpYvX857773Htm3bUve99tprWbVqFevWrcPf35/JkycX1ssQERERERERETlNkWoMXRQ8+OCDdOnShTFjxjB//nzee+89/vvf/9KzZ08uu+yy1O3atWtHu3btTts/KSmJY8eOUbVq1YIctoiIiIiIiIjk1PHjsHo11KoFTZsW9mjynSqBTuHn58cbb7zBgw8+yJgxY/D392fdunV07tz5jPtNnjyZjh07Uq9ePQ4fPsyQIUMKaMQiIiIiIiIiclaJifDPP+CZ1bN4MQQEQK9e8OWXhTu2AlJkK4EeeOABVq1a5dVj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69pqaH/+CaNG2TSvFSts2ff0265dC48+CldfDV9/DZmFXyIiIiIiIiLFmOM4y13X7ZrZY9mqBJJi4NlnYfdu+O67MwZAAL/++ivLly/n008/PT0AiomB48etwufYMbsvORkSEy0QSkiwi+d6rVq2TZs2FsKEhNjtbt3gf//LuG36656vDU72HA8MhC5dMlbnOI6NISoq62PceaeFQPPmWcBz++0WAk2eDK+/fvqLv/BCmDRJAZCIiIiIiIiUOqoEKineeQf27IE3zjwjz3Vdevbsyb59+9iyZQt+pwZG770HDz8MmzdD48b5N15vS0qCEyegYkULjw4fhsjIjMFRSoqFW2cJyURERERERESKK1UClQYPPpitzWbPns3ixYv56KOPTg+AwKZyxccXrwAIwNc3YxVRtWp2ERERERERERFAlUDF36ef2vSn4cPP2mzZdV369u3Ljh07CAsLo2zZsgU0SBEREREREREpCGeqBNL62MWZ61p/m0mTsrX5nDlzmD9/Po8//njmAdD778OcOV4epIiIiIiIiIgUBaoEKu6Sk2159qpVz7pp//792bBhA9u2baNcuXIZHzx0COrVs8bK48bl02BFREREREREJD+pEqgkWrrUGh+XKZOtAGj+/Pn8+eef/Oc//zk9AAL44gvrBXTXXfkwWBEREREREREpbKoEKo6OHrUl2Vu1gtmzs7XLRRddxKpVq9i+fTsVKlTI+KDr2rGqVYOFC/NhwCIiIiIiIiJSELQ6WEnzzDO2HPz332dr88WLF/Pbb7/x+uuvnx4AAfz9ty0JP3GiV4cpIiIiIiIiIkWHpoMVN0uXwnvvwahR0KNHtnYZPXo01atX51//+lfmG3z8MQQGwjXXeG+cIiIiIiIiIlKkKAQqTpKSYORIqF0bXnopW7ssX76cn376iYceeohKlSqdvsGBAzB1Ktx0E5Qv7+UBi4iIiIiIiEhRoelgxcnYsbBqFXz3HVSpkq1dXnzxRQIDA7n33nsz3+DzzyExUQ2hRUREREREREo4VQIVFzt3Wi+gSy6BK6/M1i6rV69m2rRpPPjggwQEBJy+QUoKjB8PffpYo2kRERERERERKbFUCVQcuC54KnnGjQPHydZuL774IgEBAdx3332Zb5CQALffDu3be2mgIiIiIiIiIlJUKQQqDkJD4aef4I03oFGjbO2ybt06vv/+e5566ikCAwMz36hcOXjySe+NU0RERERERESKLIVAxUGHDrBsGYSEZHuXl156iYoVK/LAAw9kvsG+fbY0/LBhULasV4YpIiIiIiIiIkWXegIVdTt32tfOncE3e5ndpk2b+Oabb7jnnnuoXr165ht98w1cd13a8UVERERERESkRFMIVJQtXQrNmsHkyTna7eWXX6Z8+fI89NBDWW90332waBG0aJHHQYqIiIiIiIhIcaAQqChr3dp69lx8cbZ3CQsL43//+x//+te/qFmzZtYb+vhAjx5eGKSIiIiIiIiIFAcKgYoq14VKleD//g8yW949C6+88gp+fn488sgjWW90++3w6qteGKSIiIiIiIiIFBcKgYqiHTuga1dYuTKHu+3gyy+/ZOTIkdSuXTvzjSIiYOJEiI7O8zBFREREREREpPhQCFTUuC7ccw9s2gRZNXXOwquvvoqPjw//+c9/st7ok0/sOe68M48DFREREREREZHiREvEFzXffw8//wxvvw0NG2Z7t/DwcD777DPuuOMO6tWrl/lGSUkWAl14ITRt6qUBi4iIiIiIiEhxoEqgoiQ62lbt6tQJ/v3vHO36+uuvA/D4449nvdHPP8Pu3XDXXXkZpYiIiIiIiIgUQ6oEKkqefBL274cZM8A3+/80e/bsYcKECdx88800PFP10EcfQd26cOmlXhisiIiIiIiIiBQnqgQqKhYtgg8/hHvvtabQOfDGG2+QlJTEE088kfVGO3bAr7/aymB+fnkbq4iIiIiIiIgUOwqBioLERJuiVbcujB6do13379/Pxx9/zI033kjTM/X5+eQTcBy44448DlZEREREREREiiNNBysKxoyB0FD44QcICMjRrm+99Rbx8fE8+eSTWW+UlASffgqDB+eo2bSIiIiIiIiIlBwKgYqC4GCrBLr88hztFhkZyQcffMB1111HcHBw1hv6+sLMmTnqMyQiIiIiIiIiJYtSgaLg8stzHAABvPPOOxw/fpynnnrq7BvnsM+QiIiIiIiIiJQs6glUmKZOhVdftZ5AOXT48GHee+89rr76alq3bp31hlu3wm23wa5deRioiIiIiIiIiBR3CoEK0+zZ8N131rA5h8aOHUtMTAxPP/30mTdctcrCJk0FExERERERESnVHNd1C+WJu3bt6i5btqxQnrtIOXo0x82go6OjadSoEf379+f7778/+w7x8VC2bC4HKCIiIiIiIiLFheM4y13XzbQnjCqBCsPq1bBpk13PYQAE8N577xEdHX32KqDoaHBdBUAiIiIiIiIiosbQBS4xEW68EeLiYONGKFMmR7vHxMTwzjvvMGTIEDp16nTmjS+/HKpUgWnTcj1cERERERERESkZVAlU0N55B9asgbfeynEABPDBBx9w+PBhnnnmmTNvuHEj/P03nHNO7sYpIiIiIiIiIiWKQqCCtH07PP88DBsGl12W492PHTvGm2++yaBBg+jWrduZNx4/3ppB33pr7sYqIiIiIiIiIiVKtkIgx3EGOY6zyXGcMMdxHs/k8fMcx4l2HGfVycuz3h9qMee6MGqUVf+MHZurQ3z88cdERkby7LNn+faeOAFffAFXXAG1auXquURERERERESkZDlrTyDHccoA44CBQASw1HGcGa7rrj9l03mu616aD2MsGb79Fn79Fd59F+rXz/HucXFxvP766/Tv35+ePXueeeMpU+DwYbjrrlwOVkRERERERERKmuw0hu4OhLmuuw3AcZxvgKHAqSGQZOXIEbj/fujaFe65J1eHmDBhAvv37+fbb789+8YffQTBwXD++bl6LhEREREREREpebIzHaweEJ7udsTJ+07V03Gc1Y7j/OI4TtvMDuQ4zkjHcZY5jrPs4MGDuRhuMfXEE3DwoPXpyUUz6BMnTvDaa6/Rr18/+vbte+aN162Df/6xKiDHyeWARURERERERKSkyU4lUGZJgnvK7RVAI9d1Yx3HGQxMA4JP28l1xwPjAbp27XrqMUqmAwdg0iR44AE425LuWfj888/Zs2cPX3755dk3/vhj8PeHm2/O1XOJiIiIiIiISMmUnRAoAmiQ7nZ9YE/6DVzXPZru+s+O43zgOE6Q67qR3hlmMVazpi0JHxSUq90TEhJ45ZVX6NWrFxdccMGZN05Jgd9+g6uuyvXziYiIiIiIiEjJlJ0QaCkQ7DhOE2A3MBwYkX4Dx3FqA/td13Udx+mOTTM75O3BFjsbN0LLltC4ca4P8eWXXxIeHs6ECRNwzja9y8cHQkMhOjrXzyciIiIiIiIiJdNZewK5rpsE3AvMAjYA37quu85xnLsdx7n75GZXAWsdx1kNjAWGu65bOqZ7ZWXHDpv+9coruT5EYmIiL7/8Mt26dePCCy88+w7JyTYVrEaNXD+niIiIiIiIiJRM2akEwnXdn4GfT7nvo3TX3wfe9+7QirkGDeCNN2DYsFwf4n//+x/bt29n7NixZ68CWrkShg615eG7d8/1c4qIiIiIiIhIyZStEEhyKCXFVgG7995cHyIpKYmXX36ZTp06cckll2TvOdu3t6XhRUREREREREROkZ0l4iUnoqIgJAR++ilPh5k8eTJbtmzhmWeeOXsVEECXLvacVavm6XlFREREREREpGRSCORtjz9uDaHr1cv1IZKTk3nxxRdp3749Q4cOPfsOy5fD7t25fj4RERERERERKfk0Hcyb5s+H8ePh4YehY8dcH+b7779n48aNTJ48GR+fs+R0rgt33gmOY2GQiIiIiIiIiEgmnMJaxKtr167usmXLCuW580VCgq0GFhsL69ZBpUq5OkxKSgodOnQgOTmZNWvWUKZMmTPvsHSpNYIeNw5GjcrVc4qIiIiIiIhIyeA4znLXdbtm9pgqgbzlzTdh/XqYOTPXARDA9OnTWbt2Lf/73//OHgABfPwxVKwIN9yQ6+cUERERERERkZJPlUDeEBZmK3Ndeil8912uD+O6Lp07d+bYsWNs2LDh7CFQdDTUrQsjRsCECbl+XhEREREREREpGVQJlJ9c16Zh+fnBu+/m6VA//vgjq1atYuLEidmrApo0CY4fh7vuytPzioiIiIiIiEjJpxAor77+Gn7/Hd5/36pycsl1XUaPHk3Tpk0ZMWJEdnawqWBdukDXTAM+EREREREREZFUCoHyKjgYbrsN7r47T4eZNWsWS5cuZcKECfj5+Z19h4ULYc0aW41MREREREREROQs1BOoCHBdl3PPPZfdu3ezZcsW/P39z77TzTfDDz/Anj15akQtIiIiIiIiIiWHegIVcX/++ScLFy7kww8/zF4A5Lrg62sVSAqARERERERERCQbVAlUBPTr14+tW7eydetWypYtm/0dXRccJ/8GJiIiIiIiIiLFypkqgXwKejCS0Zw5c5g7dy6PPfZY9gIg14WNG+26AiARERERERERySaFQIVs9OjR1KpVizvuuCN7O8ybB61bw8yZ+TswERERERERESlR1BOoEP3zzz/88ccfvPXWW5QvXz57O7VtC2+9Bf375+/gRERERERERKREUU+gQjRo0CBWrFjB9u3bqVixYmEPR0RERERERESKOfUEKoKWLFnCrFmzePjhh7MfAE2eDF9+aX2BRERERERERERyQCFQIRk9ejTVqlVj1KhR2dshJQWefho++UQNoUVEREREREQkxxQCFYIVK1bw448/8tBDD1G5cuXs7fTXXxAWBnfdlb+DExEREREREZESSSFQIXjxxRcJDAzk3nvvzf5OH30E1avDlVfm38BEREREREREpMRSCFTAQkND+eGHH7j//vupUqVK9nbatw+mTYNbboFy5fJzeCIiIiIiIiJSQikEKmAvvvgilStX5v7778/+Tp99BklJMHJk/g1MREREREREREo0hUAFaP369UyZMoV///vfVK1aNXs7paTAhAlw/vnQokX+DlBERERERERESiyFQAXopZdeokKFCjz44IPZ3+m332DHDrj77nwbl4iIiIiIiIiUfAqBCsjmzZv55ptvGDVqFEFBQdnf8aOPoGZNuPzyfBubiIiIiIiIiJR8voU9gNLi5ZdfpmzZsjz88MM52/Gqq2DwYPD3z5+BiYiIiIiIiEipoBCoAGzbto1JkyZx3333UatWrZztfMMN+TMoERERERERESlVNB2sALzyyiv4+vry6KOPZn+npCT44AM4dCj/BiYiIiIiIiIipYZCoHy2c+dOJk6cyJ133kmdOnWyv+M//8A998CcOfk3OBEREREREREpNTQdLJ+9+uqr+Pj48Nhjj+Vsx379YPVqaN06fwYmIiIiIiIiIqWKQqB8FBERwWeffcatt95K/fr1c36AkBDvD0pERERERERESiVNB8tHr7/+OikpKTz++OM52/Gll+CmmyA5OX8GJiIiIiIiIiKljkKgfLJ3717Gjx/PzTffTOPGjbO/Y2IijBtnDaHLlMm38YmIiIiIiIhI6aIQKJ+8+eabJCUl8cQTT+Rsxx9/hL174a678mdgIiIiIiIiIlIqKQTKBwcOHODDDz/k+uuvp1mzZjnb+eOPoX59GDw4fwYnIiIiIiIiIqWSQqB88NZbbxEfH8+TTz6Zsx23bYNZs+COO8BXPbtFRERERERExHuyFQI5jjPIcZxNjuOEOY6TZZdjx3G6OY6T7DjOVd4bYvESGRnJuHHjuPbaa2nZsmXOdp4wAXx84Pbb82dwIiIiIiIiIlJqnTUEchynDDAOuBhoA1znOE6bLLZ7DZjl7UEWJ2PGjOH48eM89dRTOdsxIQE++wyGDLHpYCIiIiIiIiIiXpSdSqDuQJjruttc100AvgGGZrLdv4HvgQNeHF+xEhUVxdixY7nyyitp27ZtznaeNg0OHFBDaBERERERERHJF9kJgeoB4eluR5y8L5XjOPWAYcBH3hta8TN27FhiYmJ4+umnc77zhAnQqBFceKH3ByYiIiIiIiIipV52ug87mdznnnJ7DPCY67rJjpPZ5icP5DgjgZEADRs2zOYQi4fo6GjGjBnD5ZdfTocOHXJ+gM8+g+3boUwZ7w9OREREREREREq97IRAEUCDdLfrA3tO2aYr8M3JACgIGOw4TpLrutPSb+S67nhgPEDXrl1PDZKKtffff58jR47krgoIoEEDu4iIiIiIiIiI5IPsTAdbCgQ7jtPEcRx/YDgwI/0Grus2cV23seu6jYEpwKhTA6CSLCYmhrfffptLLrmELl265Gzn+Hi45hpYtCh/BiciIiIiIiIiQjZCINd1k4B7sVW/NgDfuq67znGcux3HuTu/B1gcfPjhhxw+fJhnnnkm5ztv2QILFkBsrPcHJiIiIiIiIiJykuO6hTMrq2vXru6yZcsK5bm96dixYzRp0oROnToxa9as3B0kKQl8fOwiIiIiIiIiIpJLjuMsd123a2aPKXXIo/Hjx3Pw4EGeffbZnO8cHQ2JieDrqwBIRERERERERPKVkoc8SElJ4d133+WCCy7g3HPPzfkBnn0WgoMhIcH7gxMRERERERERSSc7q4NJFnx8fJg3bx6xuennc/w4fPklDBoE/v7eH5yIiIiIiIiISDoKgfKoQW6Xdf/2WzhyBO5Wb20RERERERERyX+aDlZYPv4YWrWCvn0LeyQiIiIiIiIiUgooBCoMq1fDokVw113gOIU9GhEREREREREpBRQCFYaPP4ayZeGmmwp7JCIiIiIiIiJSSigEKmixsTBpElxzDVSrVtijEREREREREZFSQiFQQfvmG4iJUUNoERERERERESlQCoEK2l9/Qbt20LNnYY9EREREREREREoRLRFf0CZNgshINYQWERERERERkQKlSqCClJRk4U+NGoU9EhEREREREREpZRQCFZSjR6FhQ6sEEhEREREREREpYAqBCkpsLFx4IbRqVdgjEREREREREZFSSD2BCkrdujBxYmGPQkRERERERERKKVUCFYQtW2DlysIehYiIiIiIiIiUYgqBCsLLL0PfvnDsWGGPRERERERERERKKYVA+S0qCiZPhuuvh4oVC3s0IiIiIiIiIlJKKQTKb//9L8TFwV13FfZIRERERERERKQUUwiUn1wXPv4YuneHTp0KezQiIiIiIiIiUoppdbD8NH8+rF8Pn35a2CMRERERERERkVJOlUD56eOPISAArr22sEciIiIiIiIiIqWcQqD8EhkJU6bATTepIbSIiIiIiIiIFDqFQPnliy8gPl4NoUVERERERESkSFAIlF98fOCyy6Bdu8IeiYiIiIiIiIiIQqB88+CDMH16YY9CRERERERERARQCJQ/QkMhJaWwRyEiIiIiIiIikkohkLcdOABdu8Lo0YU9EhERERERERGRVL6FPYASp0oVmDgRevQo7JGIiIiIiIiIiKRSCORtZcvCiBGFPQoRERERERERkQw0Hcyb5s2DV1+FY8cKeyQiIiIiIiIiIhkoBPKmd96Bt94CXxVYiYiIiIiIiEjRohDIW/bsgRkz4NZbbUqYiIiIiIiIiEgRohDIWz77DJKTYeTIwh6JiIiIiIiIiMhpFAJ5Q3IyjB8PAwZA8+aFPRoRERERERERkdMoBPKGX3+F8HC4667CHomIiIiIiIiISKYUAnnDRx9B7dowdGhhj0REREREREREJFPZCoEcxxnkOM4mx3HCHMd5PJPHhzqOE+o4zirHcZY5jtPb+0MtosLD4eef4bbbwM+vsEcjIiIiIiIiIpKps65l7jhOGWAcMBCIAJY6jjPDdd316Tb7A5jhuq7rOE4I8C3QKj8GXOR88gm4Ltx5Z2GPREREREREREQkS2cNgYDuQJjrutsAHMf5BhgKpIZAruvGptu+IuB6c5BFWv/+4O8PjRsX9khERERERERERLKUnRCoHhCe7nYE0OPUjRzHGQa8AtQELvHK6IqDvn3tIiIiIiIiIiJShGWnJ5CTyX2nVfq4rvuD67qtgMuB0ZkeyHFGnuwZtOzgwYM5GqiIiIiIiIiIiORedkKgCKBButv1gT1Zbey67lygmeM4QZk8Nt513a6u63atUaNGjgcrIiIiIiIiIiK5k50QaCkQ7DhOE8dx/IHhwIz0GziO09xxHOfk9c6AP3DI24MVEREREREREZHcOWtPINd1kxzHuReYBZQBPnNdd53jOHeffPwj4ErgJsdxEoE44FrXdUtPc2gRERERERERkSLOKayspmvXru6yZcsK5blFREREREREREoix3GWu67bNbPHsjMdTEREREREREREijmFQCIiIiIiIiIipYBCIBERERERERGRUkAhkIiIiIiIiIhIKaAQSERERERERESkFFAIJCIiIiIiIiJSChTaEvGO4xwEdhbKk4uULEFAZGEPQqSE0ftKxLv0nhLxPr2vRLyrJL2nGrmuWyOzBwotBBIR73AcZ5nrul0LexwiJYneVyLepfeUiPfpfSXiXaXlPaXpYCIiIiIiIiIipYBCIBERERERERGRUkAhkEjxN76wByBSAul9JeJdek+JeJ/eVyLeVSreU+oJJCIiIiIiIiJSCqgSSERERERERESkFFAIJFJMOI4zyHGcTY7jhDmO83gmj1/vOE7oycsCx3E6FMY4RYqLs72n0m3XzXGcZMdxrirI8YkUR9l5XzmOc57jOKscx1nnOM6cgh6jSHGTjb8BqziOM9NxnNUn31e3FsY4RYoLx3E+cxzngOM4a7N43HEcZ+zJ91yo4zidC3qM+UkhkEgx4DhOGWAccDHQBrjOcZw2p2y2Hejnum4IMJpSMqdVJDey+Z7ybPcaMKtgRyhS/GTnfeU4TiDwAXCZ67ptgasLepwixUk2/7+6B1jvum4H4DzgLcdx/At0oCLFy0Rg0BkevxgIPnkZCXxYAGMqMAqBRIqH7kCY67rbXNdNAL4BhqbfwHXdBa7rRp28uQioX8BjFClOzvqeOunfwPfAgYIcnEgxlZ331Qhgquu6uwBc19V7S+TMsvO+coHKjuM4QCXgMJBUsMMUKT5c152LvU+yMhT40jWLgEDHceoUzOjyn0IgkeKhHhCe7nbEyfuycjvwS76OSKR4O+t7ynGcesAw4KMCHJdIcZad/6taAFUdx/nbcZzljuPcVGCjEymesvO+eh9oDewB1gD3u66bUjDDEymRcnruVaz4FvYARCRbnEzuy3RpP8dxzsdCoN75OiKR4i0776kxwGOu6ybbh6sichbZeV/5Al2A/kB5YKHjOItc192c34MTKaay8766CFgFXAA0A353HGee67pH83lsIiVVts+9iiOFQCLFQwTQIN3t+tinPRk4jhMCfAJc7LruoQIam0hxlJ33VFfgm5MBUBAw2HGcJNd1pxXICEWKn+y8ryKASNd1jwHHHMeZC3QAFAKJZC4776tbgVdd13WBMMdxtgOtgCUFM0SREidb517FlaaDiRQPS4Fgx3GanGz0NxyYkX4Dx3EaAlOBG/WJqshZnfU95bpuE9d1G7uu2xiYAoxSACRyRmd9XwHTgT6O4/g6jlMB6AFsKOBxihQn2Xlf7cKq63AcpxbQEthWoKMUKVlmADedXCXsHCDadd29hT0ob1ElkEgx4LpukuM492IrFJUBPnNdd53jOHeffPwj4FmgOvDBycqFJNd1uxbWmEWKsmy+p0QkB7LzvnJdd4PjOL8CoUAK8Inrupku0Ssi2f7/ajQw0XGcNdg0lsdc140stEGLFHGO43yNraQX5DhOBPAc4Aep76mfgcFAGHAcq7YrMRyrGhQRERERERERkZJM08FEREREREREREoBhUAiIiIiIiIiIqWAQiARERERERERkVJAIZCIiIiIiIiISCmgEEhEREREREREpBRQCCQiIiIiIiIiUgooBBIRERERERERKQUUAomIiIiIiIiIlAL/D+qrmZ2NrvzhAAAAAElFTkSuQmCC) 

再细化！

```python
#细化学习曲
axisx = np.linspace(0.75,1,25)

rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=180,subsample=i,random_state=420)
    cvresult = CVS(reg,Xtrain,Ytrain,cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean())**2+cvresult.var())
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
rs = np.array(rs)
var = np.array(var)
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="black",label="XGB")
plt.plot(axisx,rs+var,c="red",linestyle='-.')
plt.plot(axisx,rs-var,c="red",linestyle='-.')
plt.legend()
plt.show()

#不要盲目找寻泛化误差可控部分的最低值，注意观察结果
plt.figure(figsize=(20,5))
plt.plot(axisx,ge,c="gray",linestyle='-.')
plt.show()


```

 ![img](data:image/png;base64,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) 

 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABJQAAAEvCAYAAAATn5tUAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAB3qklEQVR4nO39eVSUV743fH93FRQzMsosKIiATCrgjFPilMEYE2NMOmboTuzu9JSn73O63+dd633u+15nrfPc53Sf030ydTqdwcTEGBMTM5g4JGqMA4MyCDiAghYgCMg8VtV+/6DgoNFYQFG7hu9nLZZUXVdd17cSLor61d6/LaSUICIiIiIiIiIispRGdQAiIiIiIiIiInIsLCgREREREREREdGosKBERERERERERESjwoISERERERERERGNCgtKREREREREREQ0KiwoERERERERERHRqLipDmANISEhMi4uTnUMIiIiIiIiIiKnUVhY2CSlDL3VNqcoKMXFxaGgoEB1DCIiIiIiIiIipyGEqLndNk55IyIiIiIiIiKiUWFBiYiIiIiIiIiIRoUFJSIiIiIiIiIiGhUWlIiIiIiIiIiIaFRYUCIiIiIiIiIiolFhQYmIiIiIiIiIiEaFBSUiIiIiIiIiIhoVFpSIiIiIiIiIiGhUWFAiIiIiIiIiIqJRYUGJnJbBYMD58+dhNBpVRyEiIiIiIiJyKm6qAxBZm8lkQmlpKb799lu0tbXh3nvvxZw5c1THIiIiIiIiInIaLCiR05BSorKyEgcPHkRDQwMiIiIwMDCAS5cusaBEREREREREZEUsKJFTqK2txYEDB1BdXY3AwEBs2LABM2fORF1dHQIDA1XHIyIiIiIiInIqLCiRwysvL8eHH34Ib29vrFmzBnPmzIFWqwUAREVFKU5HRERERERE5HxYUCKH1NnZifb2dkRGRiIhIQHLly9HTk4OPDw8btjPaDTixIkTmDx5MqZPn64oLREREREREZFz4Spv5JB27tyJ3bt3Q0oJnU6HxYsX/6CYBAAajQYnT55EVVWVgpREREREREREzokjlMghGI1GnD59GqmpqfD09MSqVavg6ekJIcSPPk4Igeeffx46nc5GSYmIiIiIiIicHwtKZNeklCgvL8c333yDlpYWAEBWVtaoeiOxmERERERERERkXSwokd2qrq7G/v37UVdXh8mTJ2Pz5s1ISEgY9XH6+/vx8ccfIzk5GRkZGROQlIiIiIiIiMi1sKBEdqehoQEHDhxAZWUl/P39sW7dOqSnp0OjGVvLL3d3d9TX18PNzY0FJSIiIiIiIiIrYEGJ7EZbWxu+/fZbFBcXw9PTE3fddRdycnLg7u4+ruMKIRAXF4eqqipIKe/Yd4mIiIiIiIiIfpxFQz6EEKuFEOeEEJVCiD/cYrsQQvzVvL1ECDF7xLZqIUSpEKJICFEw4v5/E0KcNe+/WwgRYL4/TgjRY96/SAjxqhWeJzmAtrY2nDlzBgsWLMCvf/1rLFy4cNzFpCFxcXHo6upCU1OTVY5HRERERERE5MruOEJJCKEF8BKAuwHoAeQLIfZIKctH7LYGwHTz11wAr5j/HbJMSnnzO/n9AP4opTQIIf5fAH8E8M/mbVVSyswxPB9yMMePH0dXVxfuuusuTJkyBS+88AK8vb2tfp64uDgAg32ZQkNDrX58IiIiIiIiIldiyQilHACVUsqLUsp+ADsArLtpn3UAtslBJwAECCEifuygUsp9UkqD+eYJANGjzE4OSko5/P3169fR1NQ0fN9EFJMAICAgAP7+/qipqZmQ4xMRERERERG5EksKSlEAroy4rTffZ+k+EsA+IUShEOLZ25zjaQB7R9yeKoQ4LYQ4LIRYbEFGcgBSSpw/fx6vvPLKcGFn9erV2LRp04T3NRrqo1RdXX1DQYuIiIiIiIiIRs+Spty3eqd/8zvyH9tnoZSyTggxGcB+IcRZKeWR4QcK8X8DMADYbr6rHsAUKWWzEGIOgE+EEDOllO03nHCwOPUsAEyZMsWCp0Eq6fV6HDhwADU1NQgKCoLJZAKAMa/cNhZxcXEoKSlBU1MTp70RERERERERjYMlBSU9gJgRt6MB1Fm6j5Ry6N9GIcRuDE6hOwIAQogtAO4FsEKah41IKfsA9Jm/LxRCVAFIBFAw4viQUr4G4DUAyMrK4pATO9XU1IRvvvkGFRUV8PHxwT333INZs2ZBq9XaPAv7KBERERERERFZhyUFpXwA04UQUwHUAtgEYPNN++wB8LwQYgcGm3G3SSnrhRA+ADRSyg7z9ysB/C9gcOU4DDbhXiKl7B46kBAiFECLlNIohJiGwUbfF8f1LMnmOjo6cPjwYZw6dQru7u5YunQp5s+fD51OpyxTQEAA0tPT4e/vrywDERERERERkTO4Y0HJvArb8wC+BqAF8IaUskwIsdW8/VUAXwJYC6ASQDeAp8wPDwOw29wfxw3Ae1LKr8zbXgTggcFpcABwQkq5FUAugP8lhDAAMALYKqVsscaTJdu4ePEiduzYAaPRiOzsbOTm5sLHx0d1LAghsH79etUxiIiIiIiIiByecIYGxVlZWbKgoODOO9KEMRqNaG9vR2BgIHp7e7Fv3z4sWrQIQUFBqqP9QHd3NzQaDTw9PVVHISIiIiIiIrJbQohCKWXWrbbZriMyObX3338f77//PkwmEzw9PXH//ffbZTGpra0N//Zv/4bS0lLVUYiIiIiIiIgcliU9lIhu6dKlS4iKioJOp8OCBQtgMplgnr5ot/z9/bFy5crhBt1ERERERERENHosKNGodXV1Yffu3aiqqsLdd9+NBQsWYNq0aapjWUQIgfnz56uOQUREREREROTQOOWNRi0vLw8XL17EypUrkZOTozrOqA0MDOD8+fNob29XHYWIiIiIiIjIIbGgRKNWW1uLyZMnY/78+XBzc7xBbp2dnXj//fdx7tw51VGIiIiIiIiIHBILSjQqUkro9XpER0erjjJmAQEB8Pf3R3V1teooRERERERERA6JBSUalaamJvT19Tl0QUkIgbi4OFRXV0NKqToOERERERERkcNhQYlGRa/XA4BDF5QAIC4uDt3d3bh27ZrqKEREREREREQOhwUlGhW9Xg9PT08EBwerjjIucXFxAMBpb0RERERERERjwIISjUptbS2ioqIghFAdZVwCAgIwadIk1NTUqI5CRERERERE5HAcb4kuUmrRokXw8PBQHWPchvooXbhwAVJKhy+QEREREREREdkSRyjRqKSmpmL69OmqY1hFbGws+ygRERERERERjQELSmSx+vp6XL16VXUMqxnqo1RXV6c2CBEREREREZGDYUGJLHb48GHs2rVLdQyrCQgIwAsvvIDMzEzVUYiIiIiIiIgcCnsokcVWr16Njo4O1TGsRggBPz8/1TGIiIiIiIiIHA5HKJHFAgICEBMTozqGVV27dg0ffPAB+ygRERERERERjQILSmSRK1euIC8vDwMDA6qjWJW7uzuuXr3qVCOviIiIiIiIiCYap7yRRcrKylBYWIisrCzVUawqICAAv/nNb1THICIiIiIiInIoHKFEFtHr9YiKioJG45w/MlJKSClVxyAiIiIi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) 

```python
#看看泛化误差的情况如何
reg = XGBR(n_estimators=100
           ,subsample=0.8320775188717384
           ,random_state=420).fit(Xtrain,Ytrain)
reg.score(Xtest,Ytest)
#发现分数降低了，所以数据太少，不使用subsample```
```



#### 迭代决策树： 重要参数eta

梯度上升树并不是说只要我新建了一棵倾向于困难样本的决策树，它就能够帮我把困难样本判断正确了，就算新建了一棵树，也不一定就判断正确。

- 怎么保证每次新添加的树一定让集成学习的效果提升？

1.笨方法：枚举 --算就完了 2.随机生成固定数目的树，然后找出生成最好的那一棵，但是不一定能找出效果最好的那一棵树。

> 平衡算法表现和运算速度是机器学习的艺术，我们希望能找出一种方法，直接帮我们求解出最优的集成算法结果。求解最优结果！！！

最优化问题：

在逻辑回归当中，我们有方程：

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C68.png)

 我们的目标是求解让逻辑回归的拟合效果最优的参数组合 。我们首先找出了逻辑回归的损失函数y ，这个损失函数可以通过带入c塔 来衡量逻辑回归在训练集上的拟合效果。然后，我们利用梯度下降来迭代我们的c塔 ：![image-2.png](data:image/png;base64,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) 

我们让第k 次迭代中的c塔 减去通过步长和特征取值x 计算出来的一个量，以此来得到第k+1次迭代后的参数向量。我们可以让这个过程持续下去，直到我们找到能够让损失函数最小化的参数 c塔为止：

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C69.png)

 迭代下去，让损失函数最小。参数可以迭代，集成的树林也可以迭代。![image-4.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAUwAAABHCAYAAABs3n4oAAARyUlEQVR4Ae2dhZPcNhTG+z+VmZsyp5QypMyUJk2ZmZmSMjcpc4ppmzJTypQGygzq/DzzbrQ6yau9s2+97vdmNueVZdn+JH16pM1iTiIEhIAQEAJZCCyWVUuVhIAQEAJCwIkwNQiEgBAQApkIiDAzgVI1ISAEhIAIU2NACAgBIZCJgAgzEyhVEwJCQAiIMDUGhIAQEAKZCIgwM4FSNSEgBISACFNjQAgIASGQiYAIMxMoVRMCQkAIiDA1BoSAEBACmQiIMDOBUjUhIASEgAhTY0AICAEhkImACDMTKFUTAkJACIgwNQaEgBAQApkIiDAzgVI1ISAEhIAIU2NACAgBIZCJgAgzEyhVEwJCQAiIMDUGhIAQEAKZCIgwM4FSNSEgBISACFNjQAgIASGQiYAIMxMoVRMCQkAItJowf/zxR3fyySe7P//8s289/ccff7iTTjrJ8Sz/N/n777/dWWed5b799tusV//pp5+K/vr111+z6quSEBhrBFpLmH/99ZebOHGimzt3bgem+++/v9tggw3ctttu21FexZcffvjBzZo1a1hT77//fvEsEMj/SU488UT30EMPdbzyueee6zbeeGO32mqrOQgylJdfftkdeOCB7t9//w1P6bsQ6DsCrSXM6667zp1//vnDAGYiTpgwwR177LHDzsUK/vnnn+jE9uteddVV7sgjj3SrrLKKO+yww/xTQ8enn366u/7664e+t/3gpZdecnvttVf0NU844QS31VZbRc9RCJZ333138rxOCIF+IdBKwvz555/dmmuu6RYsWDAMV7SaJZdc0t1zzz3DzsUK0FAvv/zy2Kmhsueff959+eWXBQmkCPOjjz5yq666qvv999+HrmvzwQ477OBmz54dfcXNN9/csYCk5JVXXnHrrbeeY7GSCIEmIdBKwrz//vvdjjvuGMX5qaeecosvvrj7+uuvo+fDwg8++KArYdo1aE0pwkSz3XDDDd3TTz9t1Vv795NPPnHLL7+8wy0Syvz58wv8H3vssfDU0HeuY3F54403hsp0MLYI/Pbbb+7OO++s/abM1e+//772+1R1g1YSJkGW0047LYoRQYiNNtqoOMfkRdM5+OCD3ccffxytXxVh0jjPVKZZRR9gAAuZaDvttFP0yZkgLFhMEiblJZdc4g4//HD35JNPdtSfMmWKu/TSSzvK2vIFzfnZZ58t3A65AbGxfHf6Zffdd3czZsyo/bbgsOWWW7p58+bVfq8qbtB4wkTbgGTeeeed7Pfddddd3c033xytj//y+OOPd59++qk7++yz3TPPPOOWWWaZ5OCokjCvvfZax7MNmuB7ffjhh7Mfm0Xp6KOPjtbHf7nNNtsUfmH69c0333RrrLGGO+WUUzrqX3DBBcVC1lHYgi9kTeDbZRzssccehSb9zTffNObNIPP99tvPHXPMMWP2TFdccUVBmv3MZsl92UYTJgNpk002cQcccEARVYVwcgQN8t577x1W9auvviq0GwbsAw88UJz/7rvvCjIggs1qj6nofyCLI444oqOM82+//faw9stMciqjebGaDoqgaey9995u++23d5tttpmbOnVq1MwO32fSpElRTRq3xOqrr160h1+Y70wStM4wYn711Ve73XbbLWx64L+T5oaGba4hjl999dXGvBeB0vHjx7uxzOhgHDAnDz300MbgkHqQxhLmF198UaT/sPoA6GeffVaQ58UXX5x6l6FySDZGmERel1pqqULDJLXlhhtuGLqGAwiaNBj/M3369MJk9Ms4jvnXIMyyTocwIZ5BEAJnaEGTJ08uAlXkkTKocV900wQwp2OuB7R1CIIoODigVcb8nOBzzTXXuJ133rkSqF577TV35ZVXVtLWaBpBu8S3u9ZaazlyTU899VR3yy23NCaFCoVi6aWXHlImRvOuvV5LoI+xkQoU9tpeXfUbS5gEZSAmX/B75aSbkGN5xx13+JcWx5iJu+yyS3GMiUkHoWFCxkS5Y9KrSX7IIYfEminKbrzxxmQwKnlRn06gXd56660dkWrIjbIUydmj4kPG9A7lpptuciussEJx/eeffz40QdAuMc19YWEkZ7YKwSI47rjjqmhqVG0YKWDyNlHwsRNs67Yg1vHsuALWXnvtUoWjjvv22mZjCbPXF/HrE6mO5WCuv/767sILLyyqQsZomwwOzKSUH6lXwkQDSwkkkPLtpa4ZxHLMabTRUOgXTHzECPOtt94qXBVPPPFER3X6BF9oFfLII480gjBx77BI51hJVbx3L21gxY0bN84ddNBBvVxWaV2skmWXXbbQvittuMLGGkOYrL5MEnxbsa1xqOq5qUCYvvvss08HTJbOMmfOnKKcqPjKK6/s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) 



**多了一个学习率的参数**

![image-5.png](data:image/png;base64,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)

##### 梯度提升树的重要部分

1. 一个能够衡量集成算法效果的，能够被最优化的损失函数
2. 一个能够实现预测的弱评估器
3. 一种能够让弱评估器集成的手段，包括我们讲解的迭代方法，抽样手段，样本加权等等过程

#### 评分函数：

> 首先我们先来定义一个评分函数，这个评分函数能够帮助我们直接打印Xtrain上的交叉验证结果
> #cv: 交叉验证的次数 scoring: 评估指标 show;是否打印结果

```python

def regassess(reg,Xtrain,Ytrain,cv,scoring = ["r2"],show=True):
    score = []
    for i in range(len(scoring)):
        if show:
            print("{}:{:.2f}".format(scoring[i]
                                     ,CVS(reg
                                         ,Xtrain,Ytrain
                                         ,cv=cv,scoring=scoring[i]).mean()))
        score.append(CVS(reg,Xtrain,Ytrain,cv=cv,scoring=scoring[i]).mean())
    return score

```

```python
#设置eta的取值，观察一下eta如何影响我们的模型：
from time import time
import datetime
for i in [0,0.2,0.5,1]:
    time0=time()
    reg = XGBR(n_estimators=180,random_state=420,learning_rate=i)
    print("learning_rate = {}".format(i))
    regassess(reg,Xtrain,Ytrain,cv,scoring = ["r2","neg_mean_squared_error"])
    print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
    print("\t")
```

* eta越小运行时间越长。除了运行时间，步长还是一个对模型效果影响巨大的参数，如果设置太大模型就无法收敛（可能导致 很小或者MSE很大的情况），如果设置太小模型速度就会非常缓慢，但它最后究竟会收敛到何处很难由经验来定，在训练集上表现出来的模样和在测试集上相差甚远，很难直接探索出一个泛化误差很低的步长 

##### 学习曲线

```python
#画学习曲线
axisx = np.arange(0.05,1,0.05)
rs = []
te = []
for i in axisx:
    reg = XGBR(n_estimators=180,random_state=420,learning_rate=i)
    score = regassess(reg,Xtrain,Ytrain,cv,scoring = 
["r2","neg_mean_squared_error"],show=False)
    test = reg.fit(Xtrain,Ytrain).score(Xtest,Ytest)
    rs.append(score[0])
    te.append(test)
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,te,c="gray",label="XGB")
plt.plot(axisx,rs,c="green",label="XGB")
plt.legend()
plt.show()
```

 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABIcAAAEvCAYAAADfBqG/AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABwkUlEQVR4nO3dd3yV5eH+8c+TTRYjCRlAIGwSVtgBGaIIsmQIrqoFFcXxa7W2tcNWW0db/VZbrTgBR1WUrcgSVEZYgYQRVtgjk5CQRea5f38gpyArQJIn43q/XufFOc95xhXwGLhy3/djGWMQEREREREREZG6ycXuACIiIiIiIiIiYh+VQyIiIiIiIiIidZjKIRERERERERGROkzlkIiIiIiIiIhIHaZySERERERERESkDlM5JCIiIiIiIiJSh7nZHeBiAgMDTYsWLeyOISIiIiIiIiJSa2zevPmEMSbop9urZTnUokUL4uLi7I4hIiIiIiIiIlJrWJZ1+GLbNa1MRERERERERKQOUzkkIiIiIiIiIlKHqRwSEREREREREanDquWaQyIiIiIiIiIilaGkpIRjx45RWFhod5RK4+XlRdOmTXF3dy/X/iqHRERERERERKTOOHbsGH5+frRo0QLLsuyOU+GMMWRmZnLs2DEiIiLKdYymlYmIiIiIiIhInVFYWEhAQECtLIYALMsiICDgqkZGqRwSERERERERkTqlthZDZ13t16dySERERERERESkihw9epSIiAhOnjwJQFZWFhERERw+fJikpCRGjhxJq1at6N69OzfeeCOrVq0CYObMmQQFBdG1a1eioqK4/fbbKSgoqJBMKodERERERERERKpIs2bNmDp1Ks888wwAzzzzDFOmTCE4OJgRI0YwZcoU9u/fz+bNm3njjTc4cOCA89g77riDhIQEEhMT8fDwYNasWRWSSQtSV5K0tDSOHz+Om5sb7u7uzselXru6utodWURERERERESqwJNPPkn37t15/fXXWbNmDW+88QYff/wxMTExjB492rlfx44d6dix4wXHl5aWkp+fT8OGDSskj8qhSrJv3z6+/fbbcu9vWdYlS6TyPL/S63Ofu7i41Pr5lSIiIiIiIiLVlbu7O6+88grDhg1j2bJleHh4kJiYSLdu3S573KxZs1izZg0pKSm0bduWUaNGVUgelUOVpGfPnnTs2JHS0lJKSkqcj/K+Pnd7UVER+fn5F93nWpxbRJW3ULrUcw8PDzw9PZ2/nn3u4qIZiyIiIiIiIlK9LVmyhNTU1Ao9Z0hICMOGDbvifosXLyY0NJQdO3YwZMiQC94fO3YsSUlJtG3blrlz5wJnppW9+eabGGN47LHHeOWVV5zT066HyqFK4uHhgYeHR6VewxhzXol0NUXUxUqokpISiouLK6SIOrc4ulh5dLn3fvpcU+5ERERERESkNklISGD58uWsX7+eG264gTvvvJOoqCjn4tMA8+bNIy4ujqeffvqC4y3LYtSoUbzxxhsqh+q6c0cAVbZLFVHFxcUUFRVRVFR00efnbsvJyTnv/fIWTq6urhVSMp0tmjSlTkRERERERIByjfCpaMYYpk6dyuuvv054eDi//vWvefrpp3n//fd5+eWXWbhwoXPdocvdjWzNmjW0atWqQjKpHJJyqYwiqqys7KIlUnkKp7y8vPO2lZSUlOuaLi4uVyyRzt3WsGFDwsPDq6SAExERERERkdrvvffeIzw83DmV7NFHH2XmzJls3LiRr7/+mqeeeopf/vKXBAcH4+fnxx//+EfnsWfXHHI4HDRt2pSZM2dWSCbLGFMhJ6pIPXr0MHFxcXbHkBrE4XBQXFxc7pLpSvudy9XVlfDwcFq2bEmrVq0ICQnR6CMREREREZEaateuXXTo0MHuGJXuYl+nZVmbjTE9frqvRg5JreDi4oKXlxdeXl7XfS5jjHMh8LS0NPbv38+BAwdYsWIFK1aswNvbm5YtWzof9evXr4CvQERERERERMQeKodEfsKyLOd6Rn5+frRu3RqA3NxcDhw44Hzs2LEDgICAAFq1akXLli1p0aIFnp6edsYXERERERERuSoqh0TKyc/Pjy5dutClSxeMMaSnpzuLoi1btrBx40ZcXFxo2rSpcwpaWFgYLi4udkcXERERERERuSSVQyLXwLIsgoODCQ4OJiYmhtLSUo4ePeqcgvb999/z/fff4+npSUREhLMsatSokd3RRURERERERM6jckikAri5uREREUFERARw5naD505B2717NwANGjRwFkURERHUq1fPztgiIiIiIiIiKodEKoO3tzcdO3akY8eOGGM4efKkc1TRjh072LJlC5ZlERYW5lzYulmzZri6utodXUREREREROoYlUMilcyyLAICAggICKBXr16UlZVx/Phx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) 

 从学习曲线来看，没有固定的趋势，所以很难看出来eta如何影响模型效果 所以通常，我们不调整 ，即便调整，一般它也会在[0.01,0.2]之间变动 



#### 选择弱评估器：重要参数booster

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C71.png)

* gbtree :梯度提升树
* gblinear：线性模型
* dart：抛弃提升树，在建树的过程中会抛弃一部分树，比梯度提升树有更好的防过拟合功能



#### XGB的目标函数：重要参数objective

 ![image.png](data:image/png;base64,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) 

传统损失函数+模型复杂度

通过树的结构来衡量树模型的复杂度的式子。这样最小化Obj来力求最优的预测值，因此我们同时最小化了模型的错误率和模型复杂度

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C72.png)

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C73.png)

```python
#使用xgb自身的调用方式

reg = XGBR(n_estimators=180,random_state=420).fit(Xtrain,Ytrain)
reg.score(Xtest,Ytest)

#写参数
param = {"slient":False ,"objective":'reg:linear',"eta":0.1}
num_round = 180

#类train，可以直接导入的参数式训练数据，树的数量，其他参数都需要通过param来导入

bst = xgb.train(param,dtrain,num_round)
preds = bst.predict(dtest)

from sklearn.metrics import r2_score
r2_score(Ytest,preds)
MSE(Ytest,preds)
```



#### XGBoost中的正则参数[¶](http://127.0.0.1:8888/notebooks/sklearn_study/XGBOOST.ipynb#XGBoost中的正则参数)

一部分是控制树的结构（控制叶子数量的参数*叶子数），另一部分是我们的正则项。叶子的数量可以代表树的结构。

XGB默认L2正则化

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C75.png)



 网格搜索： 

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C76.png)

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C77.png)





#### 让树停止生长：gamma

> 回顾一下信息增益： 信息增益最大的特征上的分枝，就被我们选中。当信息增益低于一个阈值的时候，就让树停止生长。

------

在XGB中，我们首先使用目标函数来衡量树的结构的优劣，然后让树从深度0开始生长，每进行一次分枝，我们就计算目标函数减少了多少，当目标函数的降低低于我们设定的阈值，就让树停止生长。

gamma就是每增加一片叶子就会被减去的惩罚项 ---也叫**复杂性控制** 就是在树的叶子节点上进行进一步分枝所需要的最小损失函数的减少量。

**取值范围：0到正无穷，默认为0**

防止过拟合的神器，其效果不亚于n_estimators和max_depth

学习曲线完全看不出来，哪里泛化误差最小

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C78.png)

#### 新的工具xgboost.cv

xgboost.cv()可以返回训练集和测试集上的分数，易于观察模型的泛化能力

```python
import xgboost as xgb
dfull = xgb.DMatrix(X,y)
param1 = {'slient':False,'obj':'reg:linear','gamma':0}
num_round =180 #迭代次数
n_fold = 5
time0 = time()
cvresult1 = xgb.cv(param1,dfull,num_round,n_fold)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

通过这个表，可视化一下均方误差
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
plt.grid()
plt.plot(range(1,181),cvresult1.iloc[:,0],c="red",label="train,gamma=0")
plt.plot(range(1,181),cvresult1.iloc[:,2],c="orange",label="test,gamma=0")
plt.legend()
plt.show()

```

 ![img](data:image/png;base64,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) 

 ##### xgboost的评估指标到底有哪些？ 

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C79.png)

```python
#想更改模型指标，就写在参数列表里面
param2 = {'slient':True,'obj':'reg:linear','gamma':0,'eval_metric':'mae'}
cvresult2 = xgb.cv(param2,dfull,num_round,n_fold)

plt.figure(figsize=(20,5))
plt.grid()
plt.plot(range(1,181),cvresult2.iloc[:,0],c="red",label="train,gamma=0")
plt.plot(range(1,181),cvresult2.iloc[:,2],c="orange",label="test,gamma=0")
plt.legend()
plt.show()
```

 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABH4AAAEvCAYAAAAzXwbsAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAA4oklEQVR4nO3deXiddZ3//+cnS9d0LwToQgutUNpiS8PWCpOIYlFAvjM4LC7MyFDnKyiiIuoM6tdrXEbUUS5HHRS+OD+ROmwDA8yAOMR+HZYuUKEblEKhLdCWloame9LP7487SZP2pEnOOTn3SfJ8XNe57vvc6yvJm5Pw7n1/7hBjRJIkSZIkSb1PSdoBJEmSJEmS1D1s/EiSJEmSJPVSNn4kSZIkSZJ6KRs/kiRJkiRJvZSNH0mSJEmSpF7Kxo8kSZIkSVIvVVbIk40ePTpOmDChkKfM2Y4dOxg8eHDaMdTDWDfqKmtG2bBulA3rRl1lzSgb1o2yYd1kb8mSJW/FGI/ItK6gjZ8JEyawePHiQp4yZ7W1tVRXV6cdQz2MdaOusmaUDetG2bBu1FXWjLJh3Sgb1k32Qgivtreuw1u9Qgi3hRA2hRCWHbT8MyGEF0IIy0MI38tHUEmSJEmSJOVPZ8b4uR2Y23pBCKEG+DBwcoxxKvD9/EeTJEmSJElSLjps/MQYFwBbD1r8v4Hvxhj3NG2zqRuySZIkSZIkKQfZjvHzLuCsEMK3gN3AF2OMi/IXS5IkSZIk5cu+fftYv349u3fvTjtKu4YNG8bKlSvTjlHUBgwYwNixYykvL+/0PiHG2PFGIUwAHowxTmt6vwz4b+Ba4FTgt8BxMcPBQgjzgHkAlZWVs+bPn9/pcMWgvr6eioqKtGOoh7Fu1FXWjLJh3Sgb1o26yppRNqyb4lNRUUFlZSXDhg0jhJB2nIwaGxspLS1NO0bRijFSV1fHxo0bqa+vb7OupqZmSYyxKtN+2V7xsx64t6nRszCEsB8YDWzOEOwW4BaAqqqq2NNG6HZUcWXDulFXWTPKhnWjbFg36iprRtmwborPypUrGTt2bNE2fQC2b9/OkCFD0o5R1IYMGUJ9fT1VVRl7PBl1ZnDnTP4deC9ACOFdQD/grSyPJUmSJEmSulkxN33UOdn8DDvzOPc7gSeBE0II60MIVwK3Acc13fI1H7gi021ekiRJkiRJ27Zt46c//WmX9/vgBz/Itm3b8h+oCH3nO99h0qRJnHDCCTzyyCN5O26Ht3rFGC9rZ9XH8pZCkiRJkiT1Ws2Nn09/+tNtlnc0rs/DDz/c3dGKwooVK5g/fz7Lly/n9ddf533vex8vvvhiXsY8yvZWr75rxw649VZYtiztJJIkSZIk9Qhf/vKXWbNmDTNmzODUU0+lpqaGyy+/nOnTpwNw0UUXcfbZZzN16lRuueWWlv0mTJjAW2+9xdq1a5kyZQpXXXUVU6dO5dxzz2XXrl2HnGfNmjWcccYZnHrqqXzta19rGWS8vr6ec845h1NOOYXp06dz//33A7B27VpOPPFE/uZv/oZp06bx0Y9+lMcee4w5c+YwefJkFi5cCMA3vvENrrjiCs4991wmTJjAvffey5e+9CWmT5/O3Llz2bdvHwDf/OY3OfXUU5k2bRrz5s2jszdH3X///Vx66aX079+fiRMnMmnSpJZz58rGT1ft2AyL/gb+8JO0k0iSJEmS1CN897vf5fjjj2fp0qXcdNNNLFy4kG9961usWLECgNtuu40FCxawePFibr75ZrZs2XLIMVavXs3VV1/N8uXLGT58OPfcc88h21x77bVce+21LFq0iGOOOaZl+YABA7jvvvt45plnePzxx/nCF77Q0pR56aWXuPbaa3nuuedYtWoVv/nNb/jjH//I97//fb797W+3HGPNmjU89NBD3H///XzsYx+jpqaG559/noEDB/LQQw8BcM0117Bo0SKWLVvGrl27ePDBBwG46aabmDFjxiGvz372swBs2LCBcePGtZxr7NixbNiwIddvO5D9U736rmEj4WzgzRfTTiJJkiRJUtd97nOwdGl+jzljBvzoR53e/LTTTmPixIkt72+++WbuueceSkpKWLduHatXr2bUqFFt9pk4cSIzZswAYNasWaxdu/aQ4z755JP8+7//OwCXX345X/ziF4HkUehf/epXWbBgASUlJWzYsIGNGze2HLf5yqOpU6dyzjnnEEJg+vTpbc5x3nnnUV5ezvTp02lsbGTu3LkAbbZ7/PHH+d73vsfOnTvZunUrU6dO5YILLuD666/n+uuvb/f7kenKoHwNxm3jp6v6DYG9wL5Du4+SJEmSJKljgwcPbpmvra3lscce47HHHqOyspLq6mp27959yD79+/dvmS8tLc14q1d77rjjDjZv3sySJUsoLy9n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) 



过拟合的两种方法： 1.训练集的分数降下来，红色线往上走 2.测试集的分数升上去，黄色线往下走

我们的调参目标就是让两条线越来越靠近 gamma参数通过控制训练集上的训练，降低训练集上的表现

其实树模型一般通过剪枝降低过拟合



```python
param1 = {'slient':True,'obj':'reg:linear','gamma':0}
param2 = {'slient':True,'obj':'reg:linear','gamma':20}

num_round = 180
n_fold=5

time0 = time()
cvresult1 = xgb.cv(param1,dfull,num_round,n_fold)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
cvresult2 = xgb.cv(param2,dfull,num_round,n_fold)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

plt.figure(figsize=(20,5))
plt.grid()
plt.plot(range(1,181),cvresult1.iloc[:,0],c="red",label="train,gamma=0")
plt.plot(range(1,181),cvresult1.iloc[:,2],c="orange",label="test,gamma=0")
plt.plot(range(1,181),cvresult2.iloc[:,0],c="green",label="train,gamma=20")
plt.plot(range(1,181),cvresult2.iloc[:,2],c="blue",label="test,gamma=20")
plt.legend()
plt.show()

```

 ![img](data:image/png;base64,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l8fFF19c3pCq6Dvf+Q5XXnnlbssee+wxzjrrLACWLFnC4YcfXr6uW7duLFmypMbff2VsDtWTnKy1fLbhUIgRQkg6jiRJkiRJ3HgjTJ9eu/scNAgeeKBqYydNmsSkSZMYPHgwAKWlpcyfP58RI0Zw0003ceutt3LuuecyYsSI/e5n7ty5HHXUUXTv3h2ASy+9tPzsneLiYq666irmz59PCIFt27aVbzdy5EhycnLIycmhbdu2nHfeeQAMHDiQmTNnlo87//zzy5f379+fzp07A3DUUUexaNEi8vLyGDduHBMmTABg0aJFzJ8/n7y8PJ555pkqfRe//e1vmTp1Kq+//joAMca9xoQ66ifYHKon7duso2hbGzYUl9ImNzvpOJIkSZIkJS7GyO23384111yz17pp06YxceJEbr/9ds4880zuvvvu/e5nX+666y5GjhzJhAkTKCoqoqCgoHxdy5Yty+czMjLK32dkZLB9+/a9xlUcU3FcYWEhr776Km+99RZZWVkUFBSwefNmgCqdOfTqq69y77338vrrr5fvv1u3bhQWFpaPX7x48W7Za5PNoXpyaE4pAKuWrLU5JEmSJElqEKp6hk9tysnJoaSkBIAvfOEL3HXXXVx++eVkZ2ezZMkSMjMz2b59O+3bt+eKK64gOzubJ554Yq9tAa688kquv/56Bg4cyEcffURRURH5+fm7na1TXFxM165dAcr3U9uKi4tp164dWVlZzJ07l7fffrt83YHOHHrvvfe45pprePnll+nYsWP58i984Qvccccd5TehnjRpEj/5yU/qJL/NoXpyWNvUaWuffLiI/P5HJJxGkiRJkqRk5OXlcfLJJzNgwADOOussLrvsMk488UQAsrOz+e1vf8uCBQu4+eabycjIIDMzk1/+8pcAjB49mi996Ut07dqV1157jZkzZ9K5c2dat27Nww8/zKhRo+jQoQPHH398+fFuueUWrrrqKsaOHctpp51WJ59p1KhRPPLIIxxzzDH07t2b4cOHV3nbm2++mdLSUr785S8DcMQRR/Diiy/Svn177rrrLoYNGwbA3XffTfv27eskf9jfqVdJGTp0aJw6dWrSMaqlsLBwv6d33XnTd7n3F7/gv+77A1+79cv1F0wN1oFqRqqMdaN0WDeqLmtG6bBulA7rJhlz5syhb9++ScdIW0lJCTk5Oaxfv56vfe1r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) 

#### 过拟合剪枝参数：

![](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CPictures%5CSaved%20Pictures%5C81.png)

##### 初始参数

```python
dfull = xgb.DMatrix(X,y)

param1 = {'slient':True
         ,'obj':'reg:linear'
         ,'subsample':1
         ,'max_depth':6
         ,'eta':0.3
         ,'gamma':0
         ,'lambda':1
         ,'alpha':0
         ,'colsample_bytree':1
         ,'colsample_bylevel':1
         ,'colsample_bynode':1
         ,'nfold':5}

num_round = 200



time0 = time()
cvresult1 = xgb.cv(param1 ,dfull ,num_round)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

fig ,ax = plt.subplots(1,figsize=(15,8))
ax.set_ylim(top=5)
ax.grid()
ax.plot(range(1,201),cvresult1.iloc[:,0],c='red',label='train,original')
ax.plot(range(1,201),cvresult1.iloc[:,2],c='orange',label='test,original')

param2 ={
    'slient':True
    ,'obj': 'reg:linear'
    ,'max_depth':2
    ,'eta':0.05
    ,'gamma':0
    ,'nfold': 5
}
param3 = {
    'slient':True
    ,'obj': 'reg:linear'
    ,'max_depth':2
    ,'eta':0.05
    ,'gamma':5
    ,'alpha':0
    ,'colsample_bytree':0.85
    ,'colsample_bytree':1
    ,'colsample_bylevel':0.4
    ,'colsample_bynode':1
    ,'nfold': 5
}

time0 = time()
cvresult2 =xgb.cv(param2,dfull,num_round)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

cvresult3 =xgb.cv(param3,dfull,num_round)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

ax.plot(range(1,201),cvresult2.iloc[:,0],c='green',label='train,lastl')
ax.plot(range(1,201),cvresult2.iloc[:,2],c='blue',label='test,last')
ax.plot(range(1,201),cvresult3.iloc[:,0],c='gray',label='train,this')
ax.plot(range(1,201),cvresult3.iloc[:,2],c='pink',label='test,this')

ax.legend(fontsize='xx-large')#加大字体
plt.show()



```

 ![img](data:image/png;base64,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CmAiIiIiIlJDfYYglgP/Z5rmKsMwAoGVhmHMN01zUxPX1vrFxVlrBTAREREREaEeHTDTNFNN01xV9ToP2AzENXVhbYK/P4SEKICJiIiIiAjQwHvADMOIB4YAy5qkmraoY0drEg4REREREWn3DNM067ejYQQAPwB/M03z4zq+Pw2YBhAdHT3s/fffd2edjZKfn09AQIBHaxjw4IN45eSwauZMj9bR3FrCtW/PdP09S9ffc3TtPUvX33PawrU3DAMfHx+8vb0xDMPT5TSIaZqtrua2pLmuv2malJaWUlJSQn0y1NixY1eappl49PZ6BTDDMLyAL4FvTdP858n2T0xMNFesWHHS4za1pKQkxowZ49kibrkFvvwSDh70bB3NrEVc+3ZM19+zdP09R9fes3T9Pae1X/uSkhL27NlDaGgoQUFBeHl5tapAk5eXR2BgoKfLaLea4/qbpklZWRm5ublkZWXRuXNnfHx8TvgewzDqDGD1mQXRAGYBm+sTvuQoXbrAoUNQWOjpSkRERERapMzMTEJDQ4mIiGiVHTBp+wzDwNvbm4iICEJDQ8nMzGz0sepzD9gZwPXAOMMw1lQtkxp9xvamZ09rvXOnZ+sQERERaaHy8vIICgrydBki9RIUFEReXl6j33/SaehN01wM6NcQjdWjh7Xetg0GDPBsLSIiIiItUEVFBV5eXp4uQ6RevLy8qKioaPT7GzQLojSCK4Bt3+7ZOkRERERaMA07lNbiVP+sKoA1tcBAiImxOmAiIiIiItKuKYA1hx491AETERERkSYzZsyYFjsT5qnWZhgGM2bMcFs9xzN16lTi4+Ob/DwKYM2hZ08FMBEREZF2bNWqVcyYMYOUlBRPlyIedtJJOMQNevSwpqLPzQXN8CMiIiLS7qxatYpHH32UMWPGNEmXZd68eW4/prucam1FRUU4HG0ntqgD1hw0EYeIiIiI1FNhI54f6+3tjbe3dxNU03hFRUXAqdfm6+urACYN5HoWmCbiEBEREWl3ZsyYwa233grA2LFjMQwDwzB48803mTp1KoZhsHfvXq666ipCQ0Pp168fALt37+aee+6hX79+BAQEEBAQwFlnncW33357zDnqus/KMAyuu+46fvjhB04//XScTiddunThxRdfrHftS5cu5dxzzyUoKAh/f3/OOOMMvvnmm2P2c53rq6++IjExEV9fX5566qnj1lZWVsYf/vAHOnTogJ+fH6NHj2b58uXH/Rw17wFLSkrCMAxef/11Zs6cyaBBg/Dx8WHw4MF8//33td6bmZnJgw8+yJAhQwgODsbpdJKYmMi7775b72vgbm0nSrZk3bpZa3XARERERNqdSy+9lH379jFr1iweeugh+vTpA8CoUaNISkoC4Pzzz6dPnz48/vjjlJSUALB8+XLmz5/PxRdfTNeuXcnJyeGdd95h0qRJfPfdd4wdO/ak5163bh1XXHEFt956K1OmTOG9995j+vTp9O3bl/Hjx5/wvT/99BPjx48nIiKCBx54AKfTyezZs7ngggt4//33+fWvf11r/5UrV/L1119zxx13cNttt9GpU6fjHvu2225j9uzZXHTRRUycOJEtW7Zw3nnnERoaSseOHU/6uQBeffVV8vLymDp1KgEBATz33HNcfPHF7N69m9DQUACSk5N59913ueyyy7j11lspKSnh448/5tprr6WsrIwpU6bU61zupADWHJxO6NRJAUxERESkHRo4cCAjRoxg1qxZnHPOOXXOCHjmmWcyc+bMWtsmTZrE5ZdfXmvbvffey+DBg3nqqafqFcA2btzI8uXLGTp0KAA33XQTnTt35tVXXz1pALv//vux2+0sWbKEzp07A3DrrbcycOBA7r33XiZPnlzrAdpbtmxhyZIljBw58oTHXb9+PbNnz+bqq6+u1YkaNGgQt9xyS70D2MGDB9m8eTOmaRIYGMiYMWMYNmwY7733HnfeeScAAwYMICUlBbvdXv2+++67jwkTJvDEE08ogLVpPXtqCKKIiIhIQ9x3H6xZ4+kqjhg8GJ57rkkO7QoMNfn5+VW/Li4upqCgANM0GTNmDB988EG9jnv66adXhy+w7qcaOXIkO3fuPOH7Dh06xPLly7n55purwxdAcHAwt99+Ow899BDLly9n1KhR1d8bMmTIScMXwFdffQVYYbKmKVOm8Nvf/rZenwvg+uuvJyAggLy8PACGDh1KUFBQrc/m4+NT/bq0tJT8/HwqKyuZMGECDz30EHl5eQQGBtb7nO6gANZcevSA998H0wQ96V1EREREakhISDhmW1lZGY899hhvvfUWu3fvrvU9o54/T3bp0uWYbaGhoaxbt+6E79u1axdA9XDJmvr27Vu9T80AVtdnqItrKv4eronqqjgcjnofA47/2TIzM6u/Nk2T559/npkzZ7Jt2zZM06y1f1ZWlgJYm9WzJ2RnQ0YGRER4uhoRERGRlq+Juk0tkdPpPGbbfffdxyuvvMIdd9zBmWeeSVhYGHa7ndmzZ9d7EomaQ+9qOjqIHE9dQc/13qO/V9dnONG5T3Ts+qjPZ3vmmWf43e9+x7XXXssf//hHIiMjcTgcfP311zz77LNUVlbW+3zuogDWXGpORa8AJiIiItKu1LdjVdO7777LDTfcwEsvvVRr+6xZs9xV1nG5nlW2adOmY763ZcuWWvs0lKvLtW3bNk4//fTq7eXl5aSkpDBo0KBGHbcu7777LmeffTbvvPNOre0LFixw2zkaStPQNxc9C0xERESk3QoICACsIW/1Zbfbj+nQbN26lU8//dSdpdUpJiaGxMRE3n//ffbt21e9PS8vj1dffZWYmBiGDx/eqGNPmjQJgOeff77W9jlz5pCdnd3omutS1zVMS0trlhB7POqANZeEBLDbNRGHiIiISDuUmJiIYRg88cQTZGdn43Q6a3V/6jJ58mRmzZqFn58fw4YNIzk5mVdeeYU+ffqwevVqt9Y3depU5syZw65du6o7W88++ywTJkxgxIgR3HHHHfj6+jJ79mx2797N+++/X2sGxIYYOHAgN9xwA2+99RaFhYWce+65bN26lXfeeYdu3bo1qlt4PJMnT+ZPf/oTV111FePGjSM1NZVXX32VTp06kZaW5rbzNIQCWHPx9ob4eHXARERERNqhbt268dJLL/HMM89w6623UlFRwezZs0/4nmeffRZfX18+/vhj5syZQ+/evXn11VfZvHmz2wNYfn4+TqeTkJCQ6m1nnnkmSUlJPPzwwzz55JNUVFQwZMgQvvzyy+ouVmO99tprxMTEMGfOHObNm8ewYcOYO3duddBzlwcffJDS0lLmzJnDp59+SkJCAg899BABAQHceOONbjtPQxgNudGtvhITE80VK1a4/bgNlZSUVOdzFjzm/PPh4EFw838wLVGLu/btjK6/Z+n6e46uvWfp+ntOa7/2mzdvrnO2vdbCE1OZu5NpmsTExHDDDTfw97//3WN1VFRUEBUVxWWXXca///3ver/PE9e/Pn9mDcNYaZpm4tHbdQ9Yc+rZ0+qANUHoFRERERFpjPXr11NUVMSDDz7YbOcsKio6Ztvs2bPJzMxk3LhxzVaHJ2gIYnPq0QMKCqwuWGysp6sREREREWHgwIHk5uY26zlfeOEF5s2bx7nnnktQUBDLly9nzpw5DB06lMsuu6xZa2lu6oA1p549rbUm4hARERGRdsw1Ack//vEPpk+fzrx585g2bRrz589v9OQerYU6YM2p5lT0Z5/t2VpERERERDxkzJgxrfq+xVOhDlhz6tzZmg1RMyGKiIiIiLRLCmDNyW6Hbt00BFFEREREpJ1SAGtuvXvDhg2erkJERERERDxAAay5DRsGO3ZAdranKxERERERkWamANbcEquexbZqlWfrEBERERGRZqcA1txcAWzFCs/WISIiIiIizU4BrLmFh0NCggKYiIiIiEg7pADmCYmJsHy5p6sQEREREZFmpgDmCYmJkJIC6emerkRERERE5JQZhsGMGTMa9d6kpCQMwyApKcmtNdUlPj6eqVOnNvl5TkQBzBNc94GtXOnZOkRERESkWaxatYoZM2aQkpLSZOeorKxkxowZfPrpp012Djl1CmCeMHSotdZ9YCIiIiLtwqpVq3j00UebPIA9+uijHglgRUVF/OlPf2rUe8866yyKioo466yz3FxVy+TwdAHtUkgI9OypACYiIiIirVZFRQUVFRV4e3vj6+vb6OPYbLZTen9row6YpyQmKoCJiIiItAMzZszg1ltvBWDs2LEYhoFhGLz55psA7Nq1ixtuuIHo6Gh8fHzo1asXf//736msrKx1nI8++ogRI0YQEhJCQEAAvXr14vbbbwcgJSUFLy8vAObMmVN9jjFjxpy0vrfffpuhQ4fidDoJCwvjsssuY8uWLbX2cd2n9dprr/HMM8/QrVs3fHx8WLJkCVD3PWCpqalcddVVBAUFERISwlVXXcWhQ4eO2beue8BmzJiBYRisX7+e+++/n6ioKPz8/Dj//PPZvXt3rfNs3LiRm2++mR49euDn50dISAjnnXcey1vopHfqgHlKYiK8+y4cPAgxMZ6uRkRERESayKWXXsq+ffuYNWsWDz30EH369AFg1KhR7Nixg5EjRxIQEMA999xDZGQkSUlJ/O53vyMlJYWXXnoJgAULFnDFFVdw1lln8de//hUvLy+Sk5P54osvAIiMjGTOnDlMmTKF0aNHM23aNACio6NPWNszzzzDAw88wGmnncYTTzxBZmYmL774IiNHjmT58uV079691v7PP/885eXlTJs2DT8/P2JjY+s8bnFxMRMmTGDr1q3ccccd9OnTh/nz5zNp0qQGXbubbrqJ0NBQ/vznP5Oamso///lPrrvuOhYtWlS9z4IFC1i7di1XXnklXbp04dChQ8yaNYuzzz6bVatW0bt37wads6kpgHlKzQcyX3ihZ2sRERERkSYzcOBARowYwaxZszjnnHNqdaUmTZpEYGAga9asISgoCIDbbruN2NhYnnvuOe677z569OjBl19+SWBgIN999x0Ox5Ef4Z966ikA/P39ueaaa5gyZQpdu3bluuuuO2ldGRkZPPzwwwwbNowff/wRHx8fAC655BKGDx/O73//ez788MNa70lLS2Pbtm0EBwef8NivvfYamzZt4tVXX60Og3feeSfXX389q1atOvlFq9KxY0c++eST6q/Dw8P5v//7PzZt2kTfvn0BuPnmm/njH/9Y63233347ffv25bnnnmPmzJn1Pl9zUADzlCFDwDAUwERERESOZ+V9kLXG01UcEToYhj3ntsNlZ2czd+5c/u///o/S0lLSazyiaOLEiTz77LMsXLiQHj16EBwcTEFBAXPnzuVCN/3sOH/+fIqLi7nvvvuqwxfA0KFDmTBhAl9//TXl5eW1At+111570vAF8NVXXxEUFMSNN95Ya/u9997LO++8U+8a77jjjlpfjx07FoCdO3dWBzB/f//q7xcWFlJUVATAaaed1iKHIeoeME8JCIA+fXQfmIiIiEg7tW3bNkzT5JlnniEyMrLWct555wFw+PBhwOoe9e7dm1/96ld06NCBa665hnfffZeysrJGn981I6NrSGRNffv2paioiEOHDtXanpCQUO9jd+nSpfq+NJeePXs2qMYuXbrU+jo0NBSAzMzM6m25ublMnz6d2NhY/P39iYiIIDIykq+++oqsrKwGna85qAPmScOHw9y5YJpWN0xEREREjnBjt6klck2ycfvtt3PZZZfVuU/Xrl0BiIqKYvXq1SxcuJBvv/2WefPm8d577/H000+zePFiAgICGl2HUcfPoaZp1vk9p9NZr2OaplnncRvKbrcf9/guN910Ez/88AP3338/Q4cOJTg4GJvNxhNPPMHOnTtPuQZ3UwDzpMREmDMH9u+Hjh09XY2IiIiINJG6wki3bt2qZyucMGHCSY/h5eXFxIkTmThxIgAvv/wyd911F++++y7Tpk1rcOCJj48HYNOmTQx1Pae2ypYtW3A6nURFRTXomC4JCQn8/PPPlJWV1eqCbdu2rVHHO57s7GzmzZvHI488cswsjA8//LBbz+UuGoLoSTUn4hARERGRNsvVoao5JC4yMpLx48czZ84cduzYccx7cnNzKSkpAawJM47mCk2u4Xh2ux1fX996D7s755xz8PX15YUXXqC0tLR6+9q1a5k/fz7nn39+rfu/GmLSpEnk5uYye/bsWttfeOGFRh3veOx2O4ZhHDNlf1JSEsuWLXPrudxFHTBPGjQI7HZYvhwuucTT1YiIiIhIE0lMTMQwDJ544gmys7NxOp2cfvrpvPLKK5xxxhkMGTKEW265hT59+pCTk8OGDRv46KOP2LBhA/Hx8dxyyy0cPnyY8ePH07lzZ9LT05k5cyZOp5NLL720+jzDhw/nu+++45lnnqFjx45ERUUxbtw44EjHy3XvV3h4OH/5y1/43e9+x1lnncXVV19dPQ19UFAQTz75ZKM/76233sorr7zCnXfeyYYNG+jduzfz589nz549QN0dwcYIDAxkzJgxPP300xQVFdGzZ0/Wr1/P7Nmz6devH3l5eW45jzspgHmS0wmDB8PixZ6uRERERESaULdu3XjppZd45plnuPXWW6moqGD27NlMnTqVVatW8de//pWPPvqIgwcPEhoaSs+ePXnkkUeIqXpe7HXXXcesWbN47bXXyMzMJDw8nJEjR/KnP/2p1sQWrtDzyCOPUFhYyNlnn10dwAoKCo55rtcDDzxAdHQ0zz77LA8++CBOp5OxY8fy+OOP06NHj0Z/XqfTyYIFC7jvvvuYPXs2NpuN8847jw8++IDu3bvj6+vb6GMf7fXXX+eRRx5hzpw5FBQUMGjQID799FPefvvtWg93bikUwDxt7Fh44QUoKrICmYiIiIi0SXfccccx06oDxMXF8corr5zwvZdddtlxJ+qoqV+/fvzwww/HbN+4cSPp6em8+eabx3zvhhtu4IYbbjjhcceMGVNr4ouj1fW9Dh068MEHH9Tatnr1agA6dep0wmPPmDHjmHu6wOriHb1vZGQkb7/99jH7jh8//phtru6fJ+keME8bOxZKS2HJEk9XIiIiIiJt1IIFCxg5ciQXXHBBs53T9Tyumv75z39is9k4++yzm62OlkYdME8780zrPrDvv4c6UrqIiIiIyKmaPn0606dPb9ZzXnXVVQQGBnLaaadRUVHBV199xYIFC7j77ruJi4tr1lpaEgUwTwsKgmHDoAWOTxURERERaazzzz+f1157jS+//JKioiK6du3Kk08+yQMPPODp0jxKAawlGDsW/vlPKCgAf39PVyMiIiIicspuv/12br/9dk+X0eLoHrCWYOxYKCuDn37ydCUiIiIiItKEFMBagjPOAIfDug9MRERERETaLAWwliAgAIYP131gIiIiIiJtnAJYSzF2LCxfDi3wad0iIiIiIuIeCmAtxdixUFEBixd7uhIREREREWkiCmAtxahR4OWl+8BERERERNowBbCWws8PRozQfWAiIiIiIm2YAlhLMmYMrFwJOTmerkRERERERJqAAlhLMnYsVFbCokWerkREREREWpExY8YwZswYtx5z6tSpxMfHu/WY9TVjxgwMw/DIuZuaAlhLMnKkNSX9Z595uhIRERERcaNVq1YxY8YMUlJSPF1Ki5GcnMyMGTNYs2aNp0tpVgpgLYmvL0yeDB9+CCUlnq5GRERERNxk1apVPProo00WwObNm8e8efOa5NhNJTk5mUcffVQBTDzsmmsgOxvmzvV0JSIiIiLiAYWFhQ1+j7e3N97e3k1QjbibAlhLM348RETAe+95uhIRERERcYMZM2Zw6623AjB27FgMw8AwDN58802mTp2KYRjs3buXq666itDQUPr16wfA7t27ueeee+jXrx8BAQEEBARw1lln8e233x5zjrruATMMg+uuu44ffviB008/HafTSZcuXXjxxRcb/VnmzJnDxIkT6dChA97e3nTu3Jnp06eTl5dXa7+KigqeeOIJevfujZ+fH2FhYQwbNoyXXnoJgDfffJNzzjkHgBtvvLH6msyYMaPRtbUWDk8XIEfx8oJf/xpmz4a8PAgM9HRFIiIiInIKLr30Uvbt28esWbN46KGH6NOnDwCjRo0iqeoRROeffz59+vTh8ccfp6TqVpTly5czf/58Lr74Yrp27UpOTg7vvPMOkyZN4rvvvmPs2LEnPfe6deu44ooruPXWW5kyZQrvvfce06dPp2/fvowfP77Bn+WFF16ge/fu3HvvvYSGhrJq1SpeffVV1q9fz/c1nmf7l7/8hb/85S9MnTqV3/zmNxQXF7Np0yYWLVrEXXfdxVlnncXvf/97nnzySaZNm8bo0aMBGDhwYINram0UwFqia66Bl1+2JuO47jpPVyMiIiLiEffNvY81B9d4uoxqg2MG89x5zzX4fQMHDmTEiBHMmjWLc845p87ZCs8880xmzpxZa9ukSZO4/PLLa2279957GTx4ME899VS9AtjGjRtZvnw5Q4cOBeCmm26ic+fOvPrqq40KYD/++CP+/v61to0cOZKpU6eydOlSRowYAcDnn3/OpEmTmD17dp3H6dq1K+PHj+fJJ59k5MiRXNeOfubVEMSWaORI6NwZ3n3X05WIiIiISDO48847j9nm5+dX/bq4uJiMjAzy8vIYM2YMy5cvr9dxTz/99OrwBeDr68vIkSPZuXNno+p0ha/KykpycnJIT0/nrLPOAqhVU3BwMBs3bmTz5s2NOk9bpg5YS2SzwdVXwzPPQFoaREZ6uiIRERGRZteYblNrlZCQcMy2srIyHnvsMd566y12795d63v1fUZWly5djtkWGhrKunXrGlXnL7/8wh//+EcWL15McXFxre9lZWVVv37ssce45JJL6Nu3L3369GHcuHFcdtll9eratXXqgLVU11wDFRXWlPQiIiIi0qY5nc5jtt1333389a9/5YILLuDdd99l7ty5zJ8/n2uuuQbTNOt1XLvdXuf2+r6/ppSUFMaOHcuBAwd4+umn+fzzz5k/fz5zq2bvrqysrN539OjRJCcn85///IcRI0bwySefMG7cOG6++eYGn7etUQespRowAPr2tYYh3nGHp6sRERERkVNQ345VTe+++y433HBD9cyBLrNmzXJXWQ3y2WefUVhYyFdffUV8fHz19q1bt9a5f3BwMNdccw3XXHMN5eXlTJkyhTfeeIMHH3yQnj17NuqatAXqgLVUhmF1wRYvhqNaziIiIiLSugQEBAC1h+mdjN1ur9VVAivsfPrpp+4srUH1AMfU9OSTTx6zb0ZGRq2vHQ4HAwYMACAzMxNo3DVpC9QBa8muuw7+/Gd49VV4/HFPVyMiIiIijZSYmIhhGDzxxBNkZ2fjdDo5/fTTT/ieyZMnM2vWLPz8/Bg2bBjJycm88sor9OnTh9WrV7u1vqlTpzJnzhx27dpVq7tV03nnnYevry+TJk3itttuw26388UXXxwTtgD69OnDmWeeyfDhw4mOjmbr1q3861//olevXgwbNgyAfv364efnxyuvvEJAQACBgYH079+f/v37u/WztTTqgLVkXbrAxRdbAawRT0QXERERkZahW7duvPTSS2RkZHDrrbdy9dVX88MPP5zwPc8++yx33XUXX3zxBdOnT2fu3Lm8+uqrXHTRRW6vLz8/H6fTSUhIyHH36d69O19++SXBwcH86U9/4rHHHiMuLq7OB0Pff//9HDx4kH/84x/ceeedfPjhh9x888388MMPeHl5AVYH7K233sLb25u77rqLq6++mg/bwfwHRmNuwDuZxMREc8WKFW4/bkMlJSXV+ZyFVuWHH2DMGPj3v6HqCeqtQZu49q2Yrr9n6fp7jq69Z+n6e05rv/abN2+ufjhxa5SXl0dgYKCny2g00zSJiYnhhhtu4O9//7uny2kwT1z/+vyZNQxjpWmaiUdvVwespTvrLBg8GJ5/HpogLIuIiIhI+7Z+/XqKiop48MEHPV1Ku6AA1tIZBtx7L2zcCAsXeroaEREREWljBg4cSG5uLhEREZ4upV1QAGsNrrrKehjzc895uhIRERERETkFCmCtga8v3H47fPUV7Njh6WpERERERKSRFMBaizvuAIcDXnzR05WIiIiIiEgjKYC1FrGxcOWV8MYbcOCAp6sREREREZFGUABrTf70J6ishOuvh4oKT1cjIiIiIiINdNIAZhjGG4ZhHDYMY0NzFCQn0KuXNQRx4UJ48klPVyMiIiIiIg1Unw7Ym8B5TVyH1NeNN8LVV8Mjj8BPP3m6GhERERERaYCTBjDTNH8EMpuhFqkPw4CZM6FLF7jmGsjU/zQiIiIiIq2F7gFrjYKC4P33rck4brrJui9MRERERERaPMM0zZPvZBjxwJemafY/wT7TgGkA0dHRw95//3131dho+fn5BAQEeLqMJtPxf/+j+8svs2vqVHZPmeLpcmpp69e+pdP19yxdf8/RtfcsXX/Pae3XPjg4mO7du3u6jEarqKjAbrd7uoxG6d+/P2eeeSYzZ870dCmN5onrv2PHDnJyck64z9ixY1eappl49HaHu4owTfPfwL8BEhMTzTFjxrjr0I2WlJRES6ijyZx9NuTnk/DmmyRceCFcdpmnK6rW5q99C6fr71m6/p6ja+9Zuv6e09qv/ebNmwkMDPR0GY2Wl5d30vpXrVrF559/ztSpU4mPj2+SOiorK/nLX/7C4MGDueSSS+r1HsMw8PLyarLrv3DhQn788Ufuu+8+QkJCmuQc9bn+7ubr68uQIUMa9V4NQWzNDANefRVGjIAbboC1az1dkYiIiIjUYdWqVTz66KOkpKQ02TkqKyt59NFH+fTTT5vsHA21cOFCHn30UbKzsz1dSotRn2no3wN+BnoZhrHPMIybm74sqTdfX/j4YwgNhYsugsOHPV2RiIiIiIgcR31mQbzaNM1Y0zS9TNPsaJrmrOYoTBogNhY+/dQKX9dfD/W4r09EREREmseMGTO49dZbARg7diyGYWAYBm+++SYAu3bt4oYbbiA6OhofHx969erF3//+dyqPmmjto48+YsSIEYSEhBAQEECvXr24/fbbAUhJScHLywuAOXPmVJ+joUNTS0tLmTFjBqeffjrh4eH4+vrSr18/XnjhBY6eO+Lw4cPcdtttdOnSBR8fH6Kjoxk3bhwLFy4EYOrUqfztb38DICEhobqmpKSkBtXU1rjtHjDxsMREePppmD4dPvoILr/c0xWJiIiICHDppZeyb98+Zs2axUMPPUSfPn0AGDVqFDt27GDkyJEEBARwzz33EBkZSVJSEr/73e9ISUnhpZdeAmDBggVcccUVnHXWWfz1r3/Fy8uL5ORkvvjiCwAiIyOZM2cOU6ZMYfTo0UybNg2A6OjoBtWam5vLv/71L6644gquvfZaDMNg3rx53HvvvWRlZfHII49U73vFFVewevVq7rrrLrp3705mZibLli1j1apVjBs3jttuu43s7Gw+++wznn32WSIiIgCqP397pQDWltxxB7z+OvzmN3D++eDv7+mKRERERNq9gQMHMmLECGbNmsU555xTqys1adIkAgMDWbNmDUFBQQDcdtttxMbG8txzz3HffffRo0cPvvzySwIDA/nuu+9wOI78CP/UU08B4O/vzzXXXMOUKVPo2rUr1113XaNqDQ0NZf/+/fj4+FRvu+eee7jxxhv5xz/+wR/+8Ae8vb3Jycnhxx9/5Omnn+aBBx6o81gjR46kf//+fPbZZ1xyySVNNvlIa6MA1pY4HPDSSzB6NDz+OFS1fEVERERao/vugzVrPF3FEYMHw3PPue942dnZzJ07l//7v/+jtLSU9PT06u9NnDiRZ599loULF9KjRw+Cg4MpKChg7ty5XHjhhe4r4ih2u716Svfy8nLy8vKoqKhg/PjxvPnmm2zdupUBAwbgdDrx9vYmKSmJm266ifDw8Carqa3RLIhtzZlnWveBPfMMbN/u6WpERERE5Di2bduGaZo888wzREZG1lrOO+88wLrPCuDOO++kd+/e/OpXv6JDhw5cc801vPvuu5SVlbm9rnfeeYchQ4bg6+tLWFgYkZGRXH/99QBkZWUB4O3tzdNPP828efOIiYlh1KhR/PnPf2bLli1ur6etUQesLXr6afjsM+t+sK+/tqarFxEREWll3Nltaolck2zcfvvtXHac57l27doVgKioKFavXs3ChQv59ttvmTdvHu+99x5PP/00ixcvdtuDuP/3v/9x/fXXc/7553PPPfcQExODt7c3q1at4sEHH6w1Mci9997L5MmT+fzzz1mwYAHPPvssTzzxBP/+97+58cYb3VJPW6QA1hbFxMCjj8L991uzI06e7OmKRERERNo1o45fiHfr1q16ZsAJEyac9BheXl5MnDiRiRMnAvDyyy9z11138e677zJt2rQ6z9FQ7777LgkJCXz55ZfYbEcGy+3cubPO/Tt37szdd9/N3XffTVZWFiNHjuSPf/xjdQBzR01tjYYgtlV33w0DB8Jdd0FVq1hEREREPMPVocqq8XNZZGQk48ePZ86cOezYseOY9+Tm5lJSUgJARkbGMd8fOnQoAJmZmYB1/5avr2+tczSU6/6vmlPOFxUV8cILL9Tar7CwkKKiolrbQkNDiY+Pr3X+uj53e6cOWFvlcMDs2XDaaVYnrOo5EyIiIiLS/BITEzEMgyeeeILs7GycTienn346r7zyCmeccQZDhgzhlltuoU+fPuTk5LBhwwY++ugjNmzYQHx8PLfccguHDx9m/PjxdO7cmfT0dGbOnInT+f/t3XtcVVXex/HPkouAoJKCON7wEt66qOEFu6FmlmamZeloaRmlVuYz83qy0gkqGzOdZpqG1ElUptRy8jKWjZo2WM84laVNRqZ5odSyUlPxkimu548FCAKKctlw+L5fr/U6+6yz9z4/VrstP9baawUzYMCA3O/p2LEjq1atYurUqTRs2JDIyEi6d+8OkDsLYUZGRpFx9u/fn4ULF9K7d28GDBjA/v37mTNnToEhjlu2bKFbt27cdttttGnThtDQUN577z1WrFjBPffcky8egMcff5zBgwcTGBhI9+7diYyMLKWWrXyUgPmyDh3gscdg4kQYOBD69PE6IhEREZEqqXnz5iQnJzN16lQSEhLIyspi9uzZDB8+nPXr1zNx4kQWLlzInj17CA8PJyYmhsTERKKiogAYOnQoKSkpvPzyy+zfv586deoQ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) 





#### 调参思路：

> param1是初始参数 param2是调之前的 param3是正在调的

三组进行对比

##### max_depth

- max_depth=2 ,绿线上升，蓝线下降

- max_depth=3 ,灰线下降，粉线上升，不行！

  

  ##### eta 步长（学习率）

- eta = 0.1 ,灰色先上升非常多，粉色线变化不大，非常好，继续往小了试

- eta = 0.05 ,灰色继续上移，粉色没啥变化，差不多ok了

##### gamma 曲线 ：

- gamma =10 ， 粉色和灰色被重叠覆盖，调不调无所谓

##### alpha 正则项系数

- alpha = 3 ,灰色线变化不大，粉色系往上，不调

##### colsample_bytree 抽样比例

- colsample_bytree = 0.3 ,粉色线往上太多， 不行
- colsample_bytree = 0.85 ，重叠无影响



到这里我们会发现，后面的参数作用微乎其微

**注意参数的顺序也是会影响的，先调max_depth和gamma是不一样的** 通常情况，先调eta和n_eastimators,调好这俩之后，再取掉剩下的这俩





#### 使用Pickle和joblib保存和调用模型

- 使用joblib

1.把pickle换成joblib 2.dump保存是不用open函数

```python
import pickle


dtrain = xgb.DMatrix(Xtrain,Ytrain)

param = {'slient':True
         ,'obj':'reg:linear'
         ,'subsample':1
         ,'max_depth':4
         ,'eta':0.05
         ,'gamma':20
         ,'lambda':3.5
         ,'alpha':0.2
         ,'colsample_bytree':0.4
         ,'colsample_bylevel':0.6
         ,'colsample_bynode':1
         ,'nfold':5}

num_round = 200

bst = xgb.train(param ,dfull ,num_round)

#dump(模型，文档)
pickle.dump(bst ,open("xgboostboston.dat","wb"))
joblib.dump(bst ,"xgboostboston.dat","wb")

#找一下树在哪
improt sys
sys.path

#下次使用，先导入数据，数据导入完了之后，就导模型
dtest = xgb.DMatrix(Xtest,Ytest)
load_model = pickle.load(open("xgboostboston.dat","rb"))
#导入完了直接使用
ypreds = load_model.predict(dtest)

#导入模型测试指标
from sklearn.metrics import mean_squared_error as MSE,r2_score

MSE(Ytest,y_pred)
r2_score(Ytest,y_pred)
```



### XGB做分类问题

> 参数含义： 控制正负样本比例， 表示为负/ 正样本比例 在样本不平衡问题中使用

xgb.train() : scale_pos_weight, 默认1 

xgb.XGBClassifier() : scale_pos_weight, 默认1

```python
#使用XGBOOST
dtrain = xgb.DMatrix(Xtrain,Ytrain)
dtest = xgb.DMatrix(Xtest,Ytest)
param ={'slient':True,'objective':'binary:logistic','eta':0.1,'scale_pos_weight':1}
num_round = 100

bst = xgb.train(param,dtrain,num_round)
preds = bst.predict(dtest)#这个返回指返回的分类的概率
preds


ypred = preds.copy()
ypred[preds>0.5] =1
#预测概率大于0.5的已经赋值成了1，剩下的不是1的就是小于0.5的啦
ypred[ypred!=1] =0
scale_pos_weight = [1,5,10]
names =["negative vs positive :1"
        ,"negative vs positive :5"
        ,"negative vs positive :10"]

for i in zip(names,scale_pos_weight):
    param ={"silent":True ,'objective':'binary:logistic'
           ,'eta':0.1,"scale_pos_weight":i}
    clf = xgb.train(param ,dtrain,num_round)
    preds = clf.predict(dtest)
    ypred = preds.copy()
    ypred[preds > 0.5] = 1
    ypred[preds != 1] = 0
    print(name)
    print(f"\tAccuracy:{accuracy(Ytest,ypred)}")
    print(f"\tRecall:{recall(Ytest,ypred)}")
    print(f"\tAUC:{auc(Ytest,preds)}")
```



## LightGBM

>  LightGBM是XGBoost的优化。
>
>  提出算法的原因：
>
>  GBDT在每一次迭代的时候，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。尤其面对工业级海量的数据，普通的GBDT算法是不能满足其需求的。 

### XGBoost的优缺点

**精确的贪心算法**

每轮迭代时，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。

* 优点：可以找到精确的划分条件
* 缺点： 1. 计算量巨大 2.内存占用巨大  3.容易产生过拟合

**Level-wise迭代方式**

预排序方法（pre-sorted）：首先，空间消耗大。这样的算法需要保存数据的特征值，还保存了特征排序的结果（例如排序后的索引，为了后续快速的计算分割点），这里需要消耗训练数据两倍的内存。其次时间上也有较大的开销，在遍历每一个分割点的时候，都需要进行分裂增益的计算，消耗的代价大。

优点：

- 可以使用多线程
- 可以加速精确贪心算法

缺点：

- 效率低下，可能产生不必要的叶结点



**对cache优化不友好**

在预排序后，特征对梯度的访问是一种随机访问，并且不同的特征访问的顺序不一样，无法对cache进行优化。同时，在每一层长树的时候，需要随机访问一个行索引到叶子索引的数组，并且不同特征访问的顺序也不一样，也会造成较大的cache miss。



### lightGBM主要有以下特点：

- 基于Histogram的决策树算法
- 带深度限制的Leaf-wise的叶子生长策略
- 直方图做差加速
- 直接支持类别特征(Categorical Feature)
- Cache命中率优化
- 基于直方图的稀疏特征优化
- 多线程优化



### Histogram algorithm

Histogram algorithm应该翻译为直方图算法，直方图算法的思想也很简单，首先将连续的浮点数据转换为bin数据，具体过程是首先确定对于每一个特征需要多少的桶bin，然后均分，将属于该桶的样本数据更新为bin的值，最后用直方图表示。（看起来很高大上，其实就是直方图统计，最后我们将大规模的数据放在了直方图中）



**直方图算法有几个需要注意的地方：**

- 使用bin替代原始数据相当于增加了正则化；
- 使用bin意味着很多数据的细节特征被放弃了，相似的数据可能被划分到相同的桶中，这样的数据之间的差异就消失了；
- bin数量选择决定了正则化的程度，bin越少惩罚越严重，欠拟合风险越高。

- 构建直方图时不需要对数据进行排序（比XGBoost快），因为预先设定了bin的范围；
- 直方图除了保存划分阈值和当前bin内样本数以外还保存了当前bin内所有样本的一阶梯度和（一阶梯度和的平方的均值等价于均方损失）；
- 阈值的选取是按照直方图从小到大遍历，使用了上面的一阶梯度和，目的是得到划分之后△loss最大的特征及阈值。



### 决策树生长策略

![1637655091191](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1637655091191.png)

 XGBoost采用的是按层生长level（depth）-wise生长策略，能够同时分裂同一层的叶子，从而进行多线程优化，不容易过拟合；但不加区分的对待同一层的叶子，带来了很多没必要的开销。因为实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。 



 LightGBM采用leaf-wise生长策略，每次从当前所有叶子中找到分裂增益最大（一般也是数据量最大）的一个叶子，然后分裂，如此循环。因此同Level-wise相比，在分裂次数相同的情况下，Leaf-wise可以降低更多的误差，得到更好的精度。Leaf-wise的缺点是可能会长出比较深的决策树，产生过拟合。因此LightGBM在Leaf-wise之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。 



### 直方图差加速：

 LightGBM另一个优化是Histogram（直方图）做差加速。一个容易观察到的现象：一个叶子的直方图可以由它的父亲节点的直方图与它兄弟的直方图做差得到。通常构造直方图，需要遍历该叶子上的所有数据，但直方图做差仅需遍历直方图的k个桶。利用这个方法，LightGBM可以在构造一个叶子的直方图后，可以用非常微小的代价得到它兄弟叶子的直方图，在速度上可以提升一倍。 



### 直接支持类别特征

 实际上大多数机器学习工具都无法直接支持类别特征，一般需要把类别特征，转化one-hotting特征，降低了空间和时间的效率。而类别特征的使用是在实践中很常用的。基于这个考虑，LightGBM优化了对类别特征的支持，可以直接输入类别特征，不需要额外的0/1展开。并在决策树算法上增加了类别特征的决策规则。 

one-hot编码是处理类别特征的一个通用方法，然而在树模型中，这可能并不一定是一个好的方法，尤其当类别特征中类别个数很多的情况下。主要的问题是：

- 可能无法在这个类别特征上进行切分（即浪费了这个特征）。使用one-hot编码的话，意味着在每一个决策节点上只能使用one vs rest（例如是不是狗，是不是猫等）的切分方式。当类别值很多时，每个类别上的数据可能会比较少，这时候切分会产生不平衡，这意味着切分增益也会很小（比较直观的理解是，不平衡的切分和不切分没有区别）。
- 会影响决策树的学习。因为就算可以在这个类别特征进行切分，也会把数据切分到很多零碎的小空间上，如图1左边所示。而决策树学习时利用的是统计信息，在这些数据量小的空间上，统计信息不准确，学习会变差。但如果使用下图右边的分裂方式，数据会被切分到两个比较大的空间，进一步的学习也会更好。

**LightGBM处理分类特征大致流程：**

为了解决one-hot编码处理类别特征的不足。LightGBM采用了Many vs many的切分方式，实现了类别特征的最优切分。用LightGBM可以直接输入类别特征，并产生上图右边的效果。在1个k维的类别特征中寻找最优切分，朴素的枚举算法的复杂度是O(2k)，而LightGBM采用了如[On Grouping For Maximum Homogeneity](http://www.csiss.org/SPACE/workshops/2004/SAC/files/fisher.pdf)的方法实现了O(klogk)的算法。



#### 算法流程

1.在枚举分割点之前，先把直方图按每个类别的均值进行排序；

2.然后按照均值的结果依次枚举最优分割点。

从下图可以看到，Sum(y)/Count(y)为类别的均值。当然，这个方法很容易过拟合，所以在LGBM中加入了很多对这个方法的约束和正则化。 

![1637655491551](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1637655491551.png)





### LightGBM调参

### 针对 leaf-wise 树的参数优化：

- num_leaves：控制了叶节点的数目。它是控制树模型复杂度的主要参数。
- 如果是level-wise，则该参数为2depth，其中depth为树的深度。但是当叶子数量相同时，leaf-wise的树要远远深过level-wise树，非常容易导致过拟合。因此应该让num_leaves小于2depth。在leaf-wise树中，并不存在depth的概念。因为不存在一个从leaves到depth的合理映射。
- min_data_in_leaf：每个叶节点的最少样本数量。它是处理leaf-wise树的过拟合的重要参数。将它设为较大的值，可以避免生成一个过深的树。但是也可能导致欠拟合。
- max_depth： 控制了树的最大深度。该参数可以显式的限制树的深度。

### 针对更快的训练速度：

- 通过设置 bagging_fraction 和 bagging_freq 参数来使用 bagging 方法
- 通过设置 feature_fraction 参数来使用特征的子抽样
- 使用较小的 max_bin
- 使用 save_binary 在未来的学习过程对数据加载进行加速

### 获取更好的准确率：

- 使用较大的 max_bin （学习速度可能变慢）
- 使用较小的 learning_rate 和较大的 num_iterations
- 使用较大的 num_leaves （可能导致过拟合）
- 使用更大的训练数据
- 尝试 dart

### 缓解过拟合：

- 使用较小的 max_bin
- 使用较小的 num_leaves
- 使用 min_data_in_leaf 和 min_sum_hessian_in_leaf
- 通过设置 bagging_fraction 和 bagging_freq 来使用 bagging
- 通过设置 feature_fraction 来使用特征子抽样
- 使用更大的训练数据
- 使用 lambda_l1, lambda_l2 和 min_gain_to_split 来使用正则
- 尝试 max_depth 来避免生成过深的树

### 核心参数：

- config 或者config_file：一个字符串，给出了配置文件的路径。默认为空字符串。
- task： 一个字符串，给出了要执行的任务。可以为：
  - ‘train’ 或者 ‘training’：表示是训练任务。默认为’train’。
  - ‘predict’ 或者 ‘prediction’或者’test’：表示是预测任务。
  - ‘convert_model’： 表示是模型转换任务。将模型文件转换成if-else 格式。

#### pplication或者objective或者app

- application或者objective或者app：一个字符串，表示问题类型。可以为：

  - ‘regression’或’regression_l2’或’mean_squared_error’或’mse’或’l2_root’或’root_mean_squred_error’或’rmse’：表示回归任务，但是使用L2损失函数。默认为’regression’
  - ‘regression_l1’或者mae或者mean_absolute_error：表示回归任务，但是使用L1损失函数。
  - ‘huber’： 表示回归任务，但是使用huber 损失函数。
  - ‘fair’： 表示回归任务，但是使用fair 损失函数。
  - ‘poisson’： 表示Poisson 回归任务。
  - ‘quantile’： 表示quantile回归任务。
  - ‘quantile_l2’：表示quantile回归任务，但是使用了L2 损失函数。
  - ‘mape’ 或者’mean_absolute_precentage_error’： 表示回归任务，但是使用MAPE 损失函数
  - ‘gamma’： 表示gamma 回归任务。
  - ‘tweedie’： 表示tweedie 回归任务。
  - ‘binary’： 表示二分类任务，使用对数损失函数作为目标函数。
  - ‘multiclass’： 表示多分类任务，使用softmax 函数作为目标函数。必须设置num_class 参数
  - ‘multiclassova’ 或者’multiclass_ova’ 或者’ova’ 或者’ovr’： 表示多分类任务，使用one-vs-all 的二分类目标函数。必须设置num_class 参数
  - ‘xentropy’ 或者’cross_entropy’： 目标函数为交叉熵（同时具有可选择的线性权重）。要求标签是[0,1] 之间的数值。
  - ‘xentlambda’ 或者’cross_entropy_lambda’： 替代了参数化的cross_entropy 。要求标签是[0,1]之间的数值。
  - ‘lambdarank’：表示排序任务。在lambdarank 任务中，标签应该为整数类型，数值越大表示相关性越高。label_gain 参数可以用于设置整数标签的增益（权重）

- boosting 或者’boost’ 或者 ‘boosting_type’： 一个字符串，给出了基学习器模型算法。可以为：

  - ‘gbdt’： 表示传统的梯度提升决策树。默认值为’gbdt’
  - ‘rf’： 表示随机森林。
  - ‘dart’： 表示带dropout 的gbdt
  - goss：表示Gradient-based One-Side Sampling 的gbdt

- data或者train或者train_data：

  一个字符串，给出了训练数据所在的文件的文件名。默认为空字符串。lightgbm将使用它来训练模型。

- valid或者test或者valid_data或者test_data：

  一个字符串，表示验证集所在的文件的文件名。默认为空字符串。lightgbm将输出该数据集的度量。如果有多个验证集，则用逗号分隔。

- num_iterations或者num_iteration或者num_tree或者num_trees或者num_round或者num_rounds或者num_boost_round 

  一个整数，给出了boosting的迭代次数。默认为 100。

  - 对于python/R包，该参数是被忽略的。对于python，使用train()/cv()的输入参数num_boost_round来代替。
  - 在内部，lightgbm对于multiclass 问题设置了num_class*num_iterations 棵树。

- learning_rate或者shrinkage_rate： 

  一个浮点数，给出了学习率。默认为1。在dart 中，它还会影响dropped trees 的归一化权重。

- num_leaves或者num_leaf：一个整数，给出了一棵树上的叶子数。默认为 31

- tree_learner或者tree：一个字符串，给出了tree learner，主要用于并行学习。 默认为’serial’。 可以为：

  - ‘serial’： 单台机器的tree learner
  - ‘feature’： 特征并行的tree learner
  - ‘data’： 数据并行的tree learner
  - ‘voting’： 投票并行的tree learner

- num_threads 或者num_thread 或者nthread：一个整数， 给出了lightgbm 的线程数。默认为OpenMP_default。

### 学习控制参数：

- max_depth： 

  一个整数，限制了树模型的最大深度，默认值为-1。如果小于0，则表示没有限制。

- min_data_in_leaf 或者 min_data_per_leaf 或者 min_data或者min_child_samples：

  一个整数，表示一个叶子节点上包含的最少样本数量。默认值为 20

- min_sum_hessian_in_leaf 或者 min_sum_hessian_per_leaf或者 min_sum_hessian 或者 min_hessian或者min_child_weight： 

  一个浮点数，表示一个叶子节点上的最小hessian 之和。（也就是叶节点样本权重之和的最小值） 默认为1e-3 。

- feature_fraction或者sub_feature或者colsample_bytree：

  一个浮点数，取值范围为[0.0,1.0]， 默认值为0。如果小于1.0，则lightgbm 会在每次迭代中随机选择部分特征。如0.8 表示：在每棵树训练之前选择80% 的特征来训练。

- feature_fraction_seed： 一个整数，表示feature_fraction 的随机数种子，默认为2。

- bagging_fraction 或者sub_row 或者 subsample：

  一个浮点数，取值范围为[0.0,1.0]， 默认值为0。如果小于1.0，则lightgbm 会在每次迭代中随机选择部分样本来训练（非重复采样）。如0.8 表示：在每棵树训练之前选择80% 的样本（非重复采样）来训练。

- bagging_freq 或者subsample_freq：

  一个整数，表示每bagging_freq 次执行bagging。如果该参数为0，表示禁用bagging。

- bagging_seed 或者 bagging_fraction_seed：

  一个整数，表示bagging 的随机数种子，默认为 3 。

- early_stopping_round 或者 early_stopping_rounds或者early_stopping：一个整数，默认为0。如果一个验证集的度量在early_stopping_round 循环中没有提升，则停止训练。如果为0则表示不开启早停。

- lambda_l1 或者reg_alpha： 一个浮点数，表示L1正则化系数。默认为0

- lambda_l2 或者reg_lambda： 一个浮点数，表示L2正则化系数。默认为0

- min_split_gain 或者min_gain_to_split： 

  一个浮点数，表示执行切分的最小增益，默认为0

- drop_rate： 一个浮点数，取值范围为[0.0,1.0]，表示dropout 的比例，默认为1。 该参数仅在dart 中使用

- skip_drop： 一个浮点数，取值范围为[0.0,1.0]，表示跳过dropout 的概率，默认为5。 该参数仅在dart 中使用

- max_drop： 一个整数，表示一次迭代中删除树的最大数量，默认为50。 如果小于等于0，则表示没有限制。 该参数仅在dart 中使用

- uniform_drop：一个布尔值，表示是否想要均匀的删除树，默认值为False。 该参数仅在dart 中使用

- xgboost_dart_mode： 一个布尔值，表示是否使用xgboost dart 模式，默认值为False。该参数仅在dart 中使用

- drop_seed： 一个整数，表示dropout 的随机数种子，默认值为 4。 该参数仅在dart 中使用

- top_rate： 一个浮点数，取值范围为[0.0,1.0]，表示在goss 中，大梯度数据的保留比例，默认值为2。该参数仅在goss 中使用

- other_rate： 一个浮点数，取值范围为[0.0,1.0]，表示在goss 中，小梯度数据的保留比例，默认值为1。该参数仅在goss 中使用

- min_data_per_group：一个整数，表示每个分类组的最小数据量，默认值为100。用于排序任务

- max_cat_threshold： 一个整数，表示category 特征的取值集合的最大大小。默认为 32 。

- cat_smooth： 一个浮点数，用于category 特征的概率平滑。默认值为 10。它可以降低噪声在category 特征中的影响，尤其是对于数据很少的类。

- cat_l2： 一个浮点数，用于category 切分中的L2 正则化系数。默认为 10 。

- top_k 或者 topk： 一个整数，用于投票并行中。默认为20 。将它设置为更大的值可以获得更精确的结果，但是会降低训练速度。

### 目标函数的参数：

- sigmoid： 一个浮点数，用sigmoid 函数的参数，默认为0。它用于二分类任务和lambdarank 任务。
- alpha： 一个浮点数，用于Huber 损失函数和Quantile regression ，默认值为0。它用于huber回归任务和Quantile 回归任务。
- fair_c： 一个浮点数，用于Fair 损失函数，默认值为0 。它用于fair 回归任务。
- gaussian_eta： 一个浮点数，用于控制高斯函数的宽度，默认值为0 。它用于regression_l1 回归任务和huber回归任务。
- posson_max_delta_step： 一个浮点数，用于Poisson regression 的参数，默认值为7 。它用于poisson 回归任务。
- scale_pos_weight： 一个浮点数，用于调整正样本的权重，默认值为0它用于二分类任务。
- boost_from_average： 一个布尔值，指示是否将初始得分调整为平均值（它可以使得收敛速度更快）。默认为True。它用于回归任务。
- is_unbalance或者unbalanced_set ： 一个布尔值，指示训练数据是否均衡的。默认为True。它用于二分类任务。
- max_position： 一个整数，指示将在这个NDCG 位置优化。默认为 20 。它用于lambdarank 任务。
- label_gain： 一个浮点数序列，给出了每个标签的增益。默认值为0,1,3,7,15,….它用于lambdarank 任务。
- num_class或者num_classes ： 一个整数，指示了多分类任务中的类别数量。默认为 1它用于多分类任务。
- reg_sqrt： 一个布尔值，默认为False。如果为True，则拟合的结果为：label−−−−√。同时预测的结果被自动转换为：pred2。它用于回归任务。

### 度量参数：

- metric：一个字符串，指定了度量的指标，默认为：对于回归问题，使用l2 ；对于二分类问题，使用binary_logloss；对于lambdarank 问题，使用ndcg。如果有多个度量指标，则用逗号, 分隔。
  - ‘l1’ 或者 mean_absolute_error或者 mae或者 regression_l1： 表示绝对值损失
  - ‘l2’ 或者mean_squared_error或者 mse或者 regression_l2或者 regression：表示平方损失
  - ‘l2_root’ 或者root_mean_squared_error或者 rmse：表示开方损失
  - ‘quantile’： 表示Quantile 回归中的损失
  - ‘mape’ 或者 ‘mean_absolute_percentage_error’ ：表示MAPE 损失
  - ‘huber’： 表示huber 损失
  - ‘fair’： 表示fair 损失
  - ‘poisson’： 表示poisson 回归的负对数似然
  - ‘gamma’： 表示gamma 回归的负对数似然
  - ‘gamma_deviance’： 表示gamma 回归的残差的方差
  - ‘tweedie’： 表示Tweedie 回归的负对数似然
  - ‘ndcg’： 表示NDCG
  - ‘map’ 或者’mean_average_precision’： 表示平均的精度
  - ‘auc’： 表示AUC
  - ‘binary_logloss’或者’binary’： 表示二类分类中的对数损失函数
  - ‘binary_error’： 表示二类分类中的分类错误率
  - ‘multi_logloss’或者 ‘multiclass’或者 ‘softmax’或者 ‘multiclassova’或者 ‘multiclass_ova’,或者’ova’或者 ‘ovr’： 表示多类分类中的对数损失函数
  - ‘multi_error’： 表示多分类中的分类错误率
  - ‘xentropy’或者’cross_entropy’： 表示交叉熵
  - ‘xentlambda’ 或者’cross_entropy_lambda’： 表示intensity 加权的交叉熵
  - ‘kldiv’或者’kullback_leibler’： 表示KL 散度
- metric_freq或者’output_freq’：一个正式，表示每隔多少次输出一次度量结果。默认为1。
- train_metric 或者training_metric或者 is_training_metric： 一个布尔值，默认为False。如果为True，则在训练时就输出度量结果。
- ndcg_at 或者 ndcg_eval_at 或者eval_at： 一个整数列表，指定了NDCG 评估点的位置。默认为1,2,3,4,5 。





### 调参实战

### 参数速查

| **xgb**                     | **lgb**                            | **xgb.sklearn**             | **lgb.sklearn**       |
| --------------------------- | ---------------------------------- | --------------------------- | --------------------- |
| booster=’gbtree’            | boosting=’gbdt’                    | booster=’gbtree’            | boosting_type=’gbdt’  |
| objective=’binary:logistic’ | application=’binary’               | objective=’binary:logistic’ | objective=’binary’    |
| max_depth=7                 | num_leaves=2**7                    | max_depth=7                 | num_leaves=2**7       |
| eta=0.1                     | learning_rate=0.1                  | learning_rate=0.1           | learning_rate=0.1     |
| num_boost_round=10          | num_boost_round=10                 | n_estimators=10             | n_estimators=10       |
| gamma=0                     | min_split_gain=0.0                 | gamma=0                     | min_split_gain=0.0    |
| min_child_weight=5          | min_child_weight=5                 | min_child_weight=5          | min_child_weight=5    |
| subsample=1                 | bagging_fraction=1                 | subsample=1.0               | subsample=1.0         |
| colsample_bytree=1.0        | feature_fraction=1                 | colsample_bytree=1.0        | colsample_bytree=1.0  |
| alpha=0                     | lambda_l1=0                        | reg_alpha=0.0               | reg_alpha=0.0         |
| lambda=1                    | lambda_l2=0                        | reg_lambda=1                | reg_lambda=0.0        |
| scale_pos_weight=1          | scale_pos_weight=1                 | scale_pos_weight=1          | scale_pos_weight=1    |
| seed                        | bagging_seed feature_fraction_seed | random_state=888            | random_state=888      |
| nthread                     | num_threads                        | n_jobs=4                    | n_jobs=4              |
| evals                       | valid_sets                         | eval_set                    | eval_set              |
| eval_metric                 | metric                             | eval_metric                 | eval_metric           |
| early_stopping_rounds       | early_stopping_rounds              | early_stopping_rounds       | early_stopping_rounds |
| verbose_eval                | verbose_eval                       | verbose                     | verbose               |



### 原生lightgbm调参格式

```python
import lightgbm as lgb
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
category_feature=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
cv_params = {
    'num_leaves': [13,14,15],
#     'max_depth': [-1,4,6,8],
#     'learning_rate': [0.07,0.08,0.09],
#     'n_estimators':[10,15,20],
#     'min_child_samples':[15,20,25],
#     'subsample':[0.4,0.5,0.6,0.7],
#     'colsample_bytree':[0.4,0.5,0.6,0.7],
#     'reg_alpha':[0,1,2,3,5,8],
#     'reg_lambda':[7,8,9,10],
#     'num_iterations':[30,40,50],
#     'min_data_in_leaf': [30, 50, 100, 300, 400],
#     'cat_smooth':[150,160,170,180,190]
}
# cv_params = {'learning_rate': [0.06,0.07,0.08,0.09]}
other_params = {
    'max_depth' : 4,
    'num_leaves': 15,
    'learning_rate': 0.07,
    'cat_smooth':180,
    'num_iterations':100,
    'colsample_bytree': 0.7,
    'subsample': 0.4,
    'reg_alpha':3,
    'reg_lambda':9,
}
model_lgb = lgb.LGBMRegressor(**other_params)
optimized_lgb = GridSearchCV(estimator=model_lgb, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=2)
optimized_lgb.fit(X, y, categorical_feature=category_feature)
print('参数的最佳取值：{0}'.format(optimized_lgb.best_params_))
print('最佳模型得分:{0}'.format(optimized_lgb.best_score_))
print(optimized_lgb.cv_results_['mean_test_score'])
print(optimized_lgb.cv_results_['params'])
```

### sklearn接口的lightgbm（鸢尾花）

```python
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 加载数据
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
# 创建模型，训练模型
gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)
# 测试机预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# 模型评估
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
# feature importances
print('Feature importances:', list(gbm.feature_importances_))
# 网格搜索，参数优化
estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)
```

### 原生形式lightgbm（鸢尾花）

```python
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
# 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
# 保存模型到文件
gbm.save_model('model.txt')
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# 评估模型
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
```



## 神经网络

### 基本原理：

 出了。 和今天的大部分模型比较起来，感知机的泛化能力比较弱，但支持向量机和神经网络都基于它的原理来建立。感知机 的原理就是我们在支持向量机中详细介绍过的，使用一条线性决策边界 来划分数据集，决策边界的上方 是一类数据（ ），决策边界的下方是另一类数据（ ）的决策过程。使用神经元，我们可以把它表示为：  

![1644063426373](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644063426373.png)



 不同的特征数据被输入后，我们通过神经键将它输入我们的神经元。每条神经键上对应不同的参数 ，因此特征数 据会经由神经键被匹配到一个参数向量 ，基于参数向量 算法可以求解出决策边界 ，然后由决策函 数 进行判断，最终预测出标签y并且将结果输出。其中函数 被称为“激活函数”，这是模拟人类的大脑 激活神经元的过程所命名的，其实本质就是决定了预测标签的输出会是什么内容的预测函数 

三个核心要点：

1. 每个输入的特征会被匹配到一个参数 ，我们都知道参数向量 中含有的参数数量与我们的特征数目是一致的， 在感知机中也是如此。也就是说，任何基于感知机的算法，必须至少要有参数向量 可求。 

2. 一个线性关系 ， 是由参数和输入的数据共同决定的。这个线性关系，往往就是我们的决策边界，或者它也可 以是多元线性回归，逻辑回归等算法的线性表达式
3.  3. 激活函数的结果，是基于激活函数本身，参数向量 和输入的数据一同计算出来的。也就是说，任何基于感知机 的算法，必须要存在一个激活函数 

![1644063556694](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644063556694.png)

计算一下：

![1644064068707](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644064068707.png)

第二层：

![1644064178698](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644064178698.png)

第三层：

![1644064196814](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644064196814.png)

第四层：

![1644064222509](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644064222509.png)

* 我们神经网络的每一层的结果之间的关系是嵌套，不是迭代。 



4. 



### 单层神经网络 

> 从直观来看，线性回归的网络结构明明有两层，为什么线性回归被叫做“单层神 经网络”呢？ 实际上，在描述神经网络的层数的时候，我们不考虑输入层。 输入层是每个神经网络都必须存在的一层，任意两个神经网络之间的不同之处就 在输入层之后的所有层。所以，我们把输入层之后只有一层的神经网络称为单层 神经网络。当然了，在有的深度学习课程或教材中，也会直接将所有层都算入其 中，将上述网络称为“两层神经网络”。因此，当出现“N层神经网络”的描述时，一 定要注意原作者是否将输入层考虑进去了 

```
#首先使用numpy来创建数据
import numpy as np
X = np.array([[0,0],[1,0],[0,1],[1,1]])
z_reg = np.array([-0.2, -0.05, -0.05, 0.1])

#定义实现简单线性回归的函数
def LinearR(x1,x2):
    w1, w2, b = 0.15, 0.15,-0.2 #给定一组系数w和b
    z = x1*w1 + x2*w2 + b #z是系数*特征后加和的结果
    return z

LinearR(X[:,0],X[:,1])

```



### 二分类单层神经网络：

> sigmoid与阶跃函数 在过去我们学习逻辑回归时，我们了解到sigmoid函数可以帮助我们将线性回归连续 型的结果转化为0-1之间的概率值，从而帮助我们将回归类算法转变为分类算法逻辑 回归。对于神经网络来说我们也可以使用相同的方法。首先先来复习一下Sigmoid函 数的的公式和性质： 

```
#重新定义数据中的标签
y_and = [0,0,0,1]
#根据sigmoid公式定义sigmoid函数
def sigmoid(z):
    return 1/(1 + np.exp(-z))
def AND_sigmoid(x1,x2):    
    w1, w2, b = 0.15, 0.15,-0.2 #给定的系数w和b不变
    z = x1*w1 + x2*w2 + b
    o = sigmoid(z) #使用sigmoid函数将回归结果转换到(0,1)之间
    y = [int(x) for x in o >= 0.5] 
    #根据阈值0.5，将(0,1)之间的概率转变为分类0和1
    return o, y
#o:sigmoid函数返回的概率结果
#y:对概率结果按阈值进行划分后，形成的0和1，也就是分类标签
o, y_sigm = AND_sigmoid(X[:,0],X[:,1])
o,y_sigm
```



### 为什么说神经网络是端到端的网络?

端到端学习(end-to-end)是一种解决问题的思路，与之对应的是多步骤解决问题，也就是将一个问题拆分为多个步骤分步解决，而端到端是由输入端的数据直接得到输出端的结果。

就是不要预处理和特征提取，直接把原始数据扔进去得到最终结果。

**特征提取包含在神经网络内部**，所以说神经网络是端到端的网络。

**优点**：

通过缩减人工预处理和后续处理，尽可能使模型从原始输入到最终输出，给模型更多可以根据数据自动调节的空间，增加模型的整体契合度。

**缺点**

- 它可能需要大量的数据。要直接学到这个𝑥到𝑦的映射，你可能需要大量(𝑥, 𝑦)数据。
- 它排除了可能有用的手工设计组件。

### 什么是softmax

在图像分类的情景中，softmax分类器输出可以是一个图像类别的离散值，和线性回归不同的是，**softmax输出单元从一个变成了多个。**

softmax回归和线性回归一样将输入特征与权重做线性叠加。与线性回归的一个主要不同在于，**softmax回归的输出值个数等于标签里的类别数。**下图是用神经网络描绘了softmax回归，也是一个单层神经网络，由于每个输出 [![img](https://camo.githubusercontent.com/2ba6a49a6d62d08765e34954bf04cdbb3778b72db299c90b58c66abca3009524/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f312c6f5f322c6f5f33)](https://camo.githubusercontent.com/2ba6a49a6d62d08765e34954bf04cdbb3778b72db299c90b58c66abca3009524/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f312c6f5f322c6f5f33)的计算都要依赖于所有的输入 [![img](https://camo.githubusercontent.com/f7d753d2411958974e7b1978bdfb54791a543c9ef57a7fff827c806bcd6639ba/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f785f312c785f322c785f332c785f34)](https://camo.githubusercontent.com/f7d753d2411958974e7b1978bdfb54791a543c9ef57a7fff827c806bcd6639ba/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f785f312c785f322c785f332c785f34)，softmax回归的输出层也是一个全连接层。

[![img](https://camo.githubusercontent.com/3a0b736664b026b570eb8b3bfbda5ba08aa22c52fc72ffdef1e03d7b50082e58/687474703a2f2f7778342e73696e61696d672e636e2f6d773639302f30303633304465666c793167356c706f706d3163336a3330683930363537347a2e6a7067)](https://camo.githubusercontent.com/3a0b736664b026b570eb8b3bfbda5ba08aa22c52fc72ffdef1e03d7b50082e58/687474703a2f2f7778342e73696e61696d672e636e2f6d773639302f30303633304465666c793167356c706f706d3163336a3330683930363537347a2e6a7067)

[![img](https://camo.githubusercontent.com/886ca2c3dc35f77788aa907872df8b209fd2c2182b4ada61cbf17eb1836895dd/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f313d785f31775f25374231312537442b785f32775f25374232312537442b785f33775f25374233312537442b785f34775f2537423431253744)](https://camo.githubusercontent.com/886ca2c3dc35f77788aa907872df8b209fd2c2182b4ada61cbf17eb1836895dd/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f313d785f31775f25374231312537442b785f32775f25374232312537442b785f33775f25374233312537442b785f34775f2537423431253744)

[![img](https://camo.githubusercontent.com/4bf058a2a1a5acbcda60f0f429d2a57a04c09c7af62dfc5720b29aac29a4321d/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f323d785f31775f25374231322537442b785f32775f25374232322537442b785f33775f25374233322537442b785f34775f2537423432253744)](https://camo.githubusercontent.com/4bf058a2a1a5acbcda60f0f429d2a57a04c09c7af62dfc5720b29aac29a4321d/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f323d785f31775f25374231322537442b785f32775f25374232322537442b785f33775f25374233322537442b785f34775f2537423432253744)

[![img](https://camo.githubusercontent.com/0093b91ef0d234cba6e3973f43ed778b601e4c0e8fa5bc6d106ccf2215eda1d5/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f333d785f31775f25374231332537442b785f32775f25374232332537442b785f33775f25374233332537442b785f34775f2537423433253744)](https://camo.githubusercontent.com/0093b91ef0d234cba6e3973f43ed778b601e4c0e8fa5bc6d106ccf2215eda1d5/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f333d785f31775f25374231332537442b785f32775f25374232332537442b785f33775f25374233332537442b785f34775f2537423433253744)

[![img](https://camo.githubusercontent.com/e0f108f28fdf52d81c6d60ba91c03e04e00c7081b5a92b920ff083089361f5d8/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f343d785f31775f25374231342537442b785f32775f25374232342537442b785f33775f25374233342537442b785f34775f2537423434253744)](https://camo.githubusercontent.com/e0f108f28fdf52d81c6d60ba91c03e04e00c7081b5a92b920ff083089361f5d8/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f343d785f31775f25374231342537442b785f32775f25374232342537442b785f33775f25374233342537442b785f34775f2537423434253744)

### softmax的计算

一个简单的办法是将输出值 [![img](https://camo.githubusercontent.com/3ef1f1de0451ad1f2d3bbec87d66a21ccb3fa4b7f19b9261efae3a0f76902374/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f69)](https://camo.githubusercontent.com/3ef1f1de0451ad1f2d3bbec87d66a21ccb3fa4b7f19b9261efae3a0f76902374/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f69)当做预测类别是i的置信度，并将值最大的输出所对应的类别作为预测输出。例如，如果 [![img](https://camo.githubusercontent.com/2ba6a49a6d62d08765e34954bf04cdbb3778b72db299c90b58c66abca3009524/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f312c6f5f322c6f5f33)](https://camo.githubusercontent.com/2ba6a49a6d62d08765e34954bf04cdbb3778b72db299c90b58c66abca3009524/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f312c6f5f322c6f5f33)分别为0.1 ; 10 ; 0.1，由于 [![img](https://camo.githubusercontent.com/57ee192ff4e6aaa10794f7320b66c1c128a2af89f2134f7814ab40827db4394e/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f32)](https://camo.githubusercontent.com/57ee192ff4e6aaa10794f7320b66c1c128a2af89f2134f7814ab40827db4394e/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6f5f32)最大，那么预测类别为2。

 **我们只需要看 最大的那一类就 可以了。 **

然而，直接使用输出层的输出会有两个问题：

- 由于输出层的输出值的范围不确定，我们难以直观上判断这些值得意义。
- 由于真实标签是离散值，这些离散值与不确定范围的输出值之间的误差难以衡量。

softmax运算解决了以上两个问题。它通过下面的公式将输出值变换成值为正且和为1的概率分布：

[![img](https://camo.githubusercontent.com/72b23bff233b7614a91e2f67153c4324840780c2046ac2a1b20e7166b4d6220c/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f736f66746d6178286f5f69293d25354366726163253742657870286f5f692925374425374225354373756d5f253742693d312537442535452537426e253744657870286f5f6929253744)](https://camo.githubusercontent.com/72b23bff233b7614a91e2f67153c4324840780c2046ac2a1b20e7166b4d6220c/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f736f66746d6178286f5f69293d25354366726163253742657870286f5f692925374425374225354373756d5f253742693d312537442535452537426e253744657870286f5f6929253744)

```
#定义softmax函数
def softmax(z):
    c = np.max(z)
    exp_z = np.exp(z - c) #溢出对策
    sum_exp_z = np.sum(exp_z)
    o = exp_z / sum_exp_z
    return o
#导入刚才定义的z
softmax(z)
```



![1644066329455](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644066329455.png)





### 交叉熵损失函数

我们已经知道，softmax运算将输出变换成一个合法的类别预测分布。实际上，真实标签也可以用类别分布表达：

对于样本i，我们构造向量 [![img](https://camo.githubusercontent.com/6064be0afe502d4ab6c428e8be9d7f64dc32ce14c64b392ec4817e5a767b128f/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f79253545253742286929253744253543696e5f2537422537445225354571)](https://camo.githubusercontent.com/6064be0afe502d4ab6c428e8be9d7f64dc32ce14c64b392ec4817e5a767b128f/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f79253545253742286929253744253543696e5f2537422537445225354571)，使其第 [![img](https://camo.githubusercontent.com/5e01056ffe2f72ced7bcccd420136f01a439c3b2e018843758e1c08eb2e8ad4d/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f79253545253742286929253744)](https://camo.githubusercontent.com/5e01056ffe2f72ced7bcccd420136f01a439c3b2e018843758e1c08eb2e8ad4d/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f79253545253742286929253744)个元素为1，其余为0。这样我们的训练目标可以设为使预测概率分布 [![img](https://camo.githubusercontent.com/4c9dc7adbace6873a70aece4e39e3a2cd83f2453a0d111a577fa1be8d1e8f0ed/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f25354368617425374279253744253545253742286929253744)](https://camo.githubusercontent.com/4c9dc7adbace6873a70aece4e39e3a2cd83f2453a0d111a577fa1be8d1e8f0ed/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f25354368617425374279253744253545253742286929253744)尽可能接近真实的标签概率 [![img](https://camo.githubusercontent.com/5e01056ffe2f72ced7bcccd420136f01a439c3b2e018843758e1c08eb2e8ad4d/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f79253545253742286929253744)](https://camo.githubusercontent.com/5e01056ffe2f72ced7bcccd420136f01a439c3b2e018843758e1c08eb2e8ad4d/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f79253545253742286929253744)。

想要预测分类结果正确，**我们其实并不需要预测概率完全等于标签概率**，而平方损失则过于严格。改善这个问题的一个方法是使用更适合衡量两个概率分布差异的测量函数。其中，**交叉熵(cross entropy)是一个常用的衡量方法：**

[![img](https://camo.githubusercontent.com/a8f8c92ddc73cd0e6b2e6b06a81f17d9b98f2633864813e3d3010aab1c72d830/687474703a2f2f7778332e73696e61696d672e636e2f6d773639302f303036333044656667793167356e6d6d336a7a61706a3330653230316f3734392e6a7067)](https://camo.githubusercontent.com/a8f8c92ddc73cd0e6b2e6b06a81f17d9b98f2633864813e3d3010aab1c72d830/687474703a2f2f7778332e73696e61696d672e636e2f6d773639302f303036333044656667793167356e6d6d336a7a61706a3330653230316f3734392e6a7067)

其中带下标的 [![img](https://camo.githubusercontent.com/feb4b63e62b6398de2ecdfcba03705fdf806aff39f2d42331b0345b0a8ab8e8c/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f795f6a253545253742286929253744)](https://camo.githubusercontent.com/feb4b63e62b6398de2ecdfcba03705fdf806aff39f2d42331b0345b0a8ab8e8c/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f795f6a253545253742286929253744)是向量 [![img](https://camo.githubusercontent.com/5e01056ffe2f72ced7bcccd420136f01a439c3b2e018843758e1c08eb2e8ad4d/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f79253545253742286929253744)](https://camo.githubusercontent.com/5e01056ffe2f72ced7bcccd420136f01a439c3b2e018843758e1c08eb2e8ad4d/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f79253545253742286929253744)中非 0 即 1 的元素。也就是说，**交叉熵只关心对正确类别的预测概率**，因为只要其值足够大，就可以确保分类结果正确。**即最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。**



### 框架对比

| 平台       | 优点                                                         | 缺点                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| TensorFlow | 1.功能很齐全，能够搭建的网络更丰富。 2.支持多种编程语言。 3.拥有强大的计算集群。 4.谷歌支持 5.社区活跃度高。 6.支持多GPU。 7.TensorBoard支持图形可视化。 | 1.编程入门难度较大。 2.计算图是纯 Python 的，因此速度较慢 3.图构造是静态的，意味着图必须先被「编译」再运行 |
| Keras      | 1.Keras是TensorFlow高级集成APi 2.Keras是一个简洁的API。 可以快速帮助您创建应用程序。 3.代码更加可读和简洁。 4.Keras处于高度集成框架。 5.社区活跃。 | 1.Keras框架环境配置比其他底层框架要复杂一些。 2.虽然更容易创建模型，但是面对复杂的网络结构时可能不如TensorFlow。 3.性能方面比较欠缺。 |
| Pytorch    | 1.它可以在流程中更改体系结构。 2.训练神经网络的过程简单明了。 3.可以使用标准 Python 语法编写 for 循环语句。 4.大量预训练模型 | 1.不够TensorFlow全面，不过未来会弥补。 2.PyTorch部署移动端不是很好。 |
| MXNet      | 1.支持多语言。 2.文档齐全。 3.支持多个GPU。 4.清晰且易于维护的代码。 5.命令式和符号式编程风格之间进行选择。 | 1.不被广泛使用。 2.社区不够活跃。 3.学习难度大一些。         |

### 正向传播

正向传播(forward-propagation)是指对神经网络沿着从输入层到输出层的顺序，依次计算并存储模型的中间变量(包括输出)。

逻辑回归的计算步骤：所以回想当时我们讨论逻辑回归的时候，我们有这个正向传播步骤，其中我们计算𝑧，然后𝑎，然后损失函数𝐿。 正向传播类似，计算，最后得到**loss function**。

![1644035271212](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644035271212.png)

### 反向传播(BP)

反向传播(back-propagation)指的是计算神经网络参数梯度的方法。总的来说，反向传播依据微积分中的链式法则，沿着从输出层到输入层的顺序，依次计算并存储目标函数有关神经网络各层的中间变量以及参数的梯度。

由正向传播经过所有的隐藏层到达输出层，会得到一个输出结果[![img](https://camo.githubusercontent.com/4abfed4418ef5ba246c0eadcbe116bc6c7767d69c7eb9d1da57bde096a20dd50/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f4f5f4c)](https://camo.githubusercontent.com/4abfed4418ef5ba246c0eadcbe116bc6c7767d69c7eb9d1da57bde096a20dd50/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f4f5f4c)，然后根据这个[![img](https://camo.githubusercontent.com/4abfed4418ef5ba246c0eadcbe116bc6c7767d69c7eb9d1da57bde096a20dd50/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f4f5f4c)](https://camo.githubusercontent.com/4abfed4418ef5ba246c0eadcbe116bc6c7767d69c7eb9d1da57bde096a20dd50/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f4f5f4c)带入**loss funcation**中，利用SGD算法进行最优化求解，其中每次梯度下降都会使用一次BP来更新各个网络层中的参数值，这就是BP回传误差的意思。

![1644035298775](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644035298775.png)

- 正向传播求损失，BP回传误差。
- 根据误差信号修正每层的权重。对各个w进行求导，然后更新各个w。
- **链式依赖损失函数**：[![img](https://camo.githubusercontent.com/fb184c9a3d7239b875cf66ad55512ed2d703150fcd6a91749a8620df16c15567/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f792535452537422d2537443d68286728662878292929)](https://camo.githubusercontent.com/fb184c9a3d7239b875cf66ad55512ed2d703150fcd6a91749a8620df16c15567/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f792535452537422d2537443d68286728662878292929)



### 随机梯度下降法(SGD)

#### 3.6.1 mini-batch梯度下降

你可以把训练集分割为小一点的子集训练，这些子集被取名为 **mini-batch**，假设每一个子集中只有 1000 个样本，那么把其中的𝑥 (1)到𝑥 (1000)取出来，将其称为第一个子训练集，也叫做 **mini-batch**，然后你再取出接下来的 1000 个样本，从𝑥 (1001)到𝑥 (2000)，然后再取 1000个样本，以此类推。

在训练集上运行 **mini-batch** 梯度下降法，你运行 for t=1……5000，因为我们有5000个各有 1000 个样本的组，在 **for** 循环里你要做得基本就是对𝑋 {𝑡}和𝑌 {𝑡}执行一步梯度下降法。

- batch_size=1，就是SGD。
- batch_size=n，就是mini-batch
- batch_size=m，就是batch

其中1<n<m，m表示整个训练集大小。

**优缺点：**

- batch：相对噪声低些，幅度也大一些，你可以继续找最小值。
- SGD：大部分时候你向着全局最小值靠近，有时候你会远离最小值，因为那个样本恰好给你指的方向不对，因此随机梯度下降法是有很多噪声的，平均来看，它最终会靠近最小值，不过有时候也会方向错误，因为随机梯度下降法永远不会收敛，而是会一直在最小值附近波动。一次性只处理了一个训练样本，这样效率过于低下。
- mini-batch：实践中最好选择不大不小的 **mini-batch**，得到了大量向量化，效率高，收敛快。

首先，如果训练集较小，直接使用 **batch** 梯度下降法，这里的少是说小于 2000 个样本。一般的 **mini-batch** 大小为 64 到 512，考虑到电脑内存设置和使用的方式，如果 **mini-batch** 大小是 2 的𝑛次方，代码会运行地快一些。

#### 3.6.2 调节 Batch_Size 对训练效果影响到底如何？

1. Batch_Size 太小，模型表现效果极其糟糕(error飙升)。
2. 随着 Batch_Size 增大，处理相同数据量的速度越快。
3. 随着 Batch_Size 增大，达到相同精度所需要的 epoch 数量越来越多。
4. 由于上述两种因素的矛盾， Batch_Size 增大到某个时候，达到时间上的最优。
5. 由于最终收敛精度会陷入不同的局部极值，因此 Batch_Size 增大到某些时候，达到最终收敛精度上的最优。



### 实战代码

#### 神经网络的参数调整:

![1644064333292](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644064333292.png)

1. 激活函数：就如同核函数一样，必然有各种各样的激活函数可以供我们选择 

2. 神经网络的结构：隐藏层有多少，每层有多少神经元？必然是一个超参数，需要我们自己来进行调整 

3. 怎么确定：之前一直说得非常理所当然，每个神经键上都有一个参数 ，然而这些参数从哪里来的呢？在SVM 中和逻辑回归中，我们写了十页数学计算来求解我们的 ，在神经网络中，求解参数必然也不会轻松。是不是要 最优化呢？如果要最优化的话，是不是有损失函数呢？有损失函数的话，是不是有梯度下降呢？如果我们使用 的是类似于梯度下降，最小二乘这样的数学过程，则必然有众多的数学过程相关的参数在sklearn中等待着我 们。

4. 除此之外，我们还想知道一些其他的问题：比如说，神经网络为什么要使用嵌套的方式进行？为什么需要多 层，一层不行吗？ 

![1644066906743](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644066906743.png)

*  neural_network.MLPClassifier 

参数列表：

```
sklearn.neural_network.MLPClassifier (
hidden_layer_sizes=(100, ), 
activation=’relu’,
solver=’adam’,
alpha=0.0001, 
batch_size=’auto’, 
learning_rate=’constant’, 
learning_rate_init=0.001,
power_t=0.5, 
max_iter=200,
shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
n_iter_no_change=10)
```

#### 案例

```python
import numpy as np
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score as cv
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split as TTS
from time import time
import datetime
data = load_breast_cancer()
X = data.data
y = data.target
Xtrain, Xtest, Ytrain, Ytest = TTS(X,y,test_size=0.3,random_state=420)

times = time()
dnn = DNN(hidden_layer_sizes=(100,),random_state=420)
print(cv(dnn,X,y,cv=5).mean())
print(time() - times)
#使用决策树进行一个对比
times = time()
clf = DTC(random_state=420)
print(cv(clf,X,y,cv=5).mean())
print(time() - times)

dnn = DNN(hidden_layer_sizes=(100,),random_state=420).fit(Xtrain,Ytrain)
dnn.score(Xtest,Ytest)

#可见，默认层数是三层，由于必须要有输入和输出层，所以默认其实就只有一层隐藏层
#如果增加一个隐藏层上的神经元个数，会发生什么呢？
dnn = DNN(hidden_layer_sizes=(200,),random_state=420)
dnn = dnn.fit(Xtrain,Ytrain)
dnn.score(Xtest,Ytest)

#来试试看学习曲线
s = []
for i in range(100,2000,100):
    dnn = DNN(hidden_layer_sizes=(int(i),),random_state=420).fit(Xtrain,Ytrain)
    s.append(dnn.score(Xtest,Ytest))
print(i,max(s))
plt.figure(figsize=(20,5))
plt.plot(range(200,2000,100),s)
plt.show()

#那如果增加隐藏层，控制神经元个数，会发生什么呢？
s = []
layers = [(100,),(100,100),(100,100,100),(100,100,100,100),(100,100,100,100,100),
(100,100,100,100,100,100)]
for i in layers:
    dnn = DNN(hidden_layer_sizes=(i),random_state=420).fit(Xtrain,Ytrain)
    s.append(dnn.score(Xtest,Ytest))
print(i,max(s))
plt.figure(figsize=(20,5))
plt.plot(range(3,9),s)
plt.xticks([3,4,5,6,7,8])
plt.xlabel("Total number of layers")
plt.show()
```

