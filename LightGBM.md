# LightGBM

>  LightGBM是XGBoost的优化。
>
> 提出算法的原因：
>
>  GBDT在每一次迭代的时候，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。尤其面对工业级海量的数据，普通的GBDT算法是不能满足其需求的。 

## XGBoost的优缺点

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





## LightGBM调参

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





## 调参实战

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

