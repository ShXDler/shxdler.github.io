# 4.1 基本流程

采用分治（divide-and-conquer）策略。

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210405133039.png" alt="image-20210405133031723" style="zoom:80%;" /></div>

可见决策树是一个递归过程，三种情况会结束递归生成叶子节点：（1）当前节点样本全部属于同一个属性、（2）当前属性集为空集或所有样本在所有属性上取值相同、（3）当前节点包含样本集合为空集。其中（2）利用当前节点的后验分布，（3）是把父节点的样本分布作为当前节点的先验分布。

# 4.2 划分选择

决策树关键在于选择最优划分属性，生成“纯度”（purity）尽可能高的样本。

## 4.2.1 信息增益

“信息熵”（information entropy）定义为：

$$Ent(D)=-\sum_{k=1}^{|y|}p_k\log_2p_k$$

$Ent(D)$越小，纯度越高。

使用属性$a$进行划分后，减少的信息量称为“信息增益”（information gain）：

$$Gain(D,a)=Ent(D)-\sum^V_{v=1}\frac{|D^v|}{|D|}Ent(D^v)$$

信息增益越大，得到的“纯度提升”越大。ID3决策树学习算法就是以信息增益为准则选择划分属性。

## 4.2.2 增益率

不难发现，信息增益会偏向分类更多的属性，可能不利于模型的泛化能力。C4.5决策树算法使用了“增益率”（gain ratio）进行划分：

$$Gain_ratio(D,a)=\frac{Gain(D,a)}{IV(a)}$$

$$IV(a)=-\sum^V_{v=1}\frac{|D^v|}{|D|}\log_2\frac{|D^v|}{|D|}$$

$IV$称$a$的“固有值”（intrinsic value），实际上就是$D$在属性$a$的信息量。

增益率又可能会偏向分类较少的属性，C4.5使用了一种启发式方法：先从候选属性中挑选出信息增益高于平均值的，再选择增益率最大的。

## 4.2.3 基尼系数

CART（classification and regression tree）决策树使用“基尼系数”（Gini index）进行属性选择，基尼值公式如下：

$$Gini(D)=\sum^{|y|}_{k=1}\sum_{k'\ne k}p_kp_{k'}=\sum_{k=1}^{|y|}p_k(1-p_k)=1-\sum^{|y|}_{k=1}p_k^2$$

基尼值反映了随机抽选两个样本属性不同的概率，越小代表纯度越高。

计算属性$a$的基尼系数（$a$分成的所有类基尼值的加权平均）：

$$Gini\_index(D,a)=\sum^V_{v=1}\frac{|D^v|}{|D|}Gini(D^v)$$

# 4.3 剪枝处理

剪枝（pruning）可以减轻过拟合现象，分“预剪枝”（prepruning）和“后剪枝”（postpruning）两种：预剪枝是指在生成过程中，如果当前节点不能带来泛化性能的提升，则停止划分；后剪枝是指训练结束后，自下而上地搜索，如果将某一子树替换成叶子节点可以带来泛化性能的提升则进行更换。

下文以该模型为例：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210405143545.png" alt="image-20210405143545274" style="zoom:80%;" /></div>

## 4.3.1 预剪枝

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210405143925.png" alt="image-20210405143925635" style="zoom:80%;" /></div>

如图所示，每次划分时比较前后划分精度是否有提升。预剪枝的计算开销较小，但是它剪掉的节点虽然当前效果不好，但是其后续的划分可能会提升泛化性能，所以预剪枝可能会出现欠拟合的现象。

## 4.3.2 后剪枝

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210405144554.png" alt="image-20210405144554714" style="zoom:80%;" /></div>

后剪枝从下而上遍历节点，如果替换成叶子节点后可以提升效果，则进行替换。后剪枝的时间开销要比未剪枝和预剪枝都要大很多，但是其泛化能力也往往较好。

# 4.4 连续与缺失值

## 4.4.1 连续值处理

连续属性的划分常用二分法（bi-partition），将属性$a$所有的取值$\{a^1,a^2,...a^n\}$进行排序，从划分点集合中选择最优划分点：

$${T_a=\{\frac{a^i+a^{i+1}}2|i\le i\le n-1\}}$$

选取标准为：

$$Gain(D,a)=\max_{t\in T_a}Gain(D,a,t)=Ent(D)-\min_{t\in T_a}\sum_{\lambda\in\{-,+\}}\frac{|D^\lambda_t|}{|D|}Ent(D^\lambda_t)$$

当前节点如果划分的是连续属性，该属性还可以在后代节点中继续充当划分属性。

## 4.4.2 缺失值处理

出现缺失值时，应当解决两个问题：（1）如何在属性值缺失的情况下进行变量划分的选择（2）如果样本在给定属性值上确实，该如何进行划分。

对问题（1），给定训练集$D$和属性$a$，记$\tilde D$为$D$中在$a$上没有缺失的子集，对属性$a$的$V$个取值，记$\tilde {D^v}$为属性$a$取值为$v$的子集，记$\tilde {D_k}$为取值为$k$的样本子集，假定每个样本$x$有一个权重$w_x$，定义

$$\rho=\frac{\sum_{x\in\tilde D}w_x}{\sum_{x\in D}w_x}$$

$$\tilde{p_k}=\frac{\sum_{x\in\tilde {D_k}}w_x}{\sum_{x\in\tilde D}w_x}$$

$$\tilde{r_v}=\frac{\sum_{x\in\tilde {D^v}}w_x}{\sum_{x\in\tilde D}w_x}$$

直观来看$\rho$代表无缺失值所占权重比例，其余两个变量表示无缺失值样本中某类所占权重比例。

推广信息增益公式得到：

$$Gain(D,a)=\rho\times Gain(\tilde D,a)=\rho\times(Ent(\tilde D)-\sum^V_{v=1}\tilde{r_v}End(\tilde{D^v}))$$

$$Ent(\tilde{D})=-\sum^{|\mathcal Y|}_{k=1}\tilde{p_k}\log_2\tilde{p_k}$$

对问题（2），如果$x$在属性$a$上的取值未知，则将其同时划入所有子节点，并将其权重在对应的子节点中改为$\tilde{r_v}\cdot w_x$。C4.5即使用了如上算法。

# 4.5 多变量决策树

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210405200619.png"width=500 /></div>

如果把每个属性对应到坐标轴上，那么传统单变量决策树（univariate decision tree）划分的路径就是与坐标轴平行的线段。而多变量决策树（multivariate decision tree）可以实现“斜划分”或更复杂的划分方式，它们的非叶子节点是一个形如$\sum^d_{i=1}w_ia_i=t$的线性分类器，例如：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210405201001.png" alt="image-20210405201001866" style="zoom:80%;" /></div>

# 4.6 扩展阅读

信息增益和基尼系数实际上性能几乎一致。

剪枝可以在数据带有噪声时有效提高泛化性能。

多变量决策树算法主要有OC1，先寻找每个属性的最优权重，再对分类边界进行随机扰动以发现更好的边界。同时也有方法引入了最小二乘法、神经网络等结构。

有些决策树算法实现了“增量学习”，在增强新样本时只需调整原有模型而不用从头学习。