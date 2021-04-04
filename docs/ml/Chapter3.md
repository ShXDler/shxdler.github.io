# 3.1 基本形式

$$f(x)=w^\top x+b$$

# 3.2 线性回归

这里直接考虑“多元线性回归”（multivariate linear regression）对模型$f(X)=w^\top x+b$，考虑其均方误差：

$$MSE(f(X))=(f(X)-y)^\top(f(X)-y)$$

均方误差对应常用的“欧氏距离”（Euclidean distance），基于均方误差最小化得到的方法称为“最小二乘法”（least square method）。将其分别对$w$和$b$求偏导得：

$$\frac{\partial MSE}{\partial w}=2X^\top(w^\top x-y)$$

当$X$列满秩时，有

$$\hat w=(X^\top X)^{-1}X^\top y,\hat b=y-w^\top x$$

可以证明上述解使得MSE函数取唯一最小值。

而$X$并非列满秩时，我们可以引入正则化（regularization）机制。

另外，我们可以对因变量进行对数变换处理得到

$$\log y=w^\top x+b$$

实际上，上述对数线性回归（log-linear regression）是Box-Cox变换时$\lambda=0$的特例，可以使因变量变换为正态分布，满足Gauss-Markov假设。

更一般的，考虑单调可微函数$g(\cdot)$对$y$进行变换，这样的模型也称广义线性模型（generalized linear model），$g(\cdot)$也称连接函数（link function）。

# 3.3 logistic回归

对广义线性模型，考虑一个能把$[0,1]$映射到$[-\infty,+\infty]$上的logit函数$\log \frac y{1-y}$，其中$y$反映概率，比值代表概率的优势，有回归方程

$$\log \frac{p(y=1|x)}{p(y=0|x)}=w^\top x+b$$

可通过极大似然估计（maximum likelihood estimation）对$w$和$b$进行估计。

首先得到logistic模型的对数似然函数：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404214410.png" alt="image-20210404214410127" style="zoom:80%;" /></div>

得到

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404214622.png" alt="image-20210404214622430" style="zoom:80%;" /></div>

使用牛顿迭代法，有

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404214719.png" alt="image-20210404214719036" style="zoom:80%;" /></div>

# 3.4 线性判别分析

线性判别分析（Linear Discriminant Analysis，LDA）的思想是给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能近，再根据投影点的位置确认样本类别。

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404215344.png" alt="image-20210404215344437" style="zoom:80%;" /></div>

给定数据集$D$，令$X_i$、$\mu_i$、$\Sigma_i$代表第$i$类的自变量、均值向量和协方差矩阵。若将数据投影到直线$w$上，则两类样本的中心的投影为$w^\top \mu_i$，协方差投影为$w^\top \Sigma_iw$。为使同类样本尽可能近，可以让同类样例投影点的协方差尽可能小，同时要让异类样本尽可能远离，可以让它们的中心之间距离尽可能大，得到目标：

$$J=\frac{||w^\top\mu_0-w^\top\mu_1||^2_2}{w^\top \Sigma_0w+w^\top \Sigma_1w}=\frac{w^\top(\mu_0-\mu_1)(\mu_0-\mu_1)^\top w}{w^\top(\Sigma_0+\Sigma_1)w}$$

定义“类内散度矩阵”（within-class scatter matrix，类似于协方差矩阵）

$$S_w=\Sigma_0+\Sigma_1=\sum_{x\in X_0}(x-\mu_0)(x-\mu_0)^\top+\sum_{x\in X_1}(x-\mu_1)(x-\mu_1)^\top$$

和“类间散度矩阵”（between-class scatter matrix）

$$S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^\top$$

得LDA的目标函数，它是$S_w$和$S_b$的“广义瑞利商”（generalized Rayleigh quotient）。

$$J=\frac{w^\top S_bw}{w^\top S_ww}$$

发现上式分子分母都是关于$w$的二次项，所以解与其长度无关，只与方向有关，不妨令$w^\top S_ww=1$，等价于

$$\min_w-w^\top S_bw$$

$$s.t. w^\top S_ww=1$$

使用拉格朗日乘子法有

$$F=-w^\top S_bw+\lambda(w^\top S_ww-1)=0$$

求偏导得

$$S_bw=\lambda S_ww$$

注意到$(\mu_0-\mu_1)^\top w$为常数，$S_bw$的方向恒为$(\mu_0-\mu_1)$，上式等价于

$$S_bw=\lambda(\mu_0-\mu_1)$$

得

$w=S_w^{-1}(\mu_0-\mu_1)$

考虑数值解的稳定性，实践中常使用奇异值分解$S_w^{-1}=V\Sigma^-1U^\top$。

当两类数据同先验、满足正态分布且协方差相等时，使用LDA可达到最优分类。

推广至$N$分类任务中，定义“全局散度矩阵”

$$S_t=S_b+S_w=\sum^m_{i=1}(x_i-\mu)(x_i-\mu)^\top$$

定义“类内散度矩阵”

$$S_w=\sum_{i=1}^N\sum_{x\in X_i}(x-\mu_i)(x-\mu_i)^T$$

而类间散度矩阵为

$$S_b=S_t-S_w=\sum^N_{i=1}m_i(\mu_i-\mu)(\mu_i-\mu)^\top$$

常使用优化目标

$$J=\max_W\frac{tr(W^\top S_bW)}{tr(W^\top S_wW)}$$

得到$W$的解析解为$S_w^{-1}S_b$的$d'$个最大非零广义特征值所对应的特征向量矩阵，$d'\le N-1$。通常$d'$要比原属性数$d$小得多，而该投影方法也使用了类别信息，所以LDA也常用于监督降维技术。

# 3.5 多分类学习

多分类学习的基本思想是拆分成二分类任务在进行集成，经典的拆分策略有三种：“一对一”（one vs. one，OvO），“一对其余”（one vs. rest，OvR），“多对多”（Many vs. Many，MvM）。

**OvO，OvR：**

前两者的示意图如下：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404232948.png" alt="image-20210404232948855" style="zoom:80%;" /></div>

OvR中如果有多个预测结果，则选择置信度更大的。

容易发现，OvR的存储开销和测试时间要远小于OvO，但OvO的训练时间通常较小，预测性能二者则相差不大。

**MvM：**

而MvM方法则是每次使用特殊的设计，将若干个类作为正类，这里使用一种常用技术“纠错输出码”（Error Correcting Output Codes，ECOC），分为两步：

编码阶段，ECOC首先对$N$个类别进行$M$次划分，将一部分作为正类，形成M个训练集，分别训练出M个分类器。

解码阶段，$M$个分类器对测试集进行预测，将这些预测标记组成一个编码，与每个类别各自的编码进行比较，返回距离较小的作为结果。

其中类别划分主要通过“编码矩阵”（coding matrix）指定，常见有二元码和三元码的形式，后者在正类反类之外设立了“停用类”，如图a所示，测试实例返回分类$C_3$，即使有少许分类器产生错误，总体结果仍然可能保持正确，对错误有一定的容忍和修正能力：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404233703.png" alt="image-20210404233703258" style="zoom:80%;" /></div>

对同等长度的编码，任意两个类别之间的编码距离越远，则纠错能力越强，因此可以计算出效果优秀的编码。

# 3.6 类别不平衡问题

在线性回归预测中，如果正反例不平衡可以使用“再缩放”（rescaling）方法：

$$\frac {y'}{1-y'}=\frac y{1-y}\times\frac{m^-}{m^+}$$

然而这种方法需要建立在样本是总体的无偏采样这一假设上。现有的技术主要使用三种方法：“欠采样”（undersampling，去除多数样本的一部分）、“过采样”（oversampling，增强少数样本）和使用类似再缩放技术的“阈值移动”（threshold-moving）。欠采样方法可能会丢失信息，过采样方法则有可能产生过拟合现象。常用的欠采样方法有EasyEnsemble，过采样方法有SMOTE。

# 3.7 其他

“稀疏表示”近年来很受关注，本质上对应$L_0$范数的优化，LASSO通过$L_1$范数近似$L_0$范数，可以求取稀疏解。

OvO和OvR都是ECOC的特例，许多ECOC编码法都是基于具有代表性的二分类问题进行编码。

MvM还可以通过有向无环图（Directed Acyclic Graph）拆分法，将类别划分表达成树形结构。

对二分类问题，再缩放可以获得理论最优解，但对于多分类任务并不是所有情况都有解析解。

多分类中虽然有很多分类，但每个样本只属于一个类别，一个样本属于多个类别的任务叫“多标记学习”（multi-label learning）。