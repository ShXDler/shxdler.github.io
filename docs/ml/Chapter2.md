# 2.1 经验误差与过拟合

误差：实际预测输出和真实输出间的差异——训练误差（经验误差）、泛化误差

对过拟合、欠拟合问题，进行模型选择。

# 2.2 评估方法

建立测试集，与真实分布i.i.d，同时与训练集互斥。

## 2.2.1 留出法（hold-out）

将数据集$D$划分为两个互斥的训练集$S$和测试集$T$，使用$S$上训练出的模型用$T$评估模型。

首先，训练集和测试集的划分要保证数据分布i.i.d，如使用分层抽样保留类别比例。另外，在给定训练集测试集大小比例后，可以采用若干次随机划分的重复实验，计算结果的平均值。训练集比例常在2/3到4/5之间，测试集一般不少于30条观测。

## 2.2.2 交叉验证法（cross validation）

将原始数据集划分成$k$个大小相似的数据集重复$p$次，进行$p$次$k$折交叉验证。$k$等于样本量$m$时的特例又叫留一法，它的评价结果相对准确，但是计算开销很大。

## 2.2.3 自助法（bootstrap）

由于训练集要比整体数据集小，可能会产生一定的估计偏差，可以使用自助法进行实验估计。自助法以自助采样法（bootstrap sampling，亦称可重复采样或有放回采样）为基础，在给定包含$m$个样本的数据集$D$，每次随机从$D$中挑选一个样本，复制进$D'$中再放回，重复$m$次即可。在采样过程中，$m$次采样中某一样本没有被采到的概率为$\lim_{m\rightarrow\infty}(1-\frac1m)^m=\frac1e\approx0.368$，所以可以用$D\setminus D'$作为测试集，这种测试结果也称“包外估计”（out-of-bag estimate）。

自助法在数据集较小、难以划分集合的时候很有用，同时还能产生多个不同的训练集，有利于集成学习。而自助法改变了初始数据集的分布，会引入估计偏差，在数据量足够时，留出法和交叉验证法更常用。

## 2.2.4 调参与最终模型

用训练集和验证集对参数设定范围和变化步长来调参，在模型选择完后，使用所有$m$个样本训练最终模型，并在测试集上估计模型的泛化能力。

# 2.3 性能度量

回归分析常用均方误差（MSE）：

$$E(f;D)=\frac1m\sum^m_{i=1}(f(x_i)-y_i)^2$$

更一般的：

$$E(f;{\mathcal D})=E[(f(x)-y)^2]=\int_{x\sim{\mathcal D}}(f(x)-y)^2p(x)dx$$

## 2.3.1 错误率（error rate）与精度（accuracy）

错误率：$E(f;D)=\frac1m\sum^m_{i=1}{\mathbb I}(f(x_i)\ne y_i)$

精度：$acc(f;D)=\frac1m\sum^m_{i=1}{\mathbb I}(f(x_i)=y_i)=1-E(f;D)$

更一般的：

错误率：$E(f;{\mathcal D})=P(f(x)\ne y)=\int_{x\sim{\mathcal D}}{\mathbb I}(f(x_i)\ne y_i)p(x)dx$

精度：$acc(f;{\mathcal D})=P(f(x)=y)=\int_{x\sim{\mathcal D}}{\mathbb I}(f(x_i)=y_i)p(x)dx=1-E(f;{\mathcal D})$

## 2.3.2 查准率（precision）与查全率（recall）

直观上来讲，查准率计算的是“预测正例中有多少是真的正例（准不准）”，查全率则是“真的正例中有多少被预测出来了（全不全）”。

对于二分类问题，可得到如下混淆矩阵：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404151345.png" alt="image-20210404151345559" style="zoom:80%;" /></div>

于是查准率$P$和查全率$R$分别定义为：


$$P=\frac{TP}{TP+FP}$$$$R=\frac{TP}{TP+FN}$$

很多情况下二者是矛盾而无法兼顾的，因此我们可以以$W$为横轴，$P$为纵轴绘制“P-R曲线”：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404151247.png" alt="image-20210404151247533" style="zoom:80%;" /></div>


如图所示，A曲线将C完全包住，可以认为A的性能优于C。而A和B曲线出现了交叉，二者在不同情况下有对应擅长的方向，此时我们可以使用平衡点（Break-Even Point，BEP），也就是$P$和$R$相等的值进行度量。

而BEP有点过于简化，此时我们可以使用$F1$度量：$$F1=\frac{2\times P\times R}{P+R}=\frac{2\times TP}{m+TP-TN}$$

有时我们对查全率和查准率的侧重不同，这时可以使用$F_\beta$进行改进：

$$F_\beta=\frac{(1+\beta^2)\times P\times R}{(\beta^2\times P)+R}$$

其中$\beta>1$对查全率有更大影响，$\beta<1$对查准率有更大影响。

实际上，$F1$值是$P$和$R$的调和平均：$\frac1{F1}=\frac12\cdot(\frac1P+\frac1R)$

而$F_\beta$则是加权调和平均$\frac1{F_\beta}=\frac1{1+\beta^2}(\frac1P+\frac{\beta^2}R)$

和算术平均值几何平均值相比，调和平均更注重较小值。

而对进行了多次训练测试的多个混淆矩阵，我们可以使用其平均值，得到“宏查准率”（$macro-P$）、“宏查全率”（$macro-R$）、“宏$F1$”（$macro-F1$）。

$$macro-P=\frac1n\sum^m_{i=1}P_i$$

$$macro-R=\frac1n\sum^m_{i=1}R_i$$

$$macro-F1=\frac{2\times macro-P\times macro-R}{macro-P+macro-R}$$

另外，也可以先按各混淆矩阵元素得到$\bar{TP}$等平均值，再代入公式计算得到得到“微查准率”（$micro-P$）、“微查全率”（$micro-R$）、“微$F1$”（$micro-F1$）。

## 2.3.3 ROC与AUC