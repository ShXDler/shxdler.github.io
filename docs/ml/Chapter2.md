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

实际上，$F1$值是$P$和$R$的调和平均：$\frac1{F1}=\frac12\cdot(\frac1P+\frac1R)$，而$F_\beta$则是加权调和平均$\frac1{F_\beta}=\frac1{1+\beta^2}(\frac1P+\frac{\beta^2}R)$。和算术平均值几何平均值相比，调和平均更注重较小值。

而对进行了多次训练测试的多个混淆矩阵，我们可以使用其平均值，得到“宏查准率”（$macro-P$）、“宏查全率”（$macro-R$）、“宏$F1$”（$macro-F1$）。

$$macro-P=\frac1n\sum^m_{i=1}P_i$$

$$macro-R=\frac1n\sum^m_{i=1}R_i$$

$$macro-F1=\frac{2\times macro-P\times macro-R}{macro-P+macro-R}$$

另外，也可以先按各混淆矩阵元素得到$\bar{TP}$等平均值，再代入公式计算得到得到“微查准率”（$micro-P$）、“微查全率”（$micro-R$）、“微$F1$”（$micro-F1$）。

## 2.3.3 ROC与AUC

很多学习器会产生一个概率预测，然后与分类阈值（threshold）进行比较，我们根据这个概率值，可以对测试样本进行排序，整个分类过程会以某个“截断点”（cut point）分为正例反例两部分。

ROC（受试者工作特征，Receiver Operating Characteristic）曲线以“真正例率”（TPR）为纵轴，“假正例率”（FPR）为横轴：

$$TPR=\frac{TP}{TP+FN}$$$$FPR=\frac{FP}{TN+FP}$$

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404161121.png" alt="image-20210404161121539" style="zoom:80%;" /></div>

其中对角线代表着“随机猜测”得到的ROC曲线。

在现实任务使用有限个测试样例时，可以绘制如上图图（b）中不光滑的曲线：给定$m^+$个正例和$m^-$个反例，首先对样例按照预测结果进行排序，并将阈值设为最大，此时真正例率和假正例率均为0，然后将阈值依次设为每个样例的预测值，对上一个坐标$(x,y)$，如果当前是真正例，标记点为$(x,y+\frac1{m^+})$，反之标记$(x+\frac1{m-},y)$。

我们可以计算ROC曲线围成的面积得到AUC值：

$$AUC=\frac12\sum^{m-1}_{i=1}(x_{i+1}-x_i)\cdot(y_i+y_{i+1})$$

形式上来看，AUC值考虑的是样本预测的排序质量，给定$m^+$个正例和$m^-$个负例，令$D^+$和$D^-$分别代表正反例集合，定义排序损失（loss） 为：

$$\ell_{rank}=\frac1{m^+m^-}\sum_{x^+\in D^+}\sum_{x^-\in D^-}({\mathbb I}(f(x^+)<f(x^-))+\frac12{\mathbb I}(f(x^+)=f(x^-)))$$

容易发现loss值计算的是ROC曲线上部分围成的面积，即：

$$AUC=1-\ell_{rank}$$

## 2.3.4 代价敏感错误率与代价曲线

为权衡不同类型错误所造成的的不同损失，可为错误赋予“非均等代价”（unequal cost）。以二分类任务为例，我们设计如下的代价矩阵：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404164449.png" alt="image-20210404164449860" style="zoom:80%;" /></div>

前面的性能度量大多假设了均等代价。在非均等代价的条件下，可以得到“代价敏感”（cost-sensitive）的错误率为：

$$E(f;D;cost)=\frac1m(\sum_{x_i\in D^+}{\mathbb I}(f(x_i)\ne y_i)\times cost_{01}+\sum_{x_i\in D^-}{\mathbb I}(f(x_i)\ne y_i)\times cost_{10})$$

此时的ROC曲线不能反映总体代价，我们可以使用“代价曲线”（cost curve），其横轴是正例概率代价：$$P(+)cost=\frac{p\times cost_{01}}{p\times cost_{01}+(1-p)\times cost_{10}}$$

其中$p$是样例为正例的概率，纵轴是归一化代价：$$cost_{norm}=\frac{FNR\times p\times cost_{01}+FPR\times(1-p)\times cost_{10}}{p\times cost_{01}+(1-p)\times cost_{10}}$$

在绘制代价曲线时，根据ROC曲线坐标（TPR， FPR），计算出相应的FNR，然后再代价平面上绘制一条从（0，FPR）到（1，FNR）的线段，取所有线段的下界，围成的面积即为期望总体代价：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404165829.png" alt="image-20210404165829408" style="zoom:80%;" /></div>



# 2.4 比较试验

在比较不同模型得到的结果时，首先我们要比的是泛化性能，模型在训练集和测试集上的性能未必表现相同；第二，不同大小的测试集可能会影响测试结果；第三，机器学习算法本身也具有随机性。因此，我们倾向于采用假设检验（hypothesis test）比较不同学习器的性能，下文以$\epsilon$表示错误率。

## 2.4.1 假设检验

设泛化错误率为$\epsilon$的学习器的测试错误率为$\hat{\epsilon}$，则它将$m$个样本的$m'$个错误分类的概率为$\binom m{m'}\epsilon^{m'}(1-\epsilon)^{m-m'}$，得到测试错误率为$\hat{\epsilon}$的概率为$$P(\hat{\epsilon};\epsilon)=\binom{m}{\hat{\epsilon}\times m}\epsilon^{\hat{\epsilon}\times m}(1-\epsilon)^{m-\hat{\epsilon\times m}}$$

解方程$$\frac{\partial P(\hat{\epsilon};\epsilon)}{\partial\epsilon}=0$$得到$\epsilon$的极大似然估计量就等于${\hat\epsilon}$，符合二项分布特征：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404173211.png" alt="image-20210404173211283" style="zoom:80%;" /></div>

由此，我们可以使用“二项检验”（binomial test）来对“$\epsilon<0.3$”进行检验。更一般的，考虑“$\epsilon<\epsilon_0$”，置信度（confidence）为“$1-\alpha$”时，得到其临界值$\bar\epsilon$为:

$$\bar\epsilon=\min\epsilon\ s.t.\sum^m_{i=\epsilon_0\times m+1}\binom mi\epsilon^i(1-\epsilon)^{m-i}<\alpha$$

当$\hat\epsilon<\bar\epsilon$时，不能拒绝原假设。

而当我们通过多次重复留出法或交叉验证法时，可以得到多个测试错误率，此时可以使用“t检验”（t-test）。假设我们得到了$k$个测试错误率，则平均测试错误率$\bar\epsilon$和方差$s^2$为

$$\bar \epsilon=\frac1k\sum^k_{i=1}\hat\epsilon_i$$

$$S^2=\frac1{k-1}\sum^k_{i=1}(\hat\epsilon_i-\bar\epsilon)^2$$

这里认为$\epsilon$近似服从正态分布，得统计量$$\tau_t=\frac{\sqrt k(\bar\epsilon-\epsilon_0)}S\sim t(k-1)$$

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404174329.png" alt="image-20210404174329753" style="zoom:80%;" /></div>

对假设“$\bar\epsilon=\epsilon_0$”，我们采用双边检验，如果$\tau_t$在区间$[t_{-\alpha/2},t_{\alpha/2}]$内，则无法拒绝原假设。

## 2.4.2 交叉验证t检验

对两个学习器A和B使用k折交叉验证得到的结果，简单来讲可以使用配对样本t检验方法，但是注意到每次交叉验证的训练集实际上是有重合的，也就是说每次得到的结果并不独立，可采用“5*2交叉验证法”。

5*2交叉验证是做5次2折交叉验证，每次2折交叉验证之前将数据大乱，使5次交叉验证的数据划分各不相同。我们对第$i$次交叉验证分别求出2折的学习器效果之差$\Delta_i^1$和$\Delta_i^2$。缓解测试错误率的非独立性，样本均值只计算$\Delta_1^1$和$\Delta_1^2$的均值，而样本方差计算每次结果的方差：

$$\bar\Delta=\frac12(\Delta^1_1+\Delta^2_1)$$

$$S_i^2=(\Delta_i^1-\frac{\Delta_i^1+\Delta_i^2}2)^2+(\Delta_i^2-\frac{\Delta_i^1+\Delta_i^2}2)^2$$

得统计量

$$\tau_t=\frac{\bar\Delta}{\sqrt{\frac15\sum_{i=1}^5S^2_i}}\sim t(5)$$

## 2.4.3 McNemar检验

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404181211.png" alt="image-20210404181211108" style="zoom:80%;" /></div>

我们可以使用McNemar检验法对上述四格表进行边缘齐性检验（即算法A和B正确率的差异检验），计算卡方统计量

$$\tau_{\chi^2}=\frac{(|e_{01}-e_{10}|)^2-1}{e_{01}+e_{10}}\sim \chi^2(1)$$

卡方统计量和似然比统计量：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404214205.png" alt="image-20210404214205834" style="zoom:80%;" /></div>

## 2.2.4 Friedman检验与Nemenyi后续检验

前面的方法比较的是两个学习器的效果，对于多个学习器之间的比较，我们可以使用Friedman检验法。首先假设使用$D1$、$D2$、$D3$和$D4$四个数据集对算法A、B和C进行比较，先使用交叉验证法得到每个算法在每个测试集上的测试结果并进行排序，得到下表：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404183204.png" alt="image-20210404183204486" style="zoom:80%;" /></div>

然后使用Friedman检验判断每种算法性能是否相同，假设我们在$N$个数据集上比较$k$个算法，零$r_i$为第$i$个算法的平均序值，则$r_i$的均值和方差分别为$\frac12(k+1)$和$\frac{(k^2-1)}{12N}$，得统计量

$$\begin{aligned}\tau_{\chi^2}&=\frac{k-1}k\cdot\frac{12N}{k^2-1}\sum^k_{i=1}(r_i-\frac{k+1}2)^2\\ &=\frac{12N}{k(k+1)}(\sum_{i=1}^kr_i^2-\frac{k(k+1)^2}4)\end{aligned}\dot\sim\chi^2(k-1)$$

这样的Friedman检验过于保守，通常使用统计量

$$\tau_F=\frac{(N-1)\tau_{\chi^2}}{N(k-1)-\tau_{\chi^2}}\dot\sim F(k-1,(k-1)(N-1))$$

如果原假设“所有算法性能相同”被拒绝，则说明算法性能存在显著差异，此时要进行“后续检验”（post-hoc test）进一步区分各算法，常用Nemenyi检验。它计算的是平均序值的临界值

$$CD=q_\alpha\sqrt{\frac{k(k+1)}{6N}}$$

如果两个算法的平均序值的差超过了临界值$CD$，则拒绝“两个算法性能相同”的假设。

# 2.5 偏差与方差

对解释学习算法泛化性能，我们使用“偏差-方差分解”（bias-variance decomposition）这一工具。以回归任务为例，期望预测为

$$\bar f(x)={\mathbb E}_D[f(x;D)]$$

使用样本数相同的不同训练集产生的方差为

$$Var(f(x))={\mathbb E}_D[(f(x;D)-\bar f(x))^2]$$

噪声项为

$$\epsilon^2={\mathbb E}_D[(y_D-y)^2]$$

期望输出与真实值的偏差（bias）为

$$bias^2(f(x))=(\bar f(x)-y)^2$$

假设噪声项期望为零，即${\mathbb E}_D[y_D-y]=0$，可对期望泛化误差进行分解：

$$\begin{aligned}E(f;D)&={\mathbb E}_D[(f(x;D)-y_d)^2]\\&={\mathbb E}_D[(f(x;D)-\bar f(x)+\bar f(x)-y_D)^2]\\&={\mathbb E}_D[(f(x;D)-\bar f(x))^2]+{\mathbb E}_D[(\bar f(x)-y_D)^2]\\&+{\mathbb E}_D[2(f(x;D)-\bar f(x))(\bar f(x)-y-\epsilon)]\\&=Var(f(x))+{\mathbb E}_D[(\bar f(x)-y+y-y_D)^2]\\&=Var(f(x))+{\mathbb E}_D[(\bar f(x)-y)^2]+{\mathbb E}_D[(y_D-y)^2]\\&+{\mathbb E}_D[2(\bar f(x)-y)\epsilon]\\&=Var(f(x))+bias^2(f(x))+\epsilon^2\end{aligned}$$

方差偏差关系图如下：偏差较大方差较小时，模型可能欠拟合；偏差较小方差较大时，模型可能过拟合。

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210404203103.png" alt="image-20210404203103717" style="zoom:80%;" /></div>