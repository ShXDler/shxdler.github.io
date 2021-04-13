# 8 集成学习

# 8.1 个体与集成

集成学习（ensemble learning）也被称为多分类器系统（multi-classifier system）、基于委员会的学习（committee-based learning）。集成学习先产生一组“个体学习器”（individual learner），再用某种策略将它们结合起来。如果只包含同种类型的个体学习器，这样的集成是“同质”（homogeneous）的，个体学习器称“基学习器”（base learner），学习算法称“基学习算法”（base learning algorithm）。如果包含不同类型的学习器则称“异质”（heterogenous）的，这时个体学习器称“组件学习器”（component learner）或个体学习器。

集成学习往往是针对“弱学习器”（weak learner）而言的，同时，假设通过投票法（voting）产生结果，那么个体学习器应当有一定的“准确性”和“多样性”。

考虑二分类问题$y\in\{-1,+1\}$和真实函数$f$，假设基分类器的错误率为$\epsilon$，则对每个基分类器$h_i$有

$$P(h_i(x)\ne f(x))=\epsilon$$

如果集成学习通过简单投票的方法结合$T$个基分类器，如果有超过半数的基分类器正确，则集成分类就会正确：

$$F(x)=sgn(\sum^T_{i=1}h_i(x))$$

如果基分类器的错误率相互独立，由Hoeffding不等式

$$P(F(x)\ne f(x))=\sum_{k=0}^{\lfloor T/2\rfloor}\binom Tk(1-\epsilon)^k\epsilon^{T-k}\le\exp(-\frac12T(1-2\epsilon)^2)$$

由此可见，当$T$变大时，集成的错误率也会以指数级下降趋向于0。

然而，上述证明要求各错误率相互独立，实际上现实任务中这无法成立。而个体学习器的准确性和多样性本身就存在冲突，如何产生好而不同的个体学习器是集成学习研究的核心。而根据个体学习器的生成方式，集成学习方法可以分成两类：如果个体学习间存在强依赖关系则必须串行生成的序列化方法，如Boosting；个体学习器之间不存在强依赖关系可同时生成的并行化方法，如Bagging和“随机森林”（Random Forest）。

# 8.2 Boosting

Boosting的工作机制为：先从训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，更多地关注先前基学习器做错的样本，再根据新的样本分布训练下一个基学习器，直至基学习器数目达到指定的$T$，再将这$T$个基学习器进行加权结合。而Boosting算法最著名的代表是AdaBoost。

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210411164820.png" alt="image-20210411164812697" width=500/></div>

Adaboost有多种推导方式，下面一种基于“可加模型”（additive model），即基学习器的线性组合

$$H(x)=\sum^T_{i=1}\alpha_th_t(x)$$

来最小化指数损失函数（exponential loss function）

$$\ell_{\exp}(H|\mathcal D)=\mathbb E_{x\sim\mathcal D}[e^{-f(x)H(x)}]$$

对$H(X)$求导有

$$\frac{\partial\ell_{\exp}(H|\mathcal D)}{\partial H(x)}=-e^{-H(x)}P(f(x)=1|x)+e^{H(x)}P(f(x=-1)|x)$$

令上式等于零，则

$$H(x)=\frac12\log\frac{P(f(x)=1|x)}{P(f(x)=-1|x)}$$

$$\begin{aligned}sgn(H(x))&=sgn(\frac12\log\frac{P(f(x)=1|x)}{P(f(x)=-1|x)})\\
&=\left\{\begin{aligned}&1,&P(f(x)=1|x)>P(f(x)=-1|x)\\
&-1,&P(f(x)=1|x)<P(f(x)=-1|x)\end{aligned}\right.\\
&=\arg\max_{y\in\{-1,+1\}}P(f(x)=y|x)\end{aligned}$$

这意味着$sgn(H(x))$达到了贝叶斯最优错误率，因此指数损失函数是分类任务中原来0/1损失函数的一致（consistent）替代损失函数，同时也具有连续可微等性质。

在AdaBoost算法中，第一个基分类器$h_1$是通过直接将基学习算法用于初始数据分布得到，之后迭代生成$h_t$和$\alpha_t$，当基分类器$h_t$基于分布$\mathcal D_t$产生后，该基分类器的权重$\alpha_t$应使得$\alpha_th_t$最小化指数损失函数

$$\begin{aligned}\ell_{\exp}(\alpha_t,h_t|\mathcal D_t)&=E_{x\sim\mathcal D_t}[e^{-f(x\alpha_th_t(x))}]\\
&=E_{x\sim\mathcal D_t}[e^{-\alpha_t}\mathbb I(f(x)=h_t(x))+e^{\alpha_t}\mathbb I(f(x)\ne h_t(x))]\\
&=e^{-\alpha_t}P_{x\sim\mathcal D_t}(f(x)=h_t(x))+e^{\alpha_t}P_{x\sim\mathcal D_t}(f(x)\ne h_t(x))\\
&=e^{-\alpha_t}(1-\epsilon_t)+e^{\alpha_t}\epsilon_t\end{aligned}$$

其中$\epsilon_t$是第t个学习器的错误率，对$\alpha_t$求偏导得

$$-e^{-\alpha_t}(1-\epsilon_t)+e^{\alpha_t}\epsilon_t$$

令上式等于0，有

$$\alpha_t=\frac12\log{\frac{1-\epsilon_t}{\epsilon_t}}$$

得到权重更新公式。

获得$H_{t-1}$后，AdaBoost将对样本分布进行调整，使下一轮的基学习器能够纠正前面的错误，理想的$h_t$能纠正$H_{t-1}$的全部错误，即最小化$\ell_{\exp}(H_{t-1}+\alpha_th_t|\mathcal D)$，可简化为最小化

$$\ell_{\exp}(H_{t-1}+h_t|\mathcal D)=\mathbb E_{x\in\mathcal D}[e^{-f(x)(H_{t-1}(x)+h_t(x))}]=\mathbb E_{x\in\mathcal D}[e^{-f(x)H_{t-1}(x)}e^{-f(x)h_t(x))}]$$

注意到$f^2(x)=h^2_t(x)=1$，使用泰勒展开可以得到

$$\ell_{\exp}(H_{t-1}+h_t|\mathcal D)\simeq\mathbb E_{x\sim\mathcal D}[e^{-f(x)H_{t-1}(x)}(1-f(x)h_t(x)+\frac12)]$$

得到最理想学习器

$$\begin{aligned}h_t(x)&=\arg\min_h\ell_{\exp}(H_{t-1}+h|\mathcal D)\\
&=\arg\min_h\mathbb E_{x\sim \mathcal D}[e^{-f(x)H_{t-1}(x)}(1-f(x)h_t(x)+\frac12)]\\
&=\arg\min_h\mathbb E_{x\sim \mathcal D}[e^{-f(x)H_{t-1}(x)}f(x)h(x)]\\
&=\arg\min_h\mathbb E_{x\sim \mathcal D}[\frac{e^{-f(x)H_{t-1}(x)}}{\mathbb E_{x\sim\mathcal D}[e^{-f(x)H_{t-1}(x)}]}f(x)h(x)]\end{aligned}$$

期望中的分母是一个常数，记$\mathcal D_t$为分布

$$\mathcal D_t(x)=\mathcal D(x)\frac{e^{-f(x)H_{t-1}(x)}}{\mathbb E_{x\sim\mathcal D}[e^{-f(x)H_{t-1}(x)}]}$$

则上式可以转化为

$$h_t(x)=\arg\max_h\mathbb E_{x\sim\mathcal D_t}[f(x)h(x)]$$

而$f(x),h(x)\in\{-1,+1\}$，有

$$f(x)h(x)=1-2\mathbb I(f(x)\ne h(x))$$

得到

$$h_t(x)=\arg\max_h\mathbb E_{x\sim\mathcal D_t}[\mathbb I(f(x)\ne h(x))]$$

可见理想的$h_t$将在分布$\mathcal D_t$得到最小化分类误差，因此弱分类器将基于分布$\mathcal D_t$来训练，且针对$\mathcal D_t$的分类误差应当小于0.5，有

$$\begin{aligned}\mathcal D_{t+1}(x)&=\mathcal D(x)\frac{e^{-f(x)H_{t}(x)}}{\mathbb E_{x\sim\mathcal D}[e^{-f(x)H_{t}(x)}]}\\
&=\mathcal D(x)\frac{e^{-f(x)H_{t-1}(x)}e^{-f(x)\alpha_th_t(x)}}{\mathbb E_{x\sim\mathcal D}[e^{-f(x)H_{t}(x)}]}\\
&=\mathcal D_t(x)\cdot e^{-f(x)\alpha_th_t(x)}\frac{\mathbb E_{x\sim\mathcal D}[e^{-f(x)H_{t-1}(x)}]}{\mathbb E_{x\sim\mathcal D}[e^{-f(x)H_{t}(x)}]}\end{aligned}$$

得到样本分布的迭代更新公式。

---

Boosting算法要求基学习器能指定分布进行学习，可以通过“重赋权法”（re-weighting），对那些不能赋权的算法，可通过“重采样法”（re-sampling）。Boosting算法在每一轮都要检验学习器错误率是否小于50%，一旦不满足就会抛弃当前学习器并且停止学习过程，这样可能会性能不佳。如果采用“重采样法”，就可以在抛弃当前学习器后重新对训练样本进行采样并训练新的学习器，以避免训练过程过早停止。

从偏差-方差分解的角度来看，Boosting主要关注降低偏差，因此Boosting能基于泛化性能相当弱的学习器构建出很强的集成。

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210411173242.png" alt="image-20210411173242734" style="zoom:80%;" /></div>

# 8.3 Bagging与随机森林

前面提到，集成学习要求基学习器有较大差异，那么我们可以对训练样本进行采样，但如果每个子训练集都完全不同，那每个基学习器都只用了一小部分数据，难以确保生成好的学习器，所以我们可以考虑使用有交叠的采样子集。

## 8.3.1 Bagging

Bagging（Bootstrap AGGregatING）是并行式集成学习的著名代表，它使用Bootstrap算法，得到的采样集包含了初始训练集大概63.2%的样本。

由此，我们可以得到$T$个含$m$个训练样本的采样集，基于每个采样集训练出一个基学习器再进行组合，这就是Bagging的基本流程。在组合的过程中，分类任务可以使用简单投票法，回归任务可以用简单平均法，如果两个类收到同样票数可以随机选一个，也可以进一步考察投票的置信度，算法如下所示：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210411180107.png" alt="image-20210411180107762" width=500 /></div>

不难发现Bagging方法和一个学习器的复杂度同阶，比较高效，同时它与标准AdaBoost不同，可以直接用于多分类、回归等任务。

另外，Bootstrap采样后剩下的36.8%的样本可以作为验证集来对泛化性能进行“包外估计”（out-of-bag estimate），这样需要记录每个基学习器所使用的训练样本，记$D_t$为$h_t$使用的训练样本集，$H^{oob}(x)$为包外预测，则有

$$H^{oob}(x)={\arg\max}_{y\in\mathcal Y}\sum^T_{i=1}\mathbb I(h_t(x)=y)\cdot\mathbb I(x\notin D_t)$$

则Bagging的包外估计错误率为

$$\epsilon^{oob}=\frac1{|D|}\sum_{(x,y)\in D}\mathbb I(H^{oob}(x)\notin y)$$

除了进行包外估计，包外样本还可以用来辅助剪枝，或者估计决策树中节点的后验概率来辅助对零训练样本节点的处理。同时基学习器是神经网络时，包外样本还可以用来辅助提前停止减轻过拟合。

从偏差-方差分解的角度来看，Bagging主要关注降低方差，因此在不剪枝决策树、神经网络等容易受样本扰动的学习器上效果更明显。

<div align="center"><img src="C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20210413232333866.png" alt="image-20210413232333866" style="zoom:80%;" /></div>

## 8.3.2 随机森林

随机森林（Random Forest）在决策树的训练过程中引入了随机属性选择，传统决策树在选择划分属性时实在当前节点的属性集合中选一个最优属性；而RF则是先选出一个包含k个属性的子集，然后再从子集中选择最优属性。如果$k=d$，基决策树构建和普通决策树相同；$k=1$则是随机选择一个属性进行划分；一般情况下会选择$k=log_2d$。

Bagging方法的“多样性”仅来自于样本扰动，而随机森林还使用了属性扰动。RF的收敛性和Bagging差不多，起始性能往往比较差，但随着个体学习器数目的增加，它往往可以收敛到更低的泛化率，并且通常来讲，因为RF考察的属性集合小于全部属性集合，所以训练效率也比Bagging高。

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210413233319.png" alt="image-20210413233319104" style="zoom:80%;" /></div>

# 8.4 结合策略

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210413233719.png" alt="image-20210413233719236" style="zoom:80%;" /></div>

学习器结合能从三个方面带来好处：
（1）统计方面，学习任务的假设空间往往很大，可能有多个假设在训练集上达到最优性能，使用单学习器泛化性能可能不佳。
（2）计算方面：学习算法往往会陷入局部极小，多次运行进行结合可以降低陷入局部极小点的风险。
（3）表示方面：某些学习任务的真实假设可能不在当前学习算法所考虑的假设空间中，结合多个学习器可能学得更好的近似。

记集成包括$t$个学习器$\{h_1,h_2,...,h_T\}$，其中$h_i$在x上的输出记为$h(x)$，下面是几种常见策略：

## 8.4.1 平均法

对数值输出而言，最常见的结合策略是平均法（averaging）

**简单平均法**（simple averaging）

$$H(x)=\frac1T\sum^T_{i=1}h_i(x)$$

**加权平均法**（weighted averaging）

$$H(x)=\sum^T_{i=1}w_ih_i(x)$$

加权平均法的权重一般从训练数据中学习得到，但是容易产生过拟合，加权平均法未必一定优于简单平均法。一般而言，个体学习器性能差异较大时使用加权平均法较好。

## 8.4.2 投票法

对分类任务而言，学习器$h_i$就从类别标记集合$\{c_1,c_2,...,c_N\}$中预测出一个标记，常使用投票法（voting）记$h_i$在样本$x$上的输出为一个$N$维向量$(h_i^1(x);h_i^2(x);...;h_i^N(x))$。

**绝对多数投票法**（majority voting）

$$H(x)=\left\{\begin{aligned}&c_j,&\text{if}\ \sum^T_{i=1}h_i^j(x)>0.5\sum^N_{k=1}\sum^T_{i=1}h_i^k(x)\\
&\text{reject},&\text{otherwise}.\end{aligned}\right.$$

如果某标记得票超过半数，则预测为该标记，否则拒绝预测。

**相对多数投票法**（plurality voting）

$$H(x)=c_{\arg\max_j\sum_{i=1}^Th_i^j(x)}$$

即预测为得票最多的标记，如果多个标记获最高票，则从中随机选取一个。

**加权投票法**（weighted voting）

$$H(x)=c_{\arg\max_j\sum_{i=1}^Tw_ih_i^j(x)}$$

绝对多数投票法提供了“拒绝预测”的选项，可以提供较高的可靠性。而如果学习任务要求必须提供预测结果，则绝对多数投票法将退化为相对多数投票法。因此，在不允许拒绝预测的任务中，绝对多数、相对多数投票法统称“多数投票法”。

现实任务中，不同类型的个体学习器可能产生不同类型的$h_i^j(x)$，常见有：
类标记：预测类别为1，其他取0，“硬投票”（hard voting）
类概率：相当于对后验概率$P(c_j|x)$的一个估计，“软投票”（soft voting）

不同类型的预测值不能混用。而对一些能预测出