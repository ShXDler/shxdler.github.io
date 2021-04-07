# 7 贝叶斯分类器

# 7.1 贝叶斯决策论

设有$N$种可能的类别标记：${\mathcal Y}=\{c_1,c_2,...,c_N\}$，$\lambda_{ij}$是将$c_i$样本分到$c_j$的损失。基于后验概率$P(c_i|x)$可以得到期望损失（expected loss），即样本$x$上的“条件风险”（conditional risk）：

$$R(c_i|x)=\sum^N_{j=1}\lambda_{ij}P(c_j|x)$$

我们的任务是寻找一个判定准则$h:\mathcal X\mapsto \mathcal Y$以最小化总体风险：

$R(h)={\mathbb E}_x[R(h(x)|x)]$

对每个样本，如果$h$能最小化条件风险$R(h(x)|x)$，那么也一定能最小化总体风险，这也就产生了贝叶斯判定准则（Bayes decision rule）：在每个样本上选择那个能使条件风险$R(c|x)$最小的类别标记，即：

$$h^*(x)=\arg\min_{c\in \mathcal Y}R(c|x)$$

这里的$h^*$称为“贝叶斯最优分类器”（Bayes optimal classifier），与之对应的总体风险$R(h^*)$称为贝叶斯风险（Bayes risk）。$1-R(h^*)$为机器学习能产生的模型精度的理论上限。

具体来说，如果目标是最小化分类错误率，则误判损失$\lambda_{ij}$可写为

$$\lambda_{ij}=\left\{\begin{aligned}&0,if\ i=j\\
&1,otherwise\end{aligned}\right.$$

此时的条件风险为

$$R(c|x)=1-P(c|x)$$

最小化分类错误率的贝叶斯最优分类器为

$$h^*(x)=\arg\max_{c\in{\mathcal Y}}P(c|x)$$

因此，要使用贝叶斯判定准则需要获得后验概率，然而在现实任务中这通常很难。机器学习所要实现的是基于有限的训练样本集尽可能准确地估计出后验概率。主要有两种策略：

（1）“判别式模型”（discriminative models）给定$x$，直接建模$P(c|x)$来预测$c$，前面的决策树、神经网络、支持向量机等都可归入判别式模型的范畴；
（2）“生成式模型”（generative models）先对联合概率分布$P(c,x)$进行建模，然后再获得$P(c|x)$。生成式模型要考虑

$$P(c|x)=\frac{P(x,c)}{P(x)}$$

由贝叶斯定理

$$P(c|x)=\frac{P(c)P(x|c)}{P(x)}$$

其中$P(c)$是类“先验概率”（prior probability），$P(x|c)$是样本$x$相对于类标记$c$的类条件概率（class-conditional probability）或称“似然”（likelihood），$P(x)$是用于归一化的“证据”（evidence）因子，问题转化为估计先验$P(c)$和似然$P(x|c)$。而先验$P(c)$表达了样本空间中各类样本所占比例，根据大数定律，可以直接通过充足i.i.d.的样本频率进行估计。但对于似然$P(x|c)$而言，它涉及关于$x$的所有属性的联合概率，很多样本取值可能在训练集中没有出现，所以很难直接根据频率来估计。

# 7.2 极大似然估计

估计类条件概率的一种常用策略是先假定其具有某种确定的概率分布，再进行参数估计（parameter estimation），常用方法有频率学派的极大似然估计（Maximum Likelihood Estimation，MLE），具体内容见数理统计，不再赘述。

# 7.3 朴素贝叶斯分类器

基于贝叶斯公式来估计后验概率$P(c|x)$的主要困难在于它是所有属性上的联合概率，难以从有限的训练样本中直接估计得到。而朴素贝叶斯分类器（naive Bayes classifier）采用了“属性条件独立性假设”（attribute conditional independence assumption），假设所有属性相互独立，从而

$$P(c|x)=\frac{P(c)P(x|c)}{P(x)}=\frac{P(c)}{P(x)}\prod_{i=1}^dP(x_i|c)$$

而$P(x)$不变，得到朴素贝叶斯分类器表达式

$$h_{nb}(x)=\arg\max_{c\in\mathcal Y}P(c)\prod^d_{i=1}P(x_i|c)$$

其训练过程就是根据训练集$D$来估计类先验概率$P(c)$和每个属性的条件概率$P(x_i|c)$，有

$$P(c)=\frac{|D_c|}{D}$$

对离散属性，有

$$P(x_i|c)=\frac{|D_{c,x_i}|}{|D_c|}$$

而对连续属性，假设$p(x_i|c)\sim\mathcal N(\mu_{c,i},\sigma^2_{c,i/})$有

$$p(x_i|c)=\frac1{\sqrt{2\pi}\sigma_{c,i}}\exp(-\frac{(x_i-\mu_{c,i})^2}{2\sigma_{c,i}^2})$$

