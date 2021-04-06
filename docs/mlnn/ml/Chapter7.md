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

因此，要使用贝叶斯判定准则需要获得后验概率，然而在现实任务中这通常很难。机器学习所要实现的是基于有限的训练样本集尽可能准确地估计出后验概率。主要有两种策略：（1）“判别式模型”（discriminative models）给定$x$，直接建模$P(c|x)$来预测$c$（2）“生成式模型”（generative models）先对联合概率分布$P(c,x)$进行建模，然后再获得$P(c|x)$。前面的决策树、神经网络、支持向量机等都可归入判别式模型的范畴。