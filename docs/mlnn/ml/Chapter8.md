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

Boosting的工作机制为：先从训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，更多地关注先前基学习器做错的样本，再根据新的样本分布训练下一个基学习器，直至基学习器数目达到指定的$T$，再将这$T$个基学习器进行加权结合。而Boosting算法最著名的代表是AdaBoost，它有多种推导方式，下面一种基于“可加模型”（additive model），即基学习器的线性组合

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

