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

需要注意的是，如果对于某个分类，某个属性没有出现过，得到的概率连乘值会是0，而这并不符合常理，因此我们估计概率值时采用了“平滑”（smoothing）操作，常用“拉普拉斯修正”（Laplacian correction），记$N$为类别数，$N_i$为第$i$个属性的取值数。得到的新估计公式为：

$$\hat P(c)=\frac{|D_c|+1}{|D|+N}\\
\hat P(x_i|c)=\frac{|D_{c,x_i}+1|}{|D_c|+N_i}$$

拉普拉斯修正避免了因训练集样本不充分导致的概率估值为0的问题，而在训练集变大时，修正过程引入的先验影响也可以忽略。

现实任务中，朴素贝叶斯分类器有多种实现方式：
如果任务最预测速度要求高，可以将分类器涉及的所有概率估值提前计算并存储起来；
如果任务数据更替频繁，可以使用“懒惰学习”（lazy learning），即不提前进行训练，等到收到预测请求时再进行概率估算；
而如果任务数据不断增加，则可以在现有估值基础上，只对新增样本的属性进行计数修正即可实现增量学习。

# 7.4 半朴素贝叶斯分类器

朴素贝叶斯分类器采用的属性条件独立性假设往往很难成立，我们放宽这一假设，适当地考虑一部分属性间的相互依赖信息，使用“半朴素贝叶斯分类器”（semi-naive Bayes classifiers）学习方法。“独依赖估计”（One_Dependent Estimator，ODE）是半朴素贝叶斯分类器最常用的一种策略，假设每个属性在类别之外最多依赖于一个其他的属性，有：

$$P(c|x)\propto P(c)\prod^p_{i=1}P(x_i|c,pa_i)$$

其中$pa_i$为$x$所依赖的属性，称$x_i$的父属性。如果确定了$x$所依赖的父属性$pa_i$，就可以直接估计概率值$P(x_i|c,pa_i)$，问题的关键就转化为如何确定每个属性的父属性。

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210408210943.png" alt="image-20210408210936491" style="zoom:80%;" /></div>

**SPODE**（Super-Parent ODE）

最直接的方法是假设所有属性都依赖于同一个“超父”（super-parent）属性，然后通过交叉验证等模型选择的方法确定超父属性。

**TAN**（Tree Augmented naive Bayes）

TAN算法则是在最大加权生成树（maximum weighted spanning tree）算法的基础上，通过以下步骤将属性间的依赖关系简化为树形结构：

（1）计算任意两个属性之间的条件互信息（conditional mutual information）
$$I(x_i,x_j|y)=\sum_{x_i,x_j;c\in\mathcal Y}P(x_i,x_j|c)\log\frac{P(x_i,x_j|c)}{P(x_i|c)P(x_j|c)}$$
（2）以属性为节点构建完全图，任意两个节点之间边的权重设为$I(x_i,x_j|y)$；
（3）构建此完全图的最大加权生成树，挑选根变量，将边置为有向
（4）加入类别节点$y$，增加从$y$到每个属性的有向边

其中，条件互信息$I(x_i,x_j|y)$刻画了两个属性在已知类别下的相关性，因此通过最大生成树方法，TAN保留了强相关属性之间的依赖性

**AODE**（Averaged One-Dependent Estimator）

AODE是一种基于集成学习机制的独依赖分类器，和SPODE通过模型选择选父属性不同，AODE尝试将每个属性作为超父属性来构建SPODE，然后将那些具有足够训练数据支撑的SPODE集成作为最终结果，即

$$P(c|x)\propto\sum^d_{i=1,|D_{x_i}|\ge m'}P(c,x_i)\prod^d_{j=1}P(x_j|c,x_i)$$

其中$D_{x_i}$表示第$i$个属性上取值为$x_i$的样本的集合，$m'$为阈值常数（默认为30）。显然AODE需要估计$P(c,x_i)$和$P(x_j|c,x_i)$

$$\hat P(c,x_i)=\frac{|D_{c,x_i}+1}{|D|+N\times N_i}\\
\hat P(x_j|c,x_i)=\frac{|D_{c,x_i,x_j}|+1}{|D_{c,x_i}|+N_j}$$

与NB和SPODE相比，AODE无需模型选择，可以通过预计算节省预测时间，也能采取懒惰学习的方式在预测时在进行计数，易于实现增量学习。

考虑变量的高阶依赖，准确估计概率所需的训练样本数量将以指数级增加，如果训练数据非常充分，泛化性能有可能提升，但在样本有限的条件下效果仍然不好。

# 7.5 贝叶斯网

贝叶斯网（Bayesian network）又称“信念网”（belief network），它借助有向无环图（Directed Acyclic Graph，DAG）来刻画属性之间的依赖关系，并使用条件概率表（Conditional Probability Table，CPT）来描述属性的联合概率分布。

具体来说，一个贝叶斯网$B$由结构$G$和参数$\Theta$两部分组成，$G$是一个有向无环图，每个节点对应一个属性，如果它们之间有依赖关系，就用一条边连接起来；$\Theta$定量地描述了依赖关系。假设$x_i$在$G$中的父节点集为$\pi_i$，则$\Theta$包含了每个属性的条件概率表示$\theta_{x_i|\pi_i}=P_B(x_i|\pi_i)$，如图所示

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210409165640.png" alt="image-20210409165633631" style="zoom:80%;" /></div>

## 7.5.1 结构

贝叶斯网结构表达了属性间的条件独立性，它假设每个属性和非后裔属性独立，有联合概率分布

$$P_B(x_1,x_2,...,x_d)=\prod^d_{i=1}P_B(x_i|\pi_i)=\prod^d_{i=1}\theta_{x_i|\pi_i}$$

例如上图，有

$$P(x_1,x_2,x_3,x_4,x_5)=P(x_1)P(x_2)P(x_3|x_1)P(x_4|x_1,x_2)P(x_5|x_2)$$

下图展现了贝叶斯网中常见的三种结构关系：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210409171938.png" alt="image-20210409171938039" style="zoom:80%;" /></div>

“同父”（common parent）结构中$x_3$和$x_4$在给定$x_1$时独立，记为$x_3\perp x_4|x_1$。“顺序”结构中，给定$x$的值，有$y$和$z$条件独立。而V型结构（V-structure）又称“冲撞”结构，给定$x_4$后，$x_1$和$x_2$必不独立，但$x_4$未知时二者则相互独立，这种独立性称为“边际独立性”（marginal independent）。由此可见，三种结构中，一个变量取值是否确定会影响另外两个变量间的独立性。

为了分析有向图中变量的条件独立性，可使用“有向分离”（directed-seperation），我们先把有向图中所有的V型结构的两个父节点之间加上一条无向边，再将所有有向边改为无向边，使得有向图转变为一个无向图。这样产生的无向图叫“道德图”（moral graph），父节点相连的过程称为“道德化”（moralization）。（道德图将同父结构和顺序结构转化成一种形式，从条件独立的性质上看二者是等价的。）

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210409173802.png" alt="image-20210409173802615" width=500 /></div>

假定道德图中有变量$x$和$y$以及集合$\mathbf z$，如果$x$和$y$能被$\mathbf z$分开，则称$x$和$y$被$\mathbf z$有向分离，$x\perp y|\mathbf z$成立。

## 7.5.2 学习

如果知道了网络结构，只需要进行计数估算条件概率表即可，因此贝叶斯网学习的首要任务是找出结构最恰当的贝叶斯网，常用“评分搜索”的方法。我们首先定义一个能够衡量贝叶斯网和训练数据契合程度的评分函数（score function）。

常用评分函数一般基于信息论准则，这类准则将学习问题看作一个数据压缩任务，学习目标是找到一个能以最短编码长度描述训练数据的模型，编码长度包括描述模型的编码位数和使用模型描述数据所需的编码位数。每个贝叶斯网描述了一个在训练数据上的概率分布，我们应选择综合编码长度最短的贝叶斯网，也就是“最小描述长度”（Minimal Description Length，MDL）准则。

给定训练集$D$，贝叶斯网$B=\lang G,\Theta\rang$在$D$上的评分函数可写为

$$s(B|D)=f(\theta)|B|-LL(B|D)$$

其中$|B|$是贝叶斯网络的参数个数，$f(\theta)$表示参数$\theta$所需的编码位数，同时有$LL(B|D)=\sum^m_{i=1}\log P_B(x_i)$，显然它是贝叶斯网的对数似然。评分函数的第一项是计算编码贝叶斯网所需的编码位数，第二项计算了概率分布$P_B$对$D$描述的效果。

如果$f(\theta)=1$，即每个参数用1编码位，则会得到AIC（Akaike Information Criterion）评分函数：

$$AIC(B|D)=|B|-LL(B|D)$$

如果$f(\theta)=\frac12\log m$，则会得到BIC（Bayesian Information Criterion）评分函数：

$$BIC(B|D)=\frac{\log m}2|B|-LL(B|D)$$

如果$f(\theta)=0$，则评分函数退化为负对数似然函数，学习任务退化为极大似然估计。

由此可见，在网络结构固定时，评分函数第一项为常数，最小化$s(B|D)$等价于计算参数$\Theta$的极大似然估计，可以直接在训练数据上估计获得。然而，从所有可能的网络结构中搜索最优贝叶斯网结构是一个NP难问题，有两种常用策略可以求得近似解：（1）贪心法，从某个网络结构出发，每次调整一条边，直到评分函数值不再下降位置；（2）通过给网络结构增加约束（如树形结构）来削减搜索空间。

## 7.5.3 推断

贝叶斯网训练好之后就能回答“查询”（query）问题，通过已知变量观测值推测查询变量的过程叫做“推断”（inference），已知变量观测值称为“证据”（evidence）。直接通过贝叶斯网的条件概率分布计算后验概率是NP难的，此时需采用“近似推断”的方法，常用吉布斯抽样（Gibbs sampling）。

记$\mathbf Q=\{Q_1,...,Q_n\}$表示待查询变量，而$\mathbf E=\{E_1,...,E_k\}$为证据变量，其取值为$\mathbf e=\{e_1,...,e_k\}$，目标是计算后验概率$P(\mathbf Q=\mathbf q|\mathbf E=\mathbf e)$，算法如下：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210409224022.png" alt="image-20210409224021439" width=500 /></div>

吉布斯抽样算法在贝叶斯网状态空间中进行“随机漫步”（random walk），首先随机产生一个$\mathbf E=\mathbf e$的$q^0$样本作为初始，然后每一步依次遍历$\mathbf Q$中的变量，固定当前$\mathbf{q}^{t-1}_i$并根据其他变量当前取值计算概率抽样得到新的$\mathbf q^t_i$，假设$T$次采样得到的与$\mathbf q$一致的有$n_q$个，则可以近似估算出后验概率

$$P(\mathbf Q=\mathbf q|\mathbf E=\mathbf e)\simeq\frac{n_q}T$$

上述步骤形成的$\mathbf q^t$实际上是一个马尔科夫链（Markov chain），不可约非周期的马氏链在$t\rightarrow\infty$时收敛与平稳分布（stationary distribution），即$P(\mathbf Q=\mathbf q|\mathbf E=\mathbf e)$。

值得一提的是，马氏链往往需要很长时间才能趋于收敛，因此吉布斯抽样的算法收敛时间较慢，同时，如果贝叶斯网中存在极端概率0或1（可能不满足不可约非周期条件），则马氏链可能不存在平稳分布。

# 7.6 EM算法

现实生活中往往会存在不完整的训练样本变量，又称“隐变量”（latent variable），令$\mathbf X$表示已观测变量集，$\mathbf Z$表示隐变量集，$\Theta$表示模型参数。如果要对$\Theta$进行极大似然估计，则应最大化

$$LL(\Theta|\mathbf X,\mathbf Z)=\log P(\mathbf X,\mathbf Z|\Theta)$$

然而$\mathbf Z$是隐变量所以无法直接求解，但我们可以通过求$\mathbf Z$的期望，来最大化已经观测到的数据的对数“边际似然”（marginal likelihood）

$$LL(\Theta|\mathbf X)=\log P(\mathbf X|\Theta)=\log\sum_{\mathbf Z}P(\mathbf X,\mathbf Z|\Theta)$$

EM（Expectation-Maximization）算法常用于估计参数隐变量，基本思想是：如果$\Theta$已知，则可以推断出最优隐变量$\mathbf Z$的值；反之，如果$\mathbf Z$已知，则可以对参数$\Theta$进行极大似然估计。所以可以以初始值$\Theta^0$为起点，迭代执行以下步骤：
（1）基于$\Theta^t$推断隐变量$\mathbf Z$的期望，记为$\mathbf Z^t$
（2）基于已观测变量$\mathbf X$和$\mathbf Z^t$对参数$\Theta$做极大似然估计，记为$\Theta^{t+1}$

进一步地，如果我们不是取$\mathbf Z$的期望，而是计算其概率分布$P(\mathbf Z|\mathbf X,\Theta^t)$，则有：

（1）E步（Expectation）：以当前参数$\Theta^t$推断隐变量分布$P(\mathbf Z|\mathbf X,\Theta^t)$，并计算对数似然函数$LL(\Theta|\mathbf X,\mathbf Z)$关于$\mathbf Z$的期望
$$Q(\Theta|\Theta^t)=\mathbb E_{\mathbf Z|\mathbf X,\Theta^t}LL(\Theta|\mathbf X,\mathbf Z)$$
（2）M步（Maximization）：寻找参数最大化期望似然
$$\Theta^t=\arg\max_{\Theta}Q(\Theta|\Theta^t)$$

EM算法第一步是期望步，利用当前估计的参数值来计算对数似然的期望值；第二步是最大化步，寻找能使E步产生的似然期望最大化的参数值，并进行循环迭代。而隐变量估计也可以使用梯度下降等方法进行求解，但求和项数会随着隐变量的数目以指数级上升，EM算法则可以看做一种非梯度优化方法（实际上可看作坐标下降法最大化对数似然下界）。

简单来讲，如果知道每个数据的标签，则可以直接使用极大似然估计标签对应的参数，但在不知道时，我们可以计算不同标签的概率，并以这一概率计算条件期望，这相当于把每条样本对应的数据按照概率期望分配给了每一个标签，从而相当于知道了每个数据对应的标签，便可直接使用极大似然估计并循环迭代。而算法收敛的证明用到了Jensen不等式，详见https://zhuanlan.zhihu.com/p/40991784。

# 7.7 扩展阅读

为避免贝叶斯定理求解时的组合爆炸和样本稀疏问题，朴素贝叶斯分类器引入了属性条件独立性假设，这个假设在现实中往往很难成立，但效果却仍然不错，推测可能是因为只要各类别条件概率排序正确无需精确概率值即可产生正确分类，或者属性间依赖关系可能相互抵消。NB分类器在信息检索领域尤为常用。

根据属性间依赖关系的涉及程度，贝叶斯分类器形成了一个“谱”，NB不考虑任何属性依赖，贝叶斯网则可以表示任意属性间的依赖，而介于二者之间的半朴素贝叶斯分类器则可以基于各种假设和约束，如ODE只依赖一个父属性，以及TAN、AODE等，kDE则是最多依赖k个父属性，形成KDB等。

贝叶斯分类器（Bayesian Classifier）和贝叶斯学习（Bayesian Learning）有显著区别，前者是通过最大后验概率进行单点估计，后者则是进行分布估计。

贝叶斯网给不确定学习和推断提供了基本框架，它的表示能力和可解释性都比较好。贝叶斯网的学习分为参数学习和结构学习，参数学习通常较为简单，而对结构学习，人们往往使用多种评分搜索方法。另外，也有一些关于判别式学习贝叶斯网的研究。

EM算法是最常见的隐变量估计方法，常被用来学习高斯混合模型（Gaussian mixture model，GMM）。k-Means就是典型的EM算法。