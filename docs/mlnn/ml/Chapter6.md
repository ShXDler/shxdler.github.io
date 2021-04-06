# 6.1 间隔与支持向量

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210405235223.png" alt="image-20210405235223939" style="zoom:80%;" /></div>

对于两类样本，可能存在多种划分超平面，直观上来讲选择最中间的划分超平面的鲁棒性会更好。记超平面方程为：

$$w^\top x+b=0$$

样本空间任意点$x$到超平面距离为：

$$r=\frac{|w^\top x+b|}{||w||}$$

假设超平面可以进行正确分类，可以对$w$和$b$进行放缩变换，以满足：

$$\left\{\begin{aligned}
w^\top x_i+b\ge+1 \\
w^\top x_i+b\le-1
\end{aligned}\right.$$

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210406113412.png" width=500 /></div>

如图所示，距离超平面最近的几个训练样本点使得等号成立，成为“支持向量”（support vector），两个异类支持向量到超平面的距离和被称为“间隔”（margin）：

$$\gamma=\frac2{||w||}$$

想找到“最大间隔”（maximum margin）的划分超平面，就是要找到$w$和$b$使得$\gamma$最大，即：

$$\max_{w,b}\frac2{||w||}$$

$$s.t. y_i(w^\top x_i+b)\ge1,i=1,2,...,m$$

等价于

$$\min_{w,b}\frac12{||w||^2}$$

$$s.t. y_i(w^\top x_i+b)\ge1,i=1,2,...,m$$

这称为支持向量机（Support Vector Machine，SVM）的原型。

# 6.2 对偶问题

上述优化问题实际上是凸二次规划（convex quadratic programming）问题，可以使用现成的优化方法，但也有更高效的求解办法。我们使用拉格朗日乘子法可以得到“对偶问题”（dual problem），具体来说，对每条约束添加拉格朗日乘子$\alpha_i\ge0$，则：

$$L(w,b,\alpha)=\frac12||w||^2+\sum^m_{i=1}\alpha_i(1-y_i(w^\top x_i+b))$$

求解得

$$w=\sum^m_{i=1}\alpha_iy_ix_i$$

$$0=\sum^m_{i=1}\alpha_iy_i$$

有对偶问题

$$\max_\alpha\sum^m_{i=1}\alpha_i-\frac12\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_iy_jx_i^\top x_j$$

$$s.t. \sum^m_{i=1}\alpha_iy_i=0,\alpha_i\ge0,i=1,2,...,m$$

使用SMO方法（见下文）解出$\alpha_i$后，可求出$w$和$b$得到：

$$f(x)=w^\top x+b=\sum^m_{i=1}\alpha_iy_ix_i^\top x+b$$

上述过程需满足KKT（Karush-Kuhn-Tucker）条件：

$$\left\{\begin{aligned}
&\alpha_i\ge0 \\
&y_if(x_i)-1\ge0\\
&\alpha_i(y_if(x_i)-1)=0\\
\end{aligned}\right.$$

得$\alpha_i=0$或$y_if(x_i)=1$。如果$\alpha_i=0$，则该样本不会影响$f(x)$；而如果$\alpha_i>0$，则$y_if(x_i)=1$，对应样本在最大间隔边界上。可以发现，训练完成后大部分训练样本都不需要保留，最终模型只和支持向量有关。

**SMO算法**

上面的对偶问题是一个二次规划问题，样本数很多时采用通用方法可能产生很大的计算开销，使用SMO（Sequential Minimal Optimization）可以有效解决这一问题。

SMO的基本思路如下，不断迭代这两个步骤直到收敛：

（1）选取一对需要更新的变量$\alpha_i$和$\alpha_j$

（2）固定除这二者之外的参数，求解上式得到更新后的$\alpha_i$和$\alpha_j$

$\alpha_i$和$\alpha_j$中只要有一个不满足KKT条件，目标函数就会在迭代后增大，且违背KKT的程度越大函数值的增幅也相对越大，所以SMO先选取违背KKT条件最大的变量，第二个变量选择与第一个间隔最大的变量，可以使目标函数增长更快。

而SMO算法之所以高效是因为每次只优化两个参数：

$$\alpha_iy_i+\alpha_jy_j=c,\alpha_i\ge0,\alpha_j\ge0$$

$$c=-\sum_{k\ne i,j}\alpha_ky_k $$

不难发现$c$是个常数，将该约束条件代入对偶问题中，即可将其转化为单变量二次规划问题，可以直接求出解析解。

而对偏移项$b$，记$S$为支持向量的下标集，有：

$$y_s(\sum_{i\in S}\alpha_iy_ix_i^\top x_s+b)=1$$

理论上可以直接求解，现实中我们往往采用求取平均值的做法，它的鲁棒性更好：

$$b=\frac1{|S|}\sum_{s\in S}(\frac1{y_s}-\sum_{i\in S}\alpha_iy_ix_i^\top x_s)$$

# 6.3 核函数

对于不是线性可分的问题（如异或问题），可以将样本从原始空间映射到更高维的特征空间里：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210406122330.png" alt="image-20210406122330792" style="zoom:80%;" /></div>

记$\phi(x)$为映射后的特征向量，新模型为：

$$f(x)=w^\top\phi(x)+b$$

有目标函数

$$\min_{w,b}\frac12||w||^2$$

$$s.t. y_i(w^\top\phi(x_i)+b)\ge1,i=1,2,...,m$$

有对偶问题

$$\max_\alpha\sum^m_{i=1}\alpha_i-\frac12\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_iy_j\phi(x_i)^\top \phi(x_j)$$

$$s.t. \sum^m_{i=1}\alpha_iy_i=0,\alpha_i\ge0,i=1,2,...,m$$

求解上式需要计算$\phi(x_i)^\top\phi(x_j)$，它的维数甚至可能是无穷维，计算十分困难，可以记核函数（kernel function）

$$\kappa(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle=\phi(x_i)^\top\phi(x_j)$$

求解得到

$$f(x)=w^\top\phi(x)+b=\sum_{i=1}^m\alpha_iy_i\kappa(x,x_i)+b$$

上式可通过训练样本的核函数展开，又称“支持向量展式”（support vector expansion）。

现实任务中我们往往不知道$\phi(\cdot)$是什么形式，关于核函数有如下定理：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210406143015.png" alt="image-20210406143015232" style="zoom:80%;" /></div>

因此只要一个对称函数所对应的核矩阵半正定，它就能作为核函数使用，对于任何一个半正定核矩阵，都能找到一个与之对应的$\phi$，换言之，任何一个核函数都隐式地定义了一个“再生核希尔伯特空间”（Reproducing Kernel Hilbert Space，RKHS）的特征空间。

下表是几种常用的核函数：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210406143348.png" alt="image-20210406143348014" style="zoom:80%;" /></div>

另外，下面几种形式也均为核函数：

$$\gamma_1\kappa_1+\gamma_2\kappa_2,\gamma_1>0,\gamma_2>0\\\kappa_1\otimes\kappa_2(x,z)=\kappa_1(x,z)\kappa_2(x,z)\\\kappa(x,z)=g(x)\kappa_1(x,z)g(z)$$

# 6.4 软间隔与正则化

现实任务中可能很难确认合适的核函数，即使找到了也无法确定是不是有过拟合造成的，因此引入了“软间隔”（soft margin），允许一些样本出错：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210406144427.png" alt="image-20210406144427688" style="zoom:80%;" /></div>

前面的模型要求所有样本都满足约束，成为“硬间隔”（hard margin）。软间隔允许某些样本不满足约束以最大化间隔，当然不满足的样本数也应当尽可能少，得到优化目标

$$\min_{w,b}(\frac12||w||^2+C\sum^m_{i=1}\ell_{0/1}(y_i(w^\top x_i+b)-1))\\where\ C>0,\ell_{0/1}(z)=\left\{\begin{aligned}1\ \ \ ,if\ z<0;\\0,otherwise. \end{aligned}\right.$$

这里的$C$是一个正常数，$\ell_{0/1}(z)$是“0/1损失函数”，当$C$无穷大时，会迫使所有样本满足约束，取有限值则可以允许一些样本不满足约束。然而$\ell_{0/1}(z)$非凸、非连续，常用其他函数代替（替代损失，surrogate loss），它们通常是凸的连续函数并且是$\ell_{0/1}$的上界：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210406151720.png" alt="image-20210406151720567" style="zoom:80%;" /></div>

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210406151733.png" alt="image-20210406151733644" style="zoom:80%;" /></div>

如果使用hinge损失，优化目标变成

$$\min_{w,b}(\frac12||w||^2+C\sum^m_{i=1}\max(0,1-y_i(w^\top x_i+b)))$$

引入“松弛变量”（slack variables）$\xi_i\ge0$，得

$$\min_{w,b,\xi_i}\frac12||w||^2+C\sum^m_{i=1}\xi_is\\s.t. \xi_i\ge1-y_i(w^\top x_i+b)，\xi_i\ge0,i=1,2,...,m$$

这就是常用的“软间隔支持向量机”。每个样本都对应一个松弛变量。上述问题仍是一个二次规划问题，有拉格朗日函数

$$L(w,b,\alpha,\xi,\mu)=\frac12||w||^2+C\sum^m_{i=1}\xi_i+\sum^m_{i=1}\alpha_i(1-\xi_i-y_i(w^\top x_i+b))-\sum^m_{i=1}\mu_i\xi_i$$

求导

$$w=\sum^m_{i=1}\alpha_iy_ix_i\\
0=\sum^m_{i=1}\alpha_iy_i\\
C=\alpha_i+\mu_i$$

代入得到对偶问题

$$\max_\alpha\sum^m_{i=1}\alpha_i-\frac12\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_iy_jx_i^\top x_j\\s.t. \sum^m_{i=1}\alpha_iy_i=0,0\le\alpha_i\le C,i=1,2,...,m$$

而KKT条件要求：

$$\left\{\begin{aligned}&\alpha_i\ge0,\mu_i\ge0\\&y_if(x_i)-1+\xi_i\ge0\\&\alpha_i(y_if(x_i)-1+\xi_i)=0\\&\xi_i\ge0,\mu_i\xi_i=0\end{aligned}\right.$$

所以对于任何样本都有：

$$\left\{\begin{aligned}&\alpha_i=0\rightarrow样本无影响\\
&\alpha_i>0,y_if(x_i)=1-\xi_i\rightarrow样本是支持向量\left\{\begin{aligned}\alpha_i&<C\rightarrow\mu_i>0\rightarrow\xi_i=0，在最大间隔边界上\\
\alpha_i&=C\rightarrow\mu_i=0\rightarrow\left\{\begin{aligned}\xi_i&\le1，在最大间隔内部\\
\xi_i&>1，错误分类
\end{aligned}\right.\end{aligned}\right.\end{aligned}\right.$$

不难发现软间隔支持向量机的最终模型仍只和支持向量有关，hinge损失函数保留了稀疏性。

**其他替代损失函数**

使用$\ell_{\log}$可以近似得到logistic回归模型，通常情况下二者性能相当，logistic回归模型能够输出概率，并且可以直接用于多分类任务，支持向量机则需要进行推广。而另一方面，hinge损失的平坦区域使得支持向量机的解具有稀疏性，而logit函数是光滑的递减函数，没有支持向量的概念，因此logistic模型依赖于更多的训练样本。

我们还可以使用别的替代损失函数，得到模型的性质各不相同，但是有一个共性：优化目标函数中的第一项用于划分超平面的“间隔”，另一部分用来描述训练集上的误差：

$$\min_f(\Omega(f)+C\sum^m_{i=1}\ell(f(x_i),y_i))$$

第一项$\Omega(f)$称为“结构风险”（structural risk），用于描述模型$f$的性质；而第二项$\sum^m_{i=1}\ell(f(x_i),y_i)$称为“经验风险”（empirical risk），用于描述模型和训练数据的契合程度。从经验风险最小化的角度来看，$\Omega(f)$表明了我们倾向的模型，上述目标也被称作“正则化”（regularization）问题，$\Omega(f)$称为正则化项。常用的正则化项是${\rm L}_p$范数（norm），${\rm L}_2$范数$||w||_2$倾向于$w$的分量取值均衡，即非零分量尽可能稠密；而${\rm L}_1$范数和${\rm L}_0$范数则倾向于非零分量尽可能稀疏。

# 6.5 支持向量回归

和传统回归模型追求$f(x)$和$y$相等不同，支持向量回归（Support Vector Regression，SVR）能容忍二者存在$\epsilon$的偏差，只有大于这个值时才计算损失：

<div align="center"’><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210406194641.png" alt="image-20210406194641437" style="zoom:80%;" /></div>

SVR问题可转化为：

$$\min_{w,b}\frac12||w||^2+C\sum^m_{i=1}\ell_\epsilon(f(x_i)-y_i)$$

其中$\ell_\epsilon$是“$\epsilon$-不敏感损失”（$\epsilon$-insensitive loss）函数：

$$\ell_\epsilon(z)=\left\{\begin{aligned}&0,&if\ |z|\le\epsilon\\
&|z|-\epsilon,&otherwise\end{aligned}\right.$$

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210406195538.png" alt="image-20210406195538324" style="zoom:80%;" /></div>

引入松弛变量$\xi_i$和$\hat\xi_i$，得

$$\min_{w,b,\xi_i,\hat\xi_i}\frac12||w||^2+C\sum^m_{i=1}(\xi_i+\hat\xi_i)\\
s.t.\begin{aligned}&f(x_i)-y_i\le\epsilon+\xi_i\\
&y_i-f(x_i)\le\epsilon+\hat\xi_i\\
&\xi_i\ge0,\hat\xi_i\ge0,i=1,2,...,m\end{aligned}$$

继续使用拉格朗日乘子法可以得到

$$\begin{aligned}L(w,b,\alpha,\hat\alpha,\xi,\hat\xi,\mu,\hat\mu)&=\frac12||w||^2+C\sum^m_{i=1}(\xi_i+\hat\xi_i)-\sum^m_{i=1}\mu_i\xi_i-\sum^m_{i=1}\hat\mu_i\hat\xi_i\\&+\sum_{i=1}^m\alpha_i(f(x_i)-y_i-\epsilon-\xi_i)+\sum^m_{i=1}\hat\alpha_i(y_i-f(x_i)-\epsilon-\hat\xi_i)\end{aligned}$$

求导得

$$w=\sum^m_{i=1}(\hat\alpha_i-\alpha_i)x_i,\\
0=\sum^m_{i=1}(\hat\alpha_i-\alpha_i),\\
C=\alpha_i+\mu_i,\\
C=\hat\alpha_i+\hat\mu_i$$

代入SVR得到对偶问题：

$$\max_{\alpha,\hat\alpha}(\sum^m_{i=1}y_i(\hat\alpha_i-\alpha_i)-\epsilon(\hat\alpha_i+\alpha_i)-\frac12\sum^m_{i=1}\sum^m_{j=1}(\hat\alpha_i-\alpha_j)x_i^\top x_j)\\
s.t.\sum^m_{i=1}(\hat\alpha_i-\alpha_i)=0,0\le\alpha_i,\hat\alpha_i\le C$$

KKT条件要求

$$\left\{\begin{aligned}\alpha_i(f(x_i)-y_i-\epsilon-\xi_i)=0,\\
\hat\alpha_i(y_i-f(x_i)-\epsilon-\hat\xi_i)=0,\\
\alpha_i\hat\alpha_i=0,\xi_i\hat\xi_i=0,\\
(C-\alpha_i)\xi_i=0,(C-\hat\alpha_i)\hat\xi_i=0\end{aligned}\right.$$

与前两种情况类似，当且仅当$f(x_i)-y_i-\epsilon-\xi_i=0$时，有$\alpha_i>0$；当且仅当$f(x_i)-y_i-\epsilon-\hat\xi_i=0$时，有$\hat\alpha_i>0$；也就是说只有样本$(x_i,y_i)$不落入间隔带中，$\alpha_i$和$\hat\alpha_i$才能取非零值。另外$\alpha_i$和$\hat\alpha_i$至少有一个为0，所以这两个约束不能同时成立。

对每个样本都有

$$b_i=y_i+\epsilon-\sum^m_{j=1}(\hat\alpha_j-\alpha_j)x_j^\top x_i$$

实践中常选取多个（或所有）满足$0<\alpha_i<C$的样本求解并取平均值。

得到SVR解为

$$f(x)=\sum^m_{i=1}(\hat\alpha_i-\alpha_i)x_i^\top x+b$$

使得$(\hat\alpha_i-\alpha_i)\ne0$的向量即为SVR的支持向量，他们一定落在$\epsilon$-间隔带之外，所以SVR算法具有一定的稀疏性。（不过大部分向量是不是都是支持向量？）

如果考虑特征映射形式，得到的SVR形式为

$$f(x)=\sum^m_{i=1}(\hat\alpha_i-\alpha_i)\kappa(x,x_i)+b$$

# 6.6 核方法

我们发现无论SVM还是SVR，模型都能表示成$\kappa(x,x_i)$的线性组合，我们有如下“表示定理”（representer theorem）结论：

<div align="center"><img src="https://picgo-1305404921.cos.ap-shanghai.myqcloud.com/20210406212727.png" alt="image-20210406212727694" style="zoom:80%;" /></div>

这表明，对一般的损失函数和单调递增的正则化项，最优解都可以表示成核函数的线性组合。基于这一定理，人们发展了基于核函数的方法——“核方法”（kernel methods），将线性学习器拓展为非线性学习器，如“核线性判别分析”（Kernelized Linear Discriminant Analysis，KLDA）。

我们先假设使用映射$\phi:{\mathcal X}\mapsto{\mathbb F}$，在$\mathbb F$中执行线性判别分析，有

$$h(x)=w^\top\phi(x)$$

而KLDA的目标函数为

$$\max_wJ(w)=\frac{w^\top S_b^\phi w}{w^\top S_w^\phi w}$$

而第$i$类样本的均值为

$$\mu_i^\phi=\frac1{m_i}\sum_{x\in X_i}\phi(x)$$

得到散度矩阵

$$S_b^\phi=(\mu_1^\phi-\mu_0^\phi)(\mu_1^\phi-\mu_0^\phi)^\top\\
S_w^\phi=\sum^1_{i=0}\sum_{x\in X_i}(\phi(x)-\mu_i^\phi)(\phi(x)-\mu_i^\phi)^\top$$

由表示定理，得

$$h(x)=\sum^m_{i=1}\alpha_i\kappa(x,x_i)$$

进而

$$w=\sum^m_{i=1}\alpha_i\phi(x_i)$$

记$\bf K$为核函数$\kappa$对应的核矩阵，$({\bf K})_{ij}=\kappa(x_i,x_j)$，记${\mathbb I}_i\in\{1,0\}^{m\times1}$为第$i$类样本的示性向量，如果$x_j\in X_i$，则第$j$个分量为1，否则为0，令

$$\hat\mu_0=\frac1{m_0}{\bf K1}_0\\
\hat\mu_1=\frac1{m_1}{\bf K1}_1\\
{\bf M}=(\hat\mu_0-\hat\mu_1)(\hat\mu_0-\hat\mu_1)^\top\\
{\bf N}={\bf KK^\top}-\sum^1_{i=0}m_i\hat\mu_i\hat\mu_i^\top$$

则目标函数可以转化成

$$\max_\alpha J(\alpha)=\frac{\bf\alpha^\top M\alpha}{\bf\alpha^\top N\alpha}$$

使用拉格朗日乘子法即可求解。

# 6.7 拓展阅读

线性核SVM仍是文本分类的首选技术，因为属性空间维度较高，冗余度很大。

SVM的求解通常借助于凸优化技术，对线性核SVM，有基于割平面法的$\rm SVM^{perf}$具有线性时间复杂度，基于随机梯度下降的Pegasos甚至更快，坐标下降法在稀疏数据上有很高的效率。非线性核SVM时间复杂度理论上不可能低于$O(m^2)$，一些快速近似方法有如基于采样的CVM、基于低秩逼近的Nystrom、基于随机傅里叶特征的方法等等。

支持向量机作为二分类方法，如果想要多分类或者结构输出需要进行扩展。

核函数的选择问题仍然没有得到解决，多核学习（multiple kernel learning）可以使用多个核函数进行最优凸组合得到最终函数，这实际上也是集成学习的机制。

替代损失函数存在“一致性”（consistency）问题，即这种方法得到的是否还是原本的解？有研究证明了几种常见替代损失函数的一致性，给出了基于替代损失函数进行经验风险最小化的一致性充要条件。