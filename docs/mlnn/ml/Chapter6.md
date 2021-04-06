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
\alpha_i\ge0 \\
y_if(x_i)-1\ge0\\
\alpha_i(y_if(x_i)-1)=0\\
\end{aligned}\right.$$

得$\alpha_i=0$或$y_if(x_i)=1$。如果$\alpha_i=0$，则该样本不会在求$f(x)$的和中出现，也就不会影响$f(x)$，而如果$\alpha_i>0$，则$y_if(x_i)=1$，对应样本在最大间隔边界上，可以发现，训练完成后大部分训练样本都不需要保留，最终模型只和支持向量有关。

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

$$\left\{\begin{aligned}\alpha_i\ge0,\mu_i\ge0\\y_if(x_i)-1+\xi_i\ge0\\\alpha_i(y_if(x_i)-1+\xi_i)=0\\\xi_i\ge0,\mu_i\xi_i=0\end{aligned}\right.$$

所以对于任何样本都有$\alpha_i=0$或$$