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

