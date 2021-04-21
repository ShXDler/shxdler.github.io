*原文地址：ACL2019-*[Barack's Wife Hillary: Using Knowledge-Graphs for Fact-Aware Language Modeling](https://arxiv.org/abs/1906.07241)-*Robert Logan, Nelson F. Liu, Matthew E. Peters, Matt Gardner and Sameer Singh*

## 1.简介

在语言生成上，现有模型已经很好地做到了语法上的连贯性，在一些常识和基本知识的推断上也表现不错。然而，这些模型往往最多只能记住训练中提到的事实，所以它们在稀缺或未知领域上生成语句的“事实正确”（*factually correct*）能力仍有待提升。作者用一个生动的例子描述了这一现象，在下图的语句中，AWD-LSTM模型认为“Play Station”的可能性比“Game Boy”要高（而实际上该处应该是Game Boy），产生了事实上的错误，其根源在于这一类模型往往是在整体词汇空间上对概率分布进行概率表示。

<div align="center">
    <img src="https://pic1.zhimg.com/v2-30aa69a648468d9f4db171e948e48218_b.png">  
</div>

<center style="color:#C0C0C0;text-decoration:underline">图1 示例语句</center>

本文提出了知识图谱语言模型（KGLM），利用知识图谱的外部知识进行语言的生成。KGLM模型使用了一个动态的局部知识图谱，局部图谱包含了已经被提到过的实体及其相关实体。在生成语句时，KGLM可以提供新的实体来扩增局部图谱，也可以从现存的局部图谱中选择实体提供给语句生成。在这一过程中，模型将局部图谱和标准词汇表结合起来，从而可以提供数字、日期及其他不常见的实体。

除了KGLM模型，作者还介绍了名为*Linked Wikitext-2*的数据集，该数据集与Wikidata紧密相关，将人为提供的连接和现有连接结合起来。而在实体之间，作者也建立起相应的关系将它们连接起来。

从结果来看，KGLM模型和AWD-LSTM相比在整体上拥有较低的困惑度，尤其是未知领域导致的困惑度。同时，在事实完成（*factual completion*）方面，作者也使用了一系列如题目所示的例子：“*Barack is married to _____.*”，结果发现KGLM也有着更高的准确度。另外，KGLM能够准确地生成稀缺实体，并且可以通过修改知识图谱的方法进行人为控制。

## 2.知识图谱语言模型（KGLM）

传统意义上的语言模型计算的都是在已知部分文本内容时单词出现的条件概率$p(x_t|x_{<t})$ ，以循环神经网络（RNN）为例：

$$p(x_t|x_{<t})={\rm softmax}({\bf W}_h{\bf h}_t+{\bf b})\\ {\bf h}_t={\rm RNN}({\bf h}_{t-1},{\rm x}_{t-1})$$

本文使用了LSTM进行循环模块的计算。

而知识图谱则是有向标注图，包含实体和实体间的关系： 

$${\mathcal KG}=\{(p,r,e)|p\in{\mathcal E},r\in{\mathcal R},e\in{\mathcal E}\}$$ 

而本文使用的局部知识图谱使用的是一个子集：

$${\mathcal KG}_{<t}=\{(p,r,e)|p\in{\mathcal E}_{<t},r\in{\mathcal R},e\in{\mathcal E}\}$$

### 2.1 语言生成

KGLM的主要目标是从知识图谱中生成实体和事实，它会首先将上文中已经出现的实体囊括进局部知识图谱中，以便生成上文中已经出现过的事实。同时会进一步将能够反映新实体的额外实体和事实加入局部知识图谱。正式来讲，模型计算 $p(x_t,{\mathcal E}_t|x_{<t},{\mathcal E}_{<t})$ 的过程如下：

①决定 $x_t$ 的类型，记做 $t_t$ 。它反映了一个实体是否在 ${\mathcal KG}_{<t}$ 内或者不是一个实体

a. 如果 $t_t={\sf new}$ ，则从全体实体空间 $\mathcal E$ 中选择下一个实体 $e_t$ 

b. 如果 $t_t=\sf related$ ，则首先从 ${\mathcal E}_{<t}$ 中选择父实体 $p_t$ ，并从 $\{(p,r,e)\in{\mathcal KG}_{>t}|p=p_t\}$ 中选择事实关系 $r_t$ ，再从 $\{e|(p_t,r_t,e)\in{\mathcal KG}_{<t}\}$ 中选择尾实体 $e_t$ 

c. 如果 $t_t= \emptyset$ ，则 $e_t=\emptyset$ 

②在 $e_t$ 的条件下生成 $x_t$ 

③如果 $e_t\notin{\mathcal E}_{<t}$ ，则扩增局部图谱： ${\mathcal E}_{<(t+1)}\leftarrow{\mathcal E}_{<t}\cup e_t$ ；否则保持不变 ${\mathcal E}_{<(t+1)}\leftarrow{\mathcal E}_{<t}$ 。

对于要生成已经提到过的实体，作者使用了 ${\sf Reflexive}$ 关系进行自连接。

<div align="center">
    <img src="https://pic4.zhimg.com/v2-eea6da84eac0e6b798f0d1d075117c27_b.png">  
</div>

<center style="color:#C0C0C0;text-decoration:underline">图2 KGLM语句生成算法模式图</center>

上述算法计算的 $p(x_t,{\mathcal E}_t|x_{<t},{\mathcal E}_{<t})$ 与 $p(x_t|x_{<t})$ 有些许不同，为解决这个问题，作者在后文提到使用了 $p({\rm x})=\sum_{\mathcal E}p({\rm x},{\mathcal E})$ 进行了单词实体的边际概率计算。

### 2.2 分布参数化

作者使用前文提到的LSTM算法对隐状态 ${\bf h}_t$ 进行计算，并将其分解成三部分： ${\bf h}_t=[{\bf h}_{t,x};{\bf h}_{t,p};{\bf h}_{t,r}]$ ，分别用来预测单词、父级实体和关系，实体类型 $t_t$ 则通过对 ${\bf h}_{t,x}$ 进行单层softmax计算得到。

**挑选实体**：本文同样对所有实体和关系进行了预训练嵌入（记做 $\bf v$ ）。

a. 在 $t_t=\sf new$ 时，模型使用 $p(e_t)={\rm softmax}({\bf v}_e\cdot({\bf h}_{t,p}+{\bf h}_{t,r})), \ e\in{\mathcal E}$ 计算概率，其中使用父级实体和关系的隐状态是为了模仿TransE的结构，作者也是用TranE架构进行实体和关系的嵌入。

b. 在 $t_t=\sf related$ 时，模型使用 $p(p_t)={\rm softmax}({\bf v}_p\cdot{\bf h}_{t,p}),\ p\in{\mathcal E}_t$ 计算父级实体概率，使用 $p(r_t)={\rm softmax}({\bf v}_r\cdot{\bf h}_{t,r}),\ r\in\{r|(p_t,r,e)\in{\mathcal KG}_t\}$ 计算关系概率。在得到了 $p_t$ 和 $r_t$ 后，也就确定了 $e_t$ 的取值，如果有多个选择就随机挑选。

**输出实体**：如果 $e_t=\emptyset$ ，这意味着已经没有实体可以继续输出了，模型将在词汇表中再次使用LSTM算法。如果存在实体可以输出，则构建一个原始词汇表和含有所有出现过的实体及其关联词汇表上的分布，这一分布是已知 $e_t$ 和 $x_t$ 下的条件分布。为了计算原始词汇表的得分，作者使用 ${\bf h}_{t,x}'={\bf W}_{\rm proj}[{\bf h}_{t,x};{\bf v}_{e_t}]$ 代替 $h_{t,x}$ ，其中 ${\bf W}_{\rm proj}$ 是将合并向量投影至与 ${\bf h}_{t,x}$ 相同的向量空间的一个可学习权重矩阵。同时，作者也使用了一个LSTM结构进行同义词表概率的计算： $p(x_t=a_j)\propto{\rm exp}[\sigma(({\bf h}_{t,x}')^\top{\bf W}_{\rm copy}){\bf a}_j]$ 。

## 3.*Linked WikiText-2* 数据集

除了模型因素之外，语言模型中事实推断实现的另一难点在于难以获取训练数据。标准的语料库只包含了文字，没有实体和关系的标注，而建立在关系抽取的数据集则在文本和知识图谱间架起了一座桥梁。*Linked WikiText-2*的目标则与*data-to-text*任务相似，作者感兴趣的是如何让语言模型能够动态地从知识图谱中获取事实知识。该数据集本身和Wikidata高度匹配，便于和基于*WikiText-2*训练的模型进行比较，同时Wikipedia的文章也广泛地涵盖了文本表达的事实。而在数据获取上，*WikiText-2*使用的API会丢弃一些有用的信息，而*Linked WikiText-2*则是直接从文章HTML提取信息。

**初始实体标注**：使用人为标注的连接作为初始集合，并将和它们相关的Wikipedia实体关联起来。

**局部知识图谱**：第二步则是通过迭代的方法向局部知识图谱中不断加入出现的新实体，如果一个实体在后文再次出现，则将这一实体作为父级实体。另外，该数据集也允许一个实体拥有多条事实关系。

**扩展实体标注**：对于单位、日期等实体，作者分别对其进行了处理。对于日期，作者列举了所有可能的表示方式进行字符串匹配。而对于单位，作者使用了Python的 $\sf pint$ 库处理不同的形式。

**实例标注**：对实例的标注流程如下表所示：

<div align="center">
    <img src="https://pic1.zhimg.com/v2-d5b17056f599f6e4b41c15cf4d7d2ff0_b.png">  
</div>

<center style="color:#C0C0C0;text-decoration:underline">图3 标注流程表</center>

可以看到，每个标注根据当前图谱内容反映当前实体类型是 $\sf new$ 还是 $\sf related$ ，值得注意的是这种方法也会因为Wikidata的数据缺失产生错误。表中反映对 $\sf Nintendo$ 添加了两组联系（ $\sf R:manu$ 和 $\sf platform$ ），尽管这种方法会有缺失和错误现象，但是它生成的标注更多而且更具细节，有助于KGLM的训练。

## 4.KGLM的训练和推断

**预训练知识图谱嵌入**：为了预测未出现过的字词，本文使用了TransE预训练嵌入模型，在给定 $(p,r,e)$ 时，通过最小化距离 $\delta({\rm v}_p,{\rm v}_r,{\rm v}_e)=||{\rm v}_p+{\rm v}_r-{\rm v}_e||^2$ 学习嵌入向量，最大边际损失函数定义为 ${\mathcal L}=\max(0,\gamma+\delta({\rm v}_p,{\rm v}_r,{\rm v}_e)-\delta({\rm v}_p',{\rm v}_r,{\rm v}_e'))$ ，这里 $\gamma$ 代表边际， $p'$ 和 $e'$ 是随机选择的实体嵌入。

**结合*Linked WikiText-2*进行训练**：KGLM包含许多复杂的步骤，但在*Linked WikiText-2*上的训练是十分直接的，根据数据集上的负对数似然函数得到的损失函数如下：

$$\ell(\Theta)=\sum_t\log p(x_t,{\mathcal E}_t|x_{<t},{\mathcal E}_{<t};\Theta)$$ 

对于有多个父级对象的实体，则首先计算所有父级实体的边际和。

**模型推断**：前文提到，模型的主要目标是计算边际概率 $p({\rm x})=\sum_{\mathcal E}p({\rm x},{\mathcal E})$ 而非联合分布概率，然而由于可能的标注存在的联合分布参数空间过于庞大，直接计算边际概率的方法并不可行。作者在此处使用了重要性采样的方法对边际概率进行了合理的估计：

$$\begin{aligned} p({\rm x})&=\sum_{\mathcal E}p(x,{\mathcal E})=\sum_{\mathcal E}\frac{p({\rm x},{\mathcal E})}{q({\mathcal E}|{\rm x})}q({\mathcal E}|{\rm x})\\ &\approx\frac 1N\sum_{{\mathcal E}\sim q}\frac{p({\rm x},{\mathcal E})}{q({\mathcal E}|{\rm x})} \end{aligned}$$

## 5.实验结果

完成模型构建后，作者选用了一些不同方面表现良好的模型进行对比。在**困惑度**水平上，作者选择了AWD-LSTM，ENTITYNLM，EntityCopyNet三个模型进行横向对比，结果发现KGLM在困惑度、未知惩罚困惑度都远比其他三个模型小。

而在**事实补全**方面，作者选用了*Freebase*中6个常见关系并构建了一系列填空问题，KGLM在出生地、书籍作者等方面表现都很优异，然而回答城市所属国家时正确率很低。另外，作者使用*oracle*和*NEL*两种类型（*oracle*代表给定了 $X$ 的正确实体标注，*NEL*则使用了基于重要性采样的分类判别模型，结合了*NEL*实体连接来产生 $X$ 的实体标注）的KGLM进行对比，二者的表现也不相上下。而与时下热门的GPT-2相比，KGLM生成的单词更加的具体，GPT-2则更倾向于生成更笼统和更常见的单词。

在**改进知识图谱效果**方面，因为事实知识往往和模型的生成能力紧密结合，大多数语言模型都很难控制语言的生成。而通过修改知识图谱，我们得以直接对KGLM进行控制，以此改变模型的预测。

综上所述，KGLM模型结合了知识增强学习的思想，创新性地提出了让语言模型能够不添加额外条件直接利用整体知识图谱的方法，如果单凭记忆，模型无法生成在现实生活中**事实正确**的语句，KGLM则通过引入外部知识的方法生成高质量的事实正确语句。同时，现有的数据到文本（data-to-text）生成模型往往是以BLEU和METEOR分数为指标，在一个小的领域内生成尽可能连贯的文字；而KGLM则旨在利用较大的结构化图谱来在广阔领域的话题下使用更加稀少的单词和事实，重心在于降低模型的困惑度。而KGLM的主要局限则是需要在推断的过程中进行边际化计算和对标注单词实体的依赖，这也为未来的NLP模型提出了新的问题。