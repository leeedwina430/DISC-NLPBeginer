问题简述：



Word2vec是一个学习词向量的简单神经网络架构。

我们用向量来表示单词，word2vec用向量的内积体现单词之间的相似性。同时用softmax进行归一化，表示给定center word时outside word的条件概率分布。
所以相较于其他单词表示的好处是可以将离散的单词投射到连续而且低维的向量空间。
同时这种表示方式能够捕捉到单词之间的相似性。

Word2Vec有两种不同的模型，CBOW和skip-gram。CBOW是给定上下文也就是context words，预测中心词center word；skip-gram是给定中心词，预测上下文单词。（我们这里用skip-gram模型）

这个就是skip gram的模型架构，从整体上来看，我们最终想得到就是这个隐藏层中的权重，这就是我们的词向量。



skip-gram的目标函数就是对于每一个单词，当固定它作为中心词并且给定上下文单词后的负log likelihood的总和。
需要强调我们对于每一个单词都会有两个向量，一个是center word，还有一个context word。
从偏导式子中可以看出，我们用outside word 和其他的word来更新center word，对于context word vectors，不管是outside 还是other，都只用到了center vector

首先由于我们的词向量也就是hidden layer的权重依赖于vocabulary的大小和我们选取的向量维数大小相关。20000*300=600，0000参数需要学习，是非常大的计算量，同时减慢训练速度，需要大量的数据来训练。
在原论文发布后的第二篇论文中，作者提到了两个训练的优化方法。

。。。

负采样的想法是，每个训练样本只修改一小部分的权重，而不是全部的权重；对每一个真实的center-outside pair训练一个二分类逻辑回归，center 和 other 相对比；
这个偏导的式子和之前的很相似，向量更新的方式和之前类似，

。。。



在梳理算法流程的过程中，发现我们的code里一些比较有趣的地方。
没有什么预处理，只进行了大小写转换，删掉了一些数据集中奇怪的单词
用了minibatch
在采样center context pair，window resample，例子。。



实验结果：

尝试了不同的context window 和 vector dimension的组合。可以看到正则化系数增加的时候，acc会相应降低。


Regularization Coefficient
平衡模型对训练数据的拟合和泛化能力。
过大的正则化系数会降低模型的拟合，但是可以增强模型的泛化能力。



如果只有一个向量：同一个单词在自身上下文中出现概率极低，objective会减小这个概率；
然而对于内积定义，自身向量相乘应该是内积最大值=>矛盾





同时，也很好的体现了我们对于正则化项的假设：越大的正则化项会降低一定模型的精度，但同时会减少训练集和验证集的准确度之差