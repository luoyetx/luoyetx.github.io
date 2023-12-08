Title: Face Alignment at 3000 FPS via Regressing Local Binary Features
Date: 2015-08-19
Slug: face-alignment-at-3000fps
Category: Machine Learning


Face Alignment at 3000 FPS via Regressing Local Binary Features 这篇论文(下面简称 3000fps)实现了对人脸关键点的高速检测，而且预测的精度也是相当的高。本文首先讲解了 3000fps 整篇论文的思路和方法，然后具体谈谈如何利用 C++ 实现这篇论文中的方法。

### 论文解读

3000fps总体上采用了随机森林和全局线性回归相结合的方法，相对于使用卷积神经的深度学习方法，3000fps采用的算是传统的机器学习方法。CUHK 的 Deep Convolutional Network Cascade for Facial Point Detection 采用了级联卷积神经网络的方法来预测人脸关键点，我针对这篇论文有过相应的实现，采用了 Caffe 框架，并利用论文作者开放出来的数据集进行训练，预测的结果还是相当不错的，相关代码已托管在 github 上，请戳[这里][deeplandmark]，欢迎指正。

我们回到 3000fps 这篇论文，论文中思路与前几年的论文 Face Alignment by Explicit Shape Regression(下面简称 ESR) 还有 Robust face landmark estimation under occlusion(下面简称 RCPR) 有共通之处。这三篇论文的总体思路都可以用下面这个公式来表达

$$S^{t} = S^{t-1} + R^{t}(I, S^{t-1})$$

这个公式包含了很多信息，我们先来认识几个名词。我们把关键点的集合称作形状，形状包含了关键点的位置信息，而这个位置信息一般可以用两种形式表示，第一种是关键点的位置相对于整张图像，第二种是关键点的位置相对于人脸框(标识出人脸在整个图像中的位置)。我们把第一种形状称作绝对形状，它的取值一般介于 0 到 w or h，第二种形状我们称作相对形状，它的取值一般介于 0 到 1。这两种形状可以通过人脸框来做转换。公式中的 $S^t$ 就表示了绝对形状，$R^t$ 表示一个回归器，$I$ 表示图像，$R^t$ 根据图像和形状的位置信息，预测出一个形变，并将它加到当前形状上组成一个新的形状。$t$ 表示级联层数，一般我们会通过多层级联来预测形状。

#### 回归器 $R^t$

ESR 和 RCPR 采用了随机厥作 Regression，随机厥在叶子节点中存储了对应的形变，在预测过程中，当样本落入某个叶子节点时，就将其上存储的形变作为预测的输出，我们在这里不具体展开随机厥的相关内容。而在 3000fps 中使用了较为复杂的实现。首先，3000fps 并没用采用随机厥作为预测的单元，而是采用了随机树，并且用随机森林来做预测。其次 3000fps 并没有直接采用随机树叶子节点存储的形变量作为预测输出，而是将随机森林的输出组成一种特征(称作 LBF)，利用这个 LBF 来做预测。除了采用随机森林的结构来做预测，3000fps 还针对每个关键点给出一个随机森林来做预测，并将所有关键点对应的随机森林输出的局部特征相互连接起来，称作局部二值特征(LBF)，然后利用这个局部二值特征来做全局回归，用来预测形变。

![3000fps-train-test]({filename}/images/2015/3000fps-train-test.png)

上图描述了回归器 $R^t$ 的训练和预测过程。其中 $\Phi^{t}_{l}$ 表示第 $t$ 级中第 $l$ 个关键点所对应的随机森林，所有关键点的随机森林一起组成了 $\Phi^t$，它的输出为 LBF 特征。然后利用 LBF 特征来训练全局线性回归或者预测形变。

![3000fps-one-landmark]({filename}/images/2015/3000fps-one-landmark.png)

上图描述了 $R^t$ 生成 LBF 特征的过程，图的下半部分描述了单个关键点上随机森林输出了一个局部二值特征，然后把所有随机森林的输出前后连接起来组成一个非常大但又非常稀疏的 LBF 特征。这个特征只有 01 组成，且大部分是 0，特征非常稀疏。

#### Shape-indexed 特征

每个关键点都会对应一个随机森林，而每个随机森林是由多个相互独立的随机树组成。论文中的随机树采用的特征被称作 Shape-indexed 特征，ESR 和 RCPR 中也是用到了相同的特征，这个特征主要描述为人脸区域中两个点的像素差值。关于两个像素点的选取，三个方法使用到了不同的策略。

![3000fps-esr-feature]({filename}/images/2015/3000fps-esr-feature.png)

ESR 方法采用在两个关键点附近随机出两个点，做这两个点之间的差值作为 Shape-indexed 特征.

![3000fps-rcpr-feature]({filename}/images/2015/3000fps-rcpr-feature.png)

RCPR 方法采用选取两个关键点的中点外加一个随机偏置来生成特征点，用两个这样的特征点的差值作为 Shape-indexed 特征。

在 3000fps 中，由于随机森林是针对单个关键点的，所有随机树中使用到的特征点不会关联到其他关键点上，只在当前关键点的附近区域随机产生两个特征点，做像素差值来作为 Shape-index 特征。

![3000fps-feature]({filename}/images/2015/3000fps-feature.png)

3000fps 中随着级联的深入(即 $t$ 越来越大)，随机点的范围也会逐渐变小，以期获得更加准确的局部特征。

#### 随机树的训练

上一节已经确定了随机树训练时用到的 Shape-indexed 特征。在训练随机树时，我们的输入是 $X=\{I, S\}$ 而预测目标是 $Y=\Delta S$。实际在训练随机树时，树中的每个节点的训练过程都是一样的。我们在训练某个节点时，从事先随机生成好的 Shape-indexed 特征集合 $F$ 中选取一个(当然，你也可以临时随机生成一个特征集合，或整棵随机树使用一个特征集合或整个随机森林使用一个特征集合，我们这里假设这棵随机树使用一个特征集合)，选取的特征能够将所有样本点 $x$ 映射成一个实数集合，我们再随机一个阈值将样本点分配到左右子树中，而目的是希望左右子树中的样本点的 $y$ 具有相同的模式。特征选取可以用如下公式描述

$$f = \underset{f \in F}{\operatorname{argmax}}\Delta$$

$$\\Delta = S(y | y \in Root) - [S(y | y \in Left) + S(y | y \in Right)]$$

$$ y \in
\begin{cases}
Left, & f(x) < \delta \\
Right, & f(x) >= \delta \\
\end{cases}$$

上述公式中 $F$ 表示特征函数集合，$f$ 表示选取到的特征函数(即利用随机到的特征点计算 Shape-index 特征)，$\delta$ 表示随机生成的阈值，$S$ 用来刻画样本点之间的相似度或者样本集合的熵(论文中采用了方差)。针对每个节点，训练数据 $(X, Y)$ 将会被分成两部分 $(X_1, Y_1)$ 和 $(X_2, Y_2)$，我们期望左右子树中的样本数据具有相同的模式($Y$ 的分布尽量固定下来，熵越小？)，这个论文中用了方差来刻画，所以选择特征函数 $f$ 时，我们希望方差减小最大。

![randomtree]({filename}/images/2015/randomtree.png)

随机树的每个节点都采用这种方法训练，而每棵随机树都是相互独立训练的，训练过程都是一样的，这样单个关键点的随机森林就训练完毕了。

#### 全局线性回归训练

按照常理，我们可以在随机树的叶子节点上存储预测的形变，测试时可以将随机森林中每棵随机树的预测输出做平均或者加权平均，然而 3000fps 并没有这样做，它将随机树的输出表示成一个二值特征(详情见上面的图)，将所有随机树的二值特征前后连接起来组成一个二值特征，即 LBF 特征。论文中，利用这个特征做了一次线性回归，将形变作为预测目标，训练一个线性回归器来做预测。

$$W_t = \underset{W_t}{\operatorname{argmin}} \|{\Delta S - W_t \cdot lbf}\|_2 + \lambda \|W_t\|_2$$

线性回归可以用如上公式表示，$\Delta S$ 形变目标，$lbf$ 表示特征，$W_t$ 是线性回归的参数，$\lambda$ 用来抑制模型，防止出现过拟合。预测时采用下面的公式

$$\Delta S = W^t \cdot lbf$$

在 3000fps 论文中，多级级联回归的方法的每一级都可以按如上所讲的拆分两个部分，利用随机森林提取局部二值特征，再利用局部二值特征做全局线性回归预测形状增量 $\Delta S$。

#### 关于 $S^0$

在之前的讨论中，我们并没有说明 $\Delta S$ 具体是绝对形状增量还是相对形状增量。在实际情况中，我们需要 $\Delta S$ 为相对形状增量，因为绝对形状的位置是相对于整个图像的，我们无法对数据的绝对形变的分布做约束(绝对形变虽然可以抹去位置的绝对信息，但人脸框的尺度无法约束)。在提取局部二值特征时，我们需要绝对形状下的图像像素信息，而在预测得到的则是相对形状增量，而这两者可以通过人脸框做相互之间的转换。

所有形变 $\Delta S$ 均是相对于当前形状而言，通过级联的方式叠加在一起，而初始形状 $S_0$ 与模型预测本身无关，但是这个 $S_0$ 模型预测过程中起来关键性作用。我们假设预测样本理论上的真实形状为 $S_g$，那么 $S_0$ 和 $S_g$ 两者之间的差异的大小将直接影响到预测结果的准确性。3000fps 中采用了训练样本的平均形状作为初始形状，而 RCPR 则选择从训练样本中随机选择初始形状。

一般来说，$S_0$ 是相对形状通过人脸框转变为绝对形状对应到当前的人脸中，那么人脸框的尺度对 $S_0$ 与 $S_g$ 之间的差异也起到了决定性的作用，所以我们一般都需要用相同的人脸检测方法来标记训练图像和预测图像上的人脸框，保证两者的人脸框尺度，从而提高 $S_0$ 的准确性。但是我们不得不承认算法本身仍旧受到 $S_0$ 很大的影响。包括 RCPR 和 ESR 方法也同样受制于 $S_0$。相比较而言，深度学习方法则没有太大的影响，一般可以先通过网络来预测得到 $S_0$，这时的 $S_0$ 和 $S_g$ 之间的误差能够做到非常小，进而再在 $S_0$ 的基础上做细微的修正，提高精度，深度级联卷积网络预测关键点就是采用了这个思路。虽然深度学习的方法能够摆脱 $S_0$ 的限制，但它仍然受制于人脸框的尺度，而且尺度对其模型的预测影响比其他传统方法也好不到哪里去。

### 论文实现

论文作者并没有开放源代码出来，但是已经有同学用 Matlab 实现了论文，并且开源到了 github 平台，项目地址在[这里][matlab-3000fps]。同时也有同学参照 Matlab 代码利用 C++ 重新实现了一遍，项目地址在[这里][c++-3000fps]。我参考了这两份代码自己采用 C++ 重新实现了论文中的方法。

#### 数据采集与预处理

搞机器学习的永远少不了跟数据打交道，还好在人脸关键点检测这方面的开放数据还是蛮多的，参考论文中使用到的数据集，我们可以从[这里][dataset]下载到人脸 68 个点的数据集，可供我们做训练和测试用。有了现成的数据就省去了我们自己搜刮数据这一步，而且数据集中的数据格式还是非常规范的。

数据的预处理一般都是对我们得到的数据的再加工，在送给模型做训练之前我们还需要打磨打磨我们的数据。这里我们需要对人脸框做重新定位，上面已经讨论过人脸框的尺度对 $S_0$ 的影响，我们应该在训练和检测的过程中采用相同的人脸检测器来标识人脸框，现在开源出来的用得最多的还是 OpenCV 自带的 VJ 检测器(虽然是个废=。=)。利用 VJ 检测器可以锁定关键点标识的人脸的位置，给出人脸框。

我们的训练数据大致就是人脸图像，人脸框，人脸真实形状这三种数据，$X = \{ I, BBox \}$, $Y = \{ Shape \}$。之前和一些同学交流实现时，有些同学出现的爆内存的情况，大致上是将训练图像全部载入到内存后做了一些处理。导致内存飙升致程序崩溃。其实考虑图像 $I$，我们并不需要整幅图像作为训练数据，一般只要将人脸附近一块图像区域截取出来作为 $I$ 即可，这本身并不影响我们的训练数据，只要相应的重新计算关键点位置即可，因为数据集中的图片大多都是高清无码大图，人脸可能只占图像的一小块，这种方法能够减少非常多的内存消耗。同时我们应该批量处理数据，一口气将所有数据载入内存而后在处理的方式会把程序的最大内存消耗开到最大，同样有可能出现崩溃。当然，如果你的程序是 64 位的，机子内存又是杠杠的，那就完全不用理会上面的这些优化操作了。

#### Data Augmentation

Data Augmentation 在现有数据集不够充足时，通过变换已有的数据来增加数据集的大小，可谓是一种经济又环保的方法。根据不同的数据和模型，我们可以采用不同的变换手段，如果目标具有对称不变性，那么水平翻转图像将是一种不错的手段，常见的还有旋转图像来增加数据集。针对人脸关键点定位，显然我们左右翻转人脸并不会影响人脸的结构(左右眼交错一下也没什么影响)，包括轻微旋转人脸也同样能够增加训练数据集。

在训练过程中，我们需要给每个训练样本一个初始形状，这个初始形状可以从样本中随机选取，通过选取多个初始形状，我们同样可以达到增加数据集的效果。

#### 随机森林的并行训练

随机森林的构造和训练并没有什么特别之处，这里主要谈谈如何加速训练过程。我们知道随机森林中的每棵随机树都是相互独立训练的，而每个关键点对应的随机森林也是相互独立的，这样，我们就可以并行训练随机树。考虑到并行计算的编码问题，我们并不需要通过多线程并发的方式来实现并行计算，这里可以使用到 [OpenMP][openmp] 来实现并行计算，OpenMP 会将我们的代码通过它自定义的语法将其并行化，底层用的是系统级线程的实现方式，现在的编译器已经都内置了对 OpenMP 的支持，所以代码移植也很方便。OpenMP 的性能可能没有直接使用系统极线程的方式来得高，但是它简单易学，使用非常方便，可能在线程切换方面比直接使用系统线程库来的高效，但是对于编码而言却是非常简单，几条语句就可以将原本串行的代码变成并行，修改代码的代价非常低。

#### 全局线性回归

训练完随机森林后，我们就能够得到训练数据的 LBF 特征了，根据这个 LBF 特征在加上相对形状增量这个预测目标，我们便可以训练全局线性回归模型了。由于大量的训练数据，再加上 LBF 特征的高度稀疏，论文中提到了利用双坐标下降的方法来训练高度稀疏的线性模型，本人对这个并不是十分了解，还好发明这个方法的人专门写了一套求解线性模型的库，并开源在 github 上，项目名称是 [liblinear][liblinear]。这个库本身采用 C++ 编写，也提供了很对语言的绑定，这里我们可以直接采用它的 C++ 代码，根据相关文档准备好相应的数据，直接调用就可求解到模型参数，使用起来还是很方便的。

#### 结果

3000fps 中的算法是一个级联的模型，每一级是随机森林加上全局回归，通过多次级联来求得相对形状增量，从而计算得到最终的形状。下面是我自己训练好的模型的预测结果。

![3000fps-result1]({filename}/images/2015/3000fps-result1.png)
![3000fps-result2]({filename}/images/2015/3000fps-result2.png)

### 总结

3000fps 这篇论文所用的方法除了有比较好的精度，关键在于其方法的预测速度相当的快。论文中采用的快速模型能够达到 3000fps 的速度预测 68 个点，速度非常恐怖。本人实现的结果是 300fps，CPU 单核 3.2GHz，内存 8G，论文中并没有提到其使用到的机器性能如何，只提到了其实现方法只使用了单核，我认为论文中的实现应该是在底层做过相应的优化才能达到如此高的速度，当然我们也没有必要可以追求速度，能够达到实时的性能就可以了(不过，谁都会认为越快越好)。

### 参考资料

- [Face Alignment at 3000 FPS via Regressing Local Binary Features][3000fps]
- [Deep Convolutional Network Cascade for Facial Point Detection][cuhk-landmark]
- [Face Alignment by Explicit Shape Regression][esr]
- [Robust face landmark estimation under occlusion][rcpr]
- [deeplandmark][deeplandmark]
- [Matlab 实现的 3000fps][matlab-3000fps]
- [C++ 实现的 3000fps][c++-3000fps]


[deeplandmark]: https://github.com/luoyetx/deep-landmark
[cuhk-landmark]: http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
[3000fps]: http://research.microsoft.com/en-us/people/yichenw/cvpr14_facealignment.pdf
[esr]: http://research.microsoft.com/pubs/192097/cvpr12_facealignment.pdf
[rcpr]: http://www.vision.caltech.edu/xpburgos/ICCV13/
[matlab-3000fps]: https://github.com/jwyang/face-alignment
[c++-3000fps]: https://github.com/yulequan/face-alignment-in-3000fps
[dataset]: http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
[openmp]: http://openmp.org/wp/
[liblinear]: https://github.com/cjlin1/liblinear