Title: Caffe 源码阅读 伊始
Date: 2015-10-21
Slug: reading-caffe-1
Category: Machine Learning


[Caffe][caffe] 是一个深度学习的框架，以 C++ 编写，性能卓越，并且现在已经支持单机多 GPU 运算。这篇博文包括之后的文章记录了我自己阅读学习 Caffe 源码的过程，也借此鼓励自己坚持下去，好好向 Caffe 的作者学习。

深度学习在这几年火得不行，尤其是 CNN 已经成为了解决视觉方面难题的神兵利器，而在 CNN 框架的开源实现方面，Caffe 以其使用简单，性能卓越，CPU/GPU 无缝切换等优点，受到了众多开发人员和研究人员的关注。Caffe 源码托管在 [Github][github] 上，任何人都能够免费获取并使用它。

### Caffe 概况

Caffe 中网络模型的描述及其求解都是通过 [protobuf][pb] 定义的，并不需要通过敲代码来实现。同时，模型的参数也是通过 [protobuf][pb] 实现加载和存储，包括 CPU 与 GPU 之间的无缝切换，都是通过配置来实现的，不需要通过硬编码的方式实现。原则上讲，如果只是使用 Caffe 来训练卷积神经网络的话，我们完全不需要接触或者了解 Caffe 的源码，只需要关注如何定义网络模型和求解参数的设置，并且准备好相应格式的训练数据就完了。Caffe 本身采用 C++ 编写，速度非常快，加上对 GPU 的支持，在各大 CNN 的实现中，速度还是处于领先地位的。同时 Caffe 本身也是支持纯 CPU 下的计算的，当我们在 GPU 下训练完网络，也可以很简单地切换到 CPU 下运行计算网络，只需简单修改一下 Caffe 的配置。Caffe 同时也有一个庞大的社区来支持和维护 Caffe 的代码，添加新的功能，修正 bug 等，社区非常活跃，[Google Groups][gg] 上的讨论也非常多。

### Caffe 整体结构

Caffe 代码本身非常模块化，主要由 4 部分组成 `Blob` `Layer` `Net` 和 `Solver`。

- `Blob` 主要用来表示网络中的数据，包括训练数据，网络各层自身的参数，网络之间传递的数据都是通过 Blob 来实现的，同时 Blob 数据也支持在 CPU 与 GPU 上存储，能够在两者之间做同步。
- `Layer` 是对神经网络中各种层的一个抽象，包括我们熟知的卷积层和下采样层，还有全连接层和各种激活函数层等等。同时每种 Layer 都实现了前向传播和反向传播，并通过 Blob 来传递数据。
- `Net` 是对整个网络的表示，由各种 Layer 前后连接组合而成，也是我们所构建的网络模型。
- `Solver` 定义了针对 `Net` 网络模型的求解方法，记录网络的训练过程，保存网络模型参数，中断并恢复网络的训练过程。自定义 Solver 能够实现不同的网络求解方式。

阅读 Caffe 代码可以通过由小到大，至上而下的方式来阅读学习，首先学习 Blob 的实现，然后查看 Layer 的定义并阅读各种类型的 Layer 的实现方式，最后阅读 Net 的代码来学习整个网络结构。而 Solver 的代码可以单独列出来学习，了解网络的求解优化过程。我也会以这种方式阅读 Caffe 源码并记录自己的阅读心得并做些总结，希望自己能够坚持下去。**Fighting!!!**

### 理论知识积累

如果没有理论知识的基础，我认为学习 Caffe 源码的意义不是很大，所以强烈建议大家事先学习一下神经网络相关的基础知识，并且简单地使用一下 Caffe 之后再阅读其相应的源码，这样收获会更多，意义也更大。如果想用 Caffe 练练手，可以参考我的另外一篇博文 [Caffe 小试牛刀](/2015/04/little-caffe/)，利用 Caffe 做 [Kaggle][kaggle] 上的手写体识别。

### 参考资料

- [Caffe][caffe]

[caffe]: http://caffe.berkeleyvision.org/
[github]: https://github.com/
[pb]: https://developers.google.com/protocol-buffers/
[gg]: https://groups.google.com/forum/#!forum/caffe-users
[kaggle]: https://www.kaggle.com/
