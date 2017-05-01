Title: Caffe 小试牛刀
Date: 2015-04-11
Slug: little-caffe
Category: Machine Learning


在 [Deep Learning](http://en.wikipedia.org/wiki/Deep_learning) 如此火的今天，[Caffe](http://caffe.berkeleyvision.org/) 的出现使得我们接触深度学习的门槛变得异常之低。在自己的笔记本或者远端服务器上部署这个 Caffe 框架也是异常简单，照着官方给出的安装文档可以在大部分 Linux 发行版和 Mac OS 上安装好这个深度学习框架。官方暂时没有给出 Windows 版本的 Caffe，不过社区已经有人移植到了 Windows 上，项目地址在 [github](https://github.com/niuzhiheng/caffe) 上。我们尽量还是在 *nix 平台上部署 Caffe 框架。

### Caffe 简介

Caffe 是一个清晰而高效的深度 [CNN](http://en.wikipedia.org/wiki/Convolutional_neural_network) 学习框架，其作者是博士毕业于 UC Berkeley 的贾扬清，目前在Google工作。Caffe 支持命令行，并提供了 Python 和 Matlab 接口方便开发者和研究人员调用，而且其框架本身可以在 CPU/GPU 之间无缝切换，非常方便。Caffe 的详细文档在[这里](http://caffe.berkeleyvision.org/installation.html)。根据官方文档安装完各种依赖库后就可以编译 Caffe 框架了，有些依赖库可能软件源中的版本过低或者根本就没有，可以自己源码编译安装。

### Kaggle 上的数字识别

[Kaggle](https://www.kaggle.com/) 是一个数据竞赛平台，提供各种需求和数据给全世界的参赛者。其中有一个比赛项目是 [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)，就是著名的手写体数字识别。[Yann LeCun](http://en.wikipedia.org/wiki/Yann_LeCun) 提出的 [LeNet](http://yann.lecun.com/exdb/mnist/) 已经能够很好地解决这个问题了，Caffe 官方的 Example 中就有 LeNet 的实现。我们就利用这个 CNN 网络模型再加上 Kaggle 提供的数据来走一边深度学习的流程，从数据的获得与清理，CNN 模型的训练，再到最后的数据预测。

### 利用 Caffe 解决 Kaggle 上的数字识别

Kaggle 给这个项目提供了两个数据，分别为训练数据和测试数据。但是我们拿到的数据并不是图片，而是 csv 格式的数据，至于数据的具体内容，Kaggle 官方有详细的说明，可以参考[这里](https://www.kaggle.com/c/digit-recognizer/data)。

#### csv 数据预处理

Caffe 在训练时可以采用各种格式的输入数据(不同的 Data 层)，详细的格式参见官方[文档](http://caffe.berkeleyvision.org/tutorial/layers.html#data-layers)。这里我采用了 HDF5 格式的输入数据，下面的代码将 csv 格式的数据转换成了 HDF5 格式的数据。代码采用了 Python 编写，利用 [Pandas](http://pandas.pydata.org/) 来读取 cvs 数据，并用 [h5py](http://www.h5py.org/) 来写 HDF5 格式的数据。

```python
#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
import h5py


DATA_ROOT = 'data'
join = os.path.join
TRAIN = join(DATA_ROOT, 'train.csv')
train_file = join(DATA_ROOT, 'mnist_train.h5')
test_file = join(DATA_ROOT, 'mnist_test.h5')

# logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

# load data from train.csv
logger.info('Load data from %s', TRAIN)
df = pd.read_csv(TRAIN)
data = df.values

logger.info('Get %d Rows in dataset', len(data))

# random shuffle
np.random.shuffle(data)

# all dataset
labels = data[:, 0]
images = data[:, 1:]

# process data
images = images.reshape((len(images), 1, 28, 28))
images = images / 255.

# train dataset number
trainset = len(labels) * 3 / 4

# train dataset
labels_train = labels[:trainset]
images_train = images[:trainset]
# test dataset
labels_test = labels[trainset:]
images_test = images[trainset:]

# write to hdf5
if os.path.exists(train_file):
    os.remove(train_file)
if os.path.exists(test_file):
    os.remove(test_file)

logger.info('Write train dataset to %s', train_file)
with h5py.File(train_file, 'w') as f:
    f['label'] = labels_train.astype(np.float32)
    f['data'] = images_train.astype(np.float32)

logger.info('Write test dataset to %s', test_file)
with h5py.File(test_file, 'w') as f:
    f['label'] = labels_test.astype(np.float32)
    f['data'] = images_test.astype(np.float32)

logger.info('Done')
```

在这里，我把数据分割成了两部分，分别作为训练数据和测试数据(与 Kaggle 提供的 test.csv 数据不同，这里的测试数据是带有 label 的)，方便测试模型的准确性。

#### CNN 网络的训练

这里直接用 Caffe 自带的 Example 中的模型。网络的定义可以在 Caffe 源码目录中找到，这里我就不全贴了，只贴一下输入的 DataLayer。

```
layer {
  name: "mnist"
  type: "HDF5Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  hdf5_data_param {
    source: "data/mnist_train.txt"
    batch_size: 64
  }
}
layer {
  name: "mnist"
  type: "HDF5Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  hdf5_data_param {
    source: "data/mnist_test.txt"
    batch_size: 100
  }
}
```

其中的 data/mnist_train.txt 和 data/mnist_test.txt 记录数据文件的路径。下图是我用 [Graphviz](http://www.graphviz.org/) 画的 LeNet 网络图。

{% image fancybox center /assert/img/2015/04/lenet.jpg %}

有了数据和网络定义，我们还需要训练网络时的参数配置，这些配置数据写在 lenet_solver.prototxt 中。具体含义可以参考[文档](http://caffe.berkeleyvision.org/tutorial/solver.html)。

```
# The train/test net protocol buffer definition
# 网络的定义文件路径
net: "model/lenet_train_test.prototxt"
# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
# 每次测试时迭代 100 次
test_iter: 100
# Carry out testing every 500 training iterations.
# 训练网络模型时，每迭代 500 次作一次测试
test_interval: 500
# The base learning rate, momentum and the weight decay of the network.
# 初始学习率，权值衰减
base_lr: 0.01
momentum: 0.9
weight_decay: 0.0005
# The learning rate policy
# 网络学习参数的衰减方式及其参数
lr_policy: "inv"
gamma: 0.0001
power: 0.75
# Display every 100 iterations
# 每迭代 100 次显示网络的输出(loss 等数据)
display: 100
# The maximum number of iterations
# 训练迭代次数
max_iter: 10000
# snapshot intermediate results
# 每隔 5000 次迭代就把网络参数和网络的训练状态保存到文件系统
snapshot: 5000
snapshot_prefix: "model/"
# solver mode: CPU or GPU
# 采用 CPU
solver_mode: CPU
```

`caffe train --solver=model/lenet_solver.prototxt` 这条命令开始训练网络。下面是训练时的部分输出。

```
I0411 21:21:04.364305 31883 solver.cpp:266] Iteration 0, Testing net (#0)
I0411 21:21:07.918603 31883 solver.cpp:315]     Test net output #0: accuracy = 0.0966
I0411 21:21:07.918660 31883 solver.cpp:315]     Test net output #1: loss = 2.3217 (* 1 = 2.3217 loss)
I0411 21:21:07.979472 31883 solver.cpp:189] Iteration 0, loss = 2.39515
I0411 21:21:07.979532 31883 solver.cpp:204]     Train net output #0: loss = 2.39515 (* 1 = 2.39515 loss)
I0411 21:21:07.979560 31883 solver.cpp:464] Iteration 0, lr = 0.01
I0411 21:21:13.373544 31883 solver.cpp:189] Iteration 100, loss = 0.309522
I0411 21:21:13.373603 31883 solver.cpp:204]     Train net output #0: loss = 0.309522 (* 1 = 0.309522 loss)
I0411 21:21:13.373621 31883 solver.cpp:464] Iteration 100, lr = 0.00992565
I0411 21:21:18.770283 31883 solver.cpp:189] Iteration 200, loss = 0.342084
I0411 21:21:18.770339 31883 solver.cpp:204]     Train net output #0: loss = 0.342084 (* 1 = 0.342084 loss)
I0411 21:21:18.770357 31883 solver.cpp:464] Iteration 200, lr = 0.00985258
I0411 21:21:24.270835 31883 solver.cpp:189] Iteration 300, loss = 0.178883
I0411 21:21:24.270900 31883 solver.cpp:204]     Train net output #0: loss = 0.178883 (* 1 = 0.178883 loss)
I0411 21:21:24.270920 31883 solver.cpp:464] Iteration 300, lr = 0.00978075
I0411 21:21:29.655320 31883 solver.cpp:189] Iteration 400, loss = 0.0702766
I0411 21:21:29.655375 31883 solver.cpp:204]     Train net output #0: loss = 0.0702766 (* 1 = 0.0702766 loss)
I0411 21:21:29.655393 31883 solver.cpp:464] Iteration 400, lr = 0.00971013
I0411 21:21:35.007863 31883 solver.cpp:266] Iteration 500, Testing net (#0)
I0411 21:21:38.496989 31883 solver.cpp:315]     Test net output #0: accuracy = 0.9698
I0411 21:21:38.497042 31883 solver.cpp:315]     Test net output #1: loss = 0.0967276 (* 1 = 0.0967276 loss)
I0411 21:21:38.554558 31883 solver.cpp:189] Iteration 500, loss = 0.186758
I0411 21:21:38.554613 31883 solver.cpp:204]     Train net output #0: loss = 0.186758 (* 1 = 0.186758 loss)
I0411 21:21:38.554631 31883 solver.cpp:464] Iteration 500, lr = 0.00964069
I0411 21:21:43.980552 31883 solver.cpp:189] Iteration 600, loss = 0.112056
I0411 21:21:43.980610 31883 solver.cpp:204]     Train net output #0: loss = 0.112056 (* 1 = 0.112056 loss)
I0411 21:21:43.980628 31883 solver.cpp:464] Iteration 600, lr = 0.0095724
I0411 21:21:49.568586 31883 solver.cpp:189] Iteration 700, loss = 0.074904
I0411 21:21:49.568653 31883 solver.cpp:204]     Train net output #0: loss = 0.074904 (* 1 = 0.074904 loss)
I0411 21:21:49.568675 31883 solver.cpp:464] Iteration 700, lr = 0.00950522
I0411 21:21:54.960841 31883 solver.cpp:189] Iteration 800, loss = 0.220085
I0411 21:21:54.960911 31883 solver.cpp:204]     Train net output #0: loss = 0.220085 (* 1 = 0.220085 loss)
I0411 21:21:54.960932 31883 solver.cpp:464] Iteration 800, lr = 0.00943913
I0411 21:22:00.352416 31883 solver.cpp:189] Iteration 900, loss = 0.0172225
I0411 21:22:00.352488 31883 solver.cpp:204]     Train net output #0: loss = 0.0172226 (* 1 = 0.0172226 loss)
I0411 21:22:00.352511 31883 solver.cpp:464] Iteration 900, lr = 0.00937411
I0411 21:22:05.703879 31883 solver.cpp:266] Iteration 1000, Testing net (#0)
I0411 21:22:09.199872 31883 solver.cpp:315]     Test net output #0: accuracy = 0.9801
I0411 21:22:09.199944 31883 solver.cpp:315]     Test net output #1: loss = 0.0650562 (* 1 = 0.0650562 loss)
I0411 21:22:09.256795 31883 solver.cpp:189] Iteration 1000, loss = 0.118511
I0411 21:22:09.256847 31883 solver.cpp:204]     Train net output #0: loss = 0.118511 (* 1 = 0.118511 loss)
I0411 21:22:09.256867 31883 solver.cpp:464] Iteration 1000, lr = 0.00931012
...
...
...
I0411 21:30:26.140774 31883 solver.cpp:266] Iteration 9000, Testing net (#0)
I0411 21:30:29.663858 31883 solver.cpp:315]     Test net output #0: accuracy = 0.9898
I0411 21:30:29.663919 31883 solver.cpp:315]     Test net output #1: loss = 0.0369673 (* 1 = 0.0369673 loss)
I0411 21:30:29.715962 31883 solver.cpp:189] Iteration 9000, loss = 0.00257692
I0411 21:30:29.716016 31883 solver.cpp:204]     Train net output #0: loss = 0.00257717 (* 1 = 0.00257717 loss)
I0411 21:30:29.716032 31883 solver.cpp:464] Iteration 9000, lr = 0.00617924
I0411 21:30:35.261111 31883 solver.cpp:189] Iteration 9100, loss = 0.000706766
I0411 21:30:35.261175 31883 solver.cpp:204]     Train net output #0: loss = 0.000707015 (* 1 = 0.000707015 loss)
I0411 21:30:35.261193 31883 solver.cpp:464] Iteration 9100, lr = 0.00615496
I0411 21:30:40.733172 31883 solver.cpp:189] Iteration 9200, loss = 0.00721649
I0411 21:30:40.733232 31883 solver.cpp:204]     Train net output #0: loss = 0.00721672 (* 1 = 0.00721672 loss)
I0411 21:30:40.733252 31883 solver.cpp:464] Iteration 9200, lr = 0.0061309
I0411 21:30:46.430910 31883 solver.cpp:189] Iteration 9300, loss = 0.0106291
I0411 21:30:46.430974 31883 solver.cpp:204]     Train net output #0: loss = 0.0106294 (* 1 = 0.0106294 loss)
I0411 21:30:46.430991 31883 solver.cpp:464] Iteration 9300, lr = 0.00610706
I0411 21:30:52.084485 31883 solver.cpp:189] Iteration 9400, loss = 0.0217876
I0411 21:30:52.084548 31883 solver.cpp:204]     Train net output #0: loss = 0.0217879 (* 1 = 0.0217879 loss)
I0411 21:30:52.084563 31883 solver.cpp:464] Iteration 9400, lr = 0.00608343
I0411 21:30:57.599124 31883 solver.cpp:266] Iteration 9500, Testing net (#0)
I0411 21:31:01.165457 31883 solver.cpp:315]     Test net output #0: accuracy = 0.9908
I0411 21:31:01.165515 31883 solver.cpp:315]     Test net output #1: loss = 0.0361107 (* 1 = 0.0361107 loss)
I0411 21:31:01.221964 31883 solver.cpp:189] Iteration 9500, loss = 0.00431475
I0411 21:31:01.222023 31883 solver.cpp:204]     Train net output #0: loss = 0.00431501 (* 1 = 0.00431501 loss)
I0411 21:31:01.222040 31883 solver.cpp:464] Iteration 9500, lr = 0.00606002
I0411 21:31:06.748987 31883 solver.cpp:189] Iteration 9600, loss = 0.00301128
I0411 21:31:06.749049 31883 solver.cpp:204]     Train net output #0: loss = 0.00301154 (* 1 = 0.00301154 loss)
I0411 21:31:06.749068 31883 solver.cpp:464] Iteration 9600, lr = 0.00603682
I0411 21:31:12.305821 31883 solver.cpp:189] Iteration 9700, loss = 0.0178924
I0411 21:31:12.305883 31883 solver.cpp:204]     Train net output #0: loss = 0.0178927 (* 1 = 0.0178927 loss)
I0411 21:31:12.305903 31883 solver.cpp:464] Iteration 9700, lr = 0.00601382
I0411 21:31:18.102248 31883 solver.cpp:189] Iteration 9800, loss = 0.0116095
I0411 21:31:18.102319 31883 solver.cpp:204]     Train net output #0: loss = 0.0116097 (* 1 = 0.0116097 loss)
I0411 21:31:18.102339 31883 solver.cpp:464] Iteration 9800, lr = 0.00599102
I0411 21:31:24.297734 31883 solver.cpp:189] Iteration 9900, loss = 0.0111304
I0411 21:31:24.297801 31883 solver.cpp:204]     Train net output #0: loss = 0.0111307 (* 1 = 0.0111307 loss)
I0411 21:31:24.297826 31883 solver.cpp:464] Iteration 9900, lr = 0.00596843
I0411 21:31:29.688841 31883 solver.cpp:334] Snapshotting to model/_iter_10000.caffemodel
I0411 21:31:29.713232 31883 solver.cpp:342] Snapshotting solver state to model/_iter_10000.solverstate
I0411 21:31:29.741745 31883 solver.cpp:248] Iteration 10000, loss = 0.0402425
I0411 21:31:29.741792 31883 solver.cpp:266] Iteration 10000, Testing net (#0)
I0411 21:31:33.262156 31883 solver.cpp:315]     Test net output #0: accuracy = 0.9902
I0411 21:31:33.262218 31883 solver.cpp:315]     Test net output #1: loss = 0.0349098 (* 1 = 0.0349098 loss)
I0411 21:31:33.262231 31883 solver.cpp:253] Optimization Done.
I0411 21:31:33.262240 31883 caffe.cpp:134] Optimization Done.
```

我们可以发现到后面的准确率达到了 99% 以上，我怀疑是过拟合了。这里用 CPU(Intel(R) Core(TM) i5-4200U CPU @ 1.60GHz) 训练的时间只要 10 分钟左右，速度还是相当快的。这样就能得到训练好的网络参数用来做数据预测。

#### 预测 test.csv 中的数据

test.csv 文件的数据格式与 train.csv 差不多，只是没有 label，因为这些 label 需要我们来预测。我采用了 Python 作预测，Caffe 为 Python 提供了相应的借口，编译 Caffe 时记得顺带编译 Python 模块，当然前提是你先装好相应的依赖库，numpy 肯定逃不掉，具体过程请看[文档](http://caffe.berkeleyvision.org/installation.html#python)。

我们先从 test.csv 中加载图像数据，做相应的预处理后交给 Caffe 做预测。初始化 Caffe 时需要上一步中的网络模型和训练得到的网络模型参数。

```python
#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
import caffe


DATA_ROOT = 'data'
MODEL_ROOT = 'model'
join = os.path.join
TEST = join(DATA_ROOT, 'test.csv')
OUTPUT = join(DATA_ROOT, 'result.csv')
CAFFE_MODEL = join(MODEL_ROOT, 'mnist.caffemodel')
CAFFE_SOLVER = join(MODEL_ROOT, 'lenet.prototxt')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

# load test dataset
logger.info('Load test dataset from %s', TEST)
df = pd.read_csv(TEST)
data = df.values

data = data.reshape((len(data), 28, 28, 1))
data = data / 255.

# set caffe net
net = caffe.Classifier(CAFFE_SOLVER, CAFFE_MODEL)

# predict
logger.info('Start predict')
BATCH_SIZE = 100
iter_k = 0
labels = []
while True:
    logger.info('ITER %d', iter_k)
    batch = data[iter_k*BATCH_SIZE: (iter_k+1)*BATCH_SIZE]
    if batch.size == 0:
        break
    result = net.predict(batch)
    for label in np.argmax(result, 1):
        labels.append(label)
    iter_k = iter_k + 1
logger.info('Prediction Done')

# write to file
logger.info('Save result to %s', OUTPUT)
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)
with open(OUTPUT, 'w') as fd:
    fd.write('ImageId,Label\n')
    for idx, label in enumerate(labels):
        fd.write(str(idx+1))
        fd.write(',')
        fd.write(str(label))
        fd.write('\n')
```

这里，我把预测结果按照 Kaggle 的要求写到了文件中，然后上传到 Kaggle 的评分系统中。

![caffe-kaggle-result]({filename}/images/2015/caffe-kaggle-result.png)

结果准确率有 95.5%，还是相当不错的。

### 小结

DL 已经相当流行了，Caffe 可以大大降低了入门的门槛。大牛们的论文都很开放，也开源了很多代码出来，方便我们这些门外汉学习和入门。大家有兴趣可以多接触接触。另外，我感觉到训练数据在 DL 的重要性可能已经超过了 DL 网络模型本身(虽然 DL 到现在也还是很难解释清楚其中的机理，但是它的成果还是能傲视群雄)。
