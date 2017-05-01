Title: Way to implement custom Layer for Deep Learning framework
Date: 2017-01-07
Slug: implement-custom-layer
Category: Machine Learning


It's a common situation that we may need to implement a custom operator or layer for the Deep Learning framework we are using. When I mean implement a Layer or Operator for the framework, it's because the framework doesn't offer us the Operation we want. Sometimes, awesome paper appears with strange functions that not supported by the framework. Sometimes, you want to change the behavior of traditional Layer implementation that can suits your demand. But mostly, you may want to create a new function to adapt it to the neural network that can help you get better result. As a result, you may need to create a Layer or Operator for the framework you use.

### Difference between Operator and Layer

DL frameworks like [Caffe](https://github.com/BVLC/caffe) and [Torch](http://torch.ch/) use Layer for their basic network components while [MXNet](https://github.com/dmlc/mxnet) and [TensorFlow](https://www.tensorflow.org/) use Operator. There is little difference between Operator and Layer if you only focus on the implementation of Forward and Backward operation. Layer usually holds the learnable parameters by themselves while Operator only focus on the operation and let other part of the framework to consider about the parameters. We can abstract these two conception easily using the following Python code.

```python
class Layer(object):
    '''an example for Layer
    '''

    def __init__(self, initializer):
        '''initialize learnable parameters

        Parameters
        ----------
        initializer: way to initialize parameters
        '''
        self.params = {
            'weight': initializer('layer_weight'),
        }

    def forward(self, is_train, in_data, out_data):
        '''perform forward

        Parameters
        ----------
        is_train: train or test
        in_data: input data to this layer
        out_data: output data of this layer
        '''
        pass

    def backward(self, in_data, out_data, in_grad, out_grad):
        '''perform backward

        Parameters
        ----------
        in_data: input data to this layer
        out_data: output data of this layer
        in_grad: gradient w.r.t. to in_data, backprop to former layers
        out_grad: gradient w.r.t. to out_data, backprop from latter layers
        '''
        pass

    def update(self, updater):
        '''update learnable parameters

        Parameters
        ----------
        updater: updater using different optimize strategy to update parameters
        '''
        updater(self.params)


class Operator(object):
    '''an example for Operator
    '''

    def __init__(self):
        '''initialize operator
        '''
        pass

    def forward(self, is_train, in_data, out_data):
        '''perform forward

        Parameters
        ----------
        is_train: train or test
        in_data: input data to this layer including learnable parameters attached to this Operator
        out_data: output data of this layer
        '''
        pass

    def backward(self, in_data, out_data, in_grad, out_grad):
        '''perform backward

        Parameters
        ----------
        in_data: input data to this layer
        out_data: output data of this layer
        in_grad: gradient w.r.t. to in_data, backprop to former layers
        out_grad: gradient w.r.t. to out_data, backprop from latter layers
        '''
        pass

    def infer_shape(self, in_shape, out_shape):
        '''infer data shape of in_data and out_data
        this helps framework to collect the information about operator

        Parameters
        ----------
        in_shape: in_data shape
        out_shape: out_data shape
        '''
        pass
```

Since Layer holds the parameters themselves, they may need initializer and updater to initialize and update them. However, for most frameworks, Layer holds the parameters don't need to care about the parameters update, all they need to do is put the gradients w.r.t. parameters in some place where the framework can fetch. Actually, the initialization part can also do in this way. Somehow, there seems little difference between Layer and Operator, as Layer accesses its parameters in its own class member while Operator accesses the parameters through `in_data`. But just because of this little difference, Layers are not easy (but still possible) to share parameters between each other while Operator can easily do it. It's important for RNN but not a common case for CNN. And that's a reason why frameworks like MXNet and TensorFlow use Operator instead of Layer as their basic network component.

Regardless of the difference between Layer and Operator, we still need to implement Forward and Backward for them. There's nothing difference or special. They go the same way. So we will use Layer for the next part, but it reads parameters from `in_data` which like a Operator.

### Write down formulas

Before we write any code, the first thing we need to do is to figure out all the formula that our Layer need. Let's consider a function which acts like a fully connected Layer, but will modify output result at some location. This function is adapted from paper [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf). The original function is kind of complex, I simplify it for the demonstration.

We define the function below.

$$ f_{i, j} = w_j^T \cdot x_i \quad j \neq y_i $$

$$
\begin{eqnarray}
f_{i, y_i} =
\begin{cases}
w_{y_i}^T \cdot x_i & w_{y_i}^T \cdot x_i < 0 \\
k * w_{y_i}^T \cdot x_i & w_{y_i}^T \cdot x_i > 0
\end{cases}
\end{eqnarray}
$$

It's the same thing as fully connected layer does except $f_{i, y_i}$ will be smaller than original one if $f_{i, yi} > 0$. What's more, $0 < k < 1$ is a hyperparameter of this Layer. We also omit bias term. Then we need to calculate the derivatives for $x$ and $w$.

$$ \frac {\partial f_{i, j}} {\partial x_i} = w_j \quad j \neq y_i $$

$$
\begin{eqnarray}
\frac {\partial f_{i, y_i}} {\partial x_i} =
\begin{cases}
w_{y_i} & w_{y_i} \cdot x_i < 0 \\
k * w_{y_i} & w_{y_i} \cdot x_i > 0
\end{cases}
\end{eqnarray}
$$

$w$ goes the same way.

$$ \frac {\partial f_{i, j}} {\partial w_j} = x_i \quad j \neq y_i $$

$$
\begin{eqnarray}
\frac {\partial f_{i, y_i}} {\partial w_{y_i}} =
\begin{cases}
x_i & w_{y_i} \cdot x_i < 0 \\
k * x_i & w_{y_i} \cdot x_i > 0
\end{cases}
\end{eqnarray}
$$

It's time to bring Loss $J$ in. Then we can write gradient w.r.t. $x$ and $w$.

$$
\begin{eqnarray}
\frac {\partial J} {\partial x_i} &=& \sum_j \frac {\partial J} {\partial f_{i, j}} \frac {\partial f_{i, j}} {\partial x_i} \\
&=& \sum_{j, j \neq y_i} \frac {\partial J} {\partial f_{i, j}} w_j + \frac {\partial J} {\partial f_{i, y_i}} \frac {\partial f_{i, y_i}} {\partial x_i}
\end{eqnarray}
$$

$$
\begin{eqnarray}
\frac {\partial J} {\partial w_j} &=& \sum_i \frac {\partial J} {\partial f_{i, j}} \frac {\partial f_{i, j}} {\partial w_j} \\
&=& \sum_{i, j \neq y_i} \frac {\partial J} {\partial f_{i, j}} x_i + \sum_{i, j = y_i} \frac {\partial J} {\partial f_{i, y_i}} \frac {\partial f_{i, y_i}} {\partial w_{y_i}}
\end{eqnarray}
$$

With the above formulas, we now know how the Forward and Backward of our Layer should do.

### Implement the Layer

Let's implement Forward and Backward in Python. It's your choice to pick a programming language to implement. I happens to use Python a lot and most deep learning framework have support for Python. It's a good idea to choose a language that the framework you use supports. You can easily wrap it after you finish the implementation. In this step, we really don't need to consider the performance of implementation as long as it can work.

```python
def forword(self, is_train, in_data, out_data):
    X = in_data['X']
    W = in_data['W']
    label = in_data['label']
    # traditional fully connected layer
    out = X.dot(W.T)
    if is_train:
        # some modification
        for i in range(len(X)):
            yi = int(label[i])
            if out[i, yi] > 0:
                out[i, yi] *= self.k
    out_data['output'] = out
```

The Forward function is easy since it's normally a fully connected layer that may modify the output $f_{i,y_i}$. `is_train` is used to indicate whether current context is train or test. We only want to modify $f_{i, y_i}$ during training.

```python
def backward(self, in_data, out_data, in_grad, out_grad):
    X = in_data['X']
    W = in_data['W']
    label = in_data['label']
    out = out_data['output']
    o_grad = out_grad['output']
    # traditional fully connected
    x_grad = o_grad.dot(W)
    w_grad = o_grad.T.dot(X)
    # gradient w.r.t. X
    for i in range(X.shape[0]):
        yi = int(label[i])
        if out[i, yi] > 0:
            x_grad[i] += self.k * W[yi] - W[yi]
    # gradient w.r.t W
    for j in range(W.shape[0]):
        for i in range(X.shape[1]):
            yi = int(label[i])
            if yi == j and out[i, yi] > 0:
                w_grad += self.k * X[i] - X[i]
    in_grad['X'] = x_grad
    in_grad['W'] = w_grad
```

Backward is a little tricky, we can reuse the output result of `out` to know if we have modify $f_{i, y_i}$. Also we can reuse the result of fully connected layer's backward operation.

$$
\frac {\partial J} {\partial x_i} = \sum_j \frac {\partial J} {\partial f_{i, j}} w_j + \frac {\partial J} {\partial f_{i, y_i}} (\frac {\partial f_{i, y_i}} {\partial x_i} - w_{y_i})
$$

$$
\frac {\partial J} {\partial w_j} = \sum_i \frac {\partial J} {\partial f_{i, j}} x_i + \sum_{i, j = y_i} \frac {\partial J} {\partial f_{i, y_i}} (\frac {\partial f_{i, y_i}} {\partial w_{y_i}} - x_i)
$$

the first part of two formulas is exactly what fully connected layer does.

### Gradient Check

Once you have write done the code, gradient check is important for you to verify the correctness of your implementation. The key idea is below.

$$ f'(x) = \frac {f(x + \Delta x) - f(x - \Delta x)} {2 \Delta x} $$

The formula evaluate the derivative at `X`. In this way, we can evaluate the gradient of data and parameter using Layer's Forward. We can also calculate this derivative using Layer's Backward we implement. For example, we can choose one element from `X`, we call Layer's Forward and Backward, and get the gradient from `grad` for this one element. Next, we modify this element to `x-eps` and `x+eps`, call Forward twice and get two `f` values. then we can evaluate the gradient. The calculated and evaluated gradient can be different but shouldn't differ to much.

The problem here is what if my Layer doesn't output a single value but a multi-dimension array, and where comes the gradient w.r.t. my Layer's output. The key is to plug a loss function to the output of the Layer. The most simple loss function we can choose is the L2 function.

$$ J = \frac {1} {2} \sum_i x_i^2 $$

$J$ is easy to calculate and the derivative too.

$$ \frac {\partial J} {\partial x_i} = x_i $$

$$ \frac {\partial J} {\partial X} = X $$

Thus, we can simply plug in this loss function to whatever the output of your Layer may output. The following Python code show an easy way to do gradient check on a Layer.

```python
def gradient_check(layer, in_data, out_data, in_grad, out_grad):
    '''do gradient check for parameter X
    '''
    # loss function
    loss_it = lambda x: np.square(x).sum() / 2

    # suppose X is a 2 dimension array
    eps = 1e-4
    threshold = 1e-2
    for i in range(in_data['X'].shape[0]):
        for j in range(in_data['X'].shape[1]):
            # calculate gradient
            layer.forward(is_train=True, in_data, out_data)
            out_grad['output'] = out_data['output']
            layer.backward(in_data, out_data, in_grad, out_grad)
            gradient = in_grad['X'][i, j]

            # evaluate gradient
            in_data['X'][i, j] -= eps
            layer.forward(is_train=True, in_data, out_data)
            J1 = loss_it(out_grad['output'])
            in_data['X'][i, j] += 2 * eps
            layer.forward(is_train=True, in_data, out_data)
            J2 = loss_it(out_grad['output'])
            gradient_expect = (J2 - J1) / (2 * eps)

            # calculate relative error
            error = abs(gradient_expect - gradient) / (abs(gradient) + abs(gradient_expect))
            if error > threshold:
                print 'gradient check failed on X[%d, %d]'%(i, j)
            else:
                print 'gradient check pass'
```

You can refer to cs231n course note [here](http://cs231n.github.io/neural-networks-3/) for more information about gradient check.

### Test within a toy model

After your implementation pass the gradient check, you should put your Layer into the DL framework you use. This brings other important things in. How to develop a new Layer for the DL framework? Most DL frameworks shall have documents about how to write custom Layer or Operator. They also may offer a demonstration of writing the Layer in Python or C++. Read the document and the code, you also need to understand how the framework process the data and basic idea of how the framework run your Layer. If you want to write the code in C++/CUDA, the best way you can go is to read the source code of Layer implementation in the framework. They're the best examples you can refer to.

It's also important that you should use a small network and data set to verify the efficiency of your implementation. Sometimes, passing the gradient check doesn't really mean your Layer implementation is perfect. The gradient check can't cover all situation. There might be bugs that only happens in a rarely situation. Or for some of your Layer inputs, your implementation may have some numeric issue like float underflow which cause the result wrong. Since gradient check is not perfect, it's always a good idea to deploy your Layer implementation on a toy model and see if it works the way your want (at least it shouldn't give you the wrong result).

### Optimize your implementation

Once you verify the correctness and efficiency of your Layer implementation, you may want to optimize it to get a better performance. Since most framework support using Python to implement the Layer, you can still stick to Python and optimize the code more vectorized. Then, you may want to implement it using CUDA which makes your Layer can run on GPUs. Nowadays, we depends so much on GPUs to run deep learning framework to train neural networks. You should learn some knowledge about [CUDA](https://developer.nvidia.com/cuda-zone) if you want the implementation of your Layer gets better performance.

### Summary

In a summary, If you need to implement a custom Layer for the deep learning framework, you should implement it using Python or some other language you are familiar and easy to debug. Do gradient check to verify the correctness of your implementation. Next, you need to put the Layer into the DL framework you use, this requires much that you also need to know how the framework handle and represent the Layer and Data. Train a toy network after your Layer can work with the framework to verify the efficiency. After all, if the performance is poor, you may need to optimize the Layer using CUDA which makes your Layer run on GPUs. You can take a look at [luoyetx/mx-lsoftmax](https://github.com/luoyetx/mx-lsoftmax) for a reference. It's follow the pipeline I described above.

### References

- [MXNet](https://github.com/dmlc/mxnet)
- [Caffe](https://github.com/BVLC/caffe)
- [TensorFlow](https://www.tensorflow.org/)
- [Torch](http://torch.ch/)
- [cs231n](http://cs231n.stanford.edu/)
- [mx-lsoftmax](https://github.com/luoyetx/mx-lsoftmax)
