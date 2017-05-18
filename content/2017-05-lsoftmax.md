Title: Derivatives for L-Softmax
Date: 2017-05-03
Slug: derivatives-for-lsoftmax
Category: Machine Learning


This post records the derivatives for Large-Margin Softmax. The code can be found [here](https://github.com/luoyetx/mx-lsoftmax).

### Basic

$$ w^Tx = |w||x|cos\theta $$

##### For $x$

$$ \frac{\partial |x|}{\partial x} = \frac{x}{|x|} $$

$$
\frac{\partial cos\theta}{\partial x} = \frac{\partial}{\partial x} \left( \frac{w^Tx}{|w||x|} \right) = \frac{w}{|w||x|} - \frac{(w^Tx)x}{|w||x|^3}
$$

$$ \frac{\partial sin^2\theta}{\partial x} = \frac{\partial}{\partial x} \left( 1-cos^2\theta \right) = -2cos\theta \frac{\partial cos\theta}{\partial x} $$

$$ cos(m\theta) = \sum_{n=0}^{\lfloor \frac {m} {2} \rfloor} (-1)^n {m \choose {2n}} (cos\theta)^{m-2n} (sin^2\theta)^n $$

$$
\frac{\partial cos(m\theta)}{\partial x} = m(cos\theta)^{m-1} \frac{\partial cos\theta}{\partial x} + \sum_{n=1}^{\lfloor \frac{m}{2} \rfloor} (-1)^n{m \choose {2n}} \left[ n(cos\theta)^{m-2n}(sin^2\theta)^{n-1}\frac{\partial sin^2\theta}{\partial x} + (m-2n)(cos\theta)^{m-2n-1}(sin^2\theta)^n\frac{\partial cos\theta}{\partial x} \right]
$$

$$ f = (-1)^k|w||x|cos(m\theta) - 2k|w||x| = \left[ (-1)^kcos(m\theta) - 2k \right]|w||x| $$

$$ \frac{\partial f}{\partial x} = \left[ (-1)^kcos(m\theta)-2k \right] \frac{|w|}{|x|}x + (-1)^k|w||x| \frac{\partial cos(m\theta)}{\partial x} $$

$$
\begin{align}
\frac{\partial J}{\partial x_i} &= \sum_{j,j \neq y_i} \frac{\partial J}{\partial f_{i,j}} \cdot \frac{\partial f_{i,j}}{\partial x_i} + \frac{\partial J}{\partial f_{i,y_i}}\cdot\frac{\partial f_{i,y_i}}{\partial x_i} \\
&= \sum_{j}\frac{\partial J}{\partial f_{i,j}}\cdot w_j + \frac{\partial J}{\partial f_{i,y_i}}\left( \frac{\partial f_{i,y_i}}{\partial x_i} - w_{y_i} \right)
\end{align}
$$

##### For $w$

$$ \frac{\partial |w|}{\partial w} = \frac{w}{|w|} $$

$$
\frac{\partial cos\theta}{\partial w} = \frac{\partial}{\partial x} \left( \frac{w^Tx}{|w||x|} \right) = \frac{x}{|x||w|} - \frac{(w^Tx)w}{|x||w|^3}
$$

$$ \frac{\partial sin^2\theta}{w} = \frac{\partial}{\partial w} \left( 1-cos^2\theta \right) = -2cos\theta \frac{\partial cos\theta}{\partial w} $$

$$ cos(m\theta) = \sum_{n=0}^{\lfloor \frac {m} {2} \rfloor} (-1)^n {m \choose {2n}} (cos\theta)^{m-2n} (sin^2\theta)^n $$

$$
\frac{\partial cos(m\theta)}{\partial w} = m(cos\theta)^{m-1} \frac{\partial cos\theta}{\partial w} + \sum_{n=1}^{\lfloor \frac{m}{2} \rfloor} (-1)^n{m \choose {2n}} \left[ n(cos\theta)^{m-2n}(sin^2\theta)^{n-1}\frac{\partial sin^2\theta}{\partial w} + (m-2n)(cos\theta)^{m-2n-1}(sin^2\theta)^n\frac{\partial cos\theta}{\partial w} \right]
$$

$$ f = (-1)^k|w||x|cos(m\theta) - 2k|w||x| = \left[ (-1)^kcos(m\theta) - 2k \right]|w||x| $$

$$ \frac{\partial f}{\partial w} = \left[ (-1)^kcos(m\theta)-2k \right] \frac{|x|}{|w|}w + (-1)^k|w||x| \frac{\partial cos(m\theta)}{\partial w} $$

$$
\begin{align}
\frac{\partial J}{\partial w_j} &= \sum_{i,y_i \neq j} \frac{\partial J}{\partial f_{i,j}} \cdot \frac{\partial f_{i,j}}{\partial w_j} + \sum_{i,y_i=j} \frac{\partial J}{\partial f_{i,j}} \cdot \frac{\partial f_{i,j}}{\partial w_j} \\
&= \sum_i \frac{\partial J}{\partial f_{i,j}}\cdot x_i + \sum_{i,y_i=j}\frac{\partial J}{\partial f_{i,j}}\cdot \left( \frac{\partial f_{i,j}}{\partial w_j} - x_i \right)
\end{align}
$$

### HandWrite

![lsoftmax-derivatives]({filename}/images/2017/lsoftmax-derivatives.jpg)

### Reference

- [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf)
- [luoyetx/mx-lsoftmax](https://github.com/luoyetx/mx-lsoftmax)
