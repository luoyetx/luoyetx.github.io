Title: Caffe 源码阅读 Layer 加载机制
Date: 2016-02-04
Slug: reading-caffe-3
Category: Machine Learning


Caffe 中的 Layer 是神经网络 Net 的基本结构，Caffe 内部维护一个注册表用于查找特定 Layer 对应的工厂函数。很多同学在 Windows 下使用 Caffe 遇到的一个问题就是运行 Caffe 相关的代码时出现无法找到 Layer，但是这个问题不会在 Linux 平台上出现，这个问题跟编译器有关，同时也是跟 Caffe 注册 Layer 的机制有关。

```c++
F0203 12:50:07.581297 11524 layer_factory.hpp:78] Check failed: registry.count(type) == 1 (0 vs. 1)
Unknown layer type: Convolution (known types: )
```

上面的错误是无法在注册表中找到 Convolution Layer 对应的工厂函数，程序直接崩溃。下面我们就来聊聊 Caffe 的 Layer 加载机制，以及为什么在 VC 下会出现这种问题。

Caffe 的 Layer 注册表其实就是一组键值对，key 为 Layer 的类型而 value 则对应其工厂函数。下面两组宏控制了 Layer 的注册动作。

```c++
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)
```

`REGISTER_LAYER_CLASS` 宏可以实现将特定 Layer 注册到全局注册表中，首先定义一个工厂函数用来产生 Layer 对象，然后调用 `REGISTER_LAYER_CREATOR` 将工厂函数和 Layer 的类型名进行注册，注册时只是用 Layer 的 float 和 double 类型，这是网络实际数据使用到的类型。两个静态变量一个对应 float，另一个对应 double，这两个变量的初始化，也就是它们的构造函数实际上完成 Layer 的注册动作。

```c++
template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};
```

`LayerRegisterer` 对象初始化时实际上又是调用相应类型的 `LayerRegistry` 类的静态方法 `AddCreator`。

```c++
typedef std::map<string, Creator> CreatorRegistry;

static CreatorRegistry& Registry() {
  static CreatorRegistry* g_registry_ = new CreatorRegistry();
  return *g_registry_;
}
```

注册表类型为 `CreatorRegistry`，实际类型为 `std::map<string, Creator>`。可以通过 `Registry` 函数获取注册表的全局单例。而注册的过程就是一个简单的 `map` 操作。

```c++
// Adds a creator.
static void AddCreator(const string& type, Creator creator) {
  CreatorRegistry& registry = Registry();
  CHECK_EQ(registry.count(type), 0)
    << "Layer type " << type << " already registered.";
  registry[type] = creator;
}
```

注册的过程大概就是上面说到的流程。Caffe 中的 Layer 采用静态变量初始化的方式来注册工厂函数到全局注册表中，整个注册过程依赖这些静态变量。那么问题来了，为什么 VC 中的代码无法在注册表中找到 Layer 对应的工厂函数？事实上，VC 中 Caffe 代码的全局注册表是空的，一条记录都没有，问题并不是出在这个全局注册表，而是那些完成注册动作的静态变量。由于这些静态变量存在的意义在于其构造函数完成 Layer 的注册动作，没有任何一段代码会去引用这些静态变量，这个坑在于 VC 默认会优化掉这些静态变量，那么所有这些静态变量对应的构造函数将无法执行，那么注册动作一个都不会触发，导致全局注册表为空，然后在构造网络 Net 时就会崩溃。

在 VC 下解决这个问题的关键是让 VC 编译器不将这些静态变量优化掉，可以在 Linker 的配置中设置依赖项输入，如下图所示。

![caffe-vc-linker]({filename}/images/2016/caffe-vc-linker.png)

通过上述的方法可以保证以静态库的方式链接 Caffe 代码时，Caffe 中的那些静态变量不会被优化掉。另外一种方式是直接将 Caffe 的源码加入到现有工程代码中，直接参与编译（不是编译生成静态库），这样也可以保证静态变量不被优化掉。

Caffe 的这种注册 Layer 的机制在 VC 下有点坑，但也不是不能解决，只要搞清楚 Caffe 内部的机制和 VC 的一些特征，还是很容易弄明白问题所在，进而寻求相应的解决方案。
