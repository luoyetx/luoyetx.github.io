Title: json vs simplejson vs ujson
Date: 2015-01-03
permalink: json-vs-simplejson-vs-ujson
Category: Translation


本文为原创翻译，原文地址在[这里](https://medium.com/@jyotiska/json-vs-simplejson-vs-ujson-a115a63a9e26)

JSON已经毫无争议地成为现在最常用的数据交换格式。Python中有两个常用的库来处理json数据，一个是Python标准库中自带的`json`，另一个则是`simplejson`，这个库是纯Python实现，并做了相应的优化。这篇博文的目的是向大家介绍[`ultrajson`](https://github.com/esnme/ultrajson)，也叫做`Ultra JSON`，这个库使用C实现的，执行速度非常快。

我们对三个常用的json操作做了性能测评，这三个操作是**load**，**loads**，**dumps**。我们创建一个字典类型，包含_id_，_name_，_address_这三个键。再利用**json.dumps()**将字典数据编码并保存到一个文件中。然后我们分别用**json.loads()**和**json.load()**从文件中加载数据。通过_10000_，_50000_，_100000_，_200000_，_1000000_个这样的字典数据，我们来测试三个库在这些操作上的时间消耗。

### 利用dumps操作一个一个保存数据

利用**json.dumps()**操作一个一个地保存字典数据，我们得到了如下数据。

![json-1]({filename}/images/2015/json-1.png)

我们发现`json`的性能比`simplejson`要高，但是`ultrajson`的速度将近是`json`的4倍。

### 利用dumps操作直接保存所有数据

在这个测试中，我们把所有字典数据放在一个list列表中，并用**json.dumps()**保存这个list列表。

![json-2]({filename}/images/2015/json-2.png)

`simplejson`和`json`表现得差不多，但是`ultrajson`依旧比它们快1.5倍。接下来我们看看这三个库在load和loads操作上的对比。

### 利用load操作加载数据

我们用load操作来加载数据，这个数据是一个列表，里面放着字典数据。

![json-3]({filename}/images/2015/json-3.png)

我们惊奇地发现`simplejson`比另外两个库表现得都要好。`ultrajson`的性能很接近`simplejson`，而它们的速度都将近是`json`的4倍。

### 利用loads操作加载数据

这个测试中，我们利用`json.loads()`从文件中加载数据。

![json-4]({filename}/images/2015/json-4.png)

`ultrajson`又一次打败了其他两个库，比`json`快将近6倍，比`simplejson`快3倍。

做完这些测试之后，结果很明显。在任何情况下都应该使用`simplejson`来替代`json`，而且`simplejson`这个库本身受到很好的维护。如果你追求速度，那么可以使用`ultrajson`，但是你要记住，这个库在不是序列化数据的情况下表现并不好。当然，如果你只是处理文本数据的话，那就没什么可以担忧的了。
