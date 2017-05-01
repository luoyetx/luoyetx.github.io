Title: Scrapy 使用小记 1
Date: 2014-12-28
Slug: using-scrapy-1
Category: Technology


[Scrapy](http://scrapy.org/)是一套由[Python](https://www.python.org/)编写的爬虫框架，基于异步事件驱动的[Twisted](https://twistedmatrix.com/trac/)库。 在Scrapy的框架下，我们可以很方便地编写爬虫来抓取页面。Scrapy官方文档中有一个简单的[教程](http://doc.scrapy.org/en/latest/intro/tutorial.html)。通过这个教程，我们可以基本了解到如何在Scrapy提供的框架下编写代码。

### 安装Scrapy

在Windows下安装Scrapy可能会比较费劲，主要是因为Scrapy依赖的一些库是用C写的，哪怕你在Windows下配置了gcc或者是vc的编译器，还是会因为缺少相应库的头文件而出现编译错误。Scrapy官方文档针对Windows有相应的[安装指南](http://doc.scrapy.org/en/latest/intro/install.html)。如果你不嫌麻烦的话可以照着安装指南来。不过在Windows下我比较推荐一个Python发行包[pythonxy](https://code.google.com/p/pythonxy/)。这个链接估计是常年被墙，大家可以通过别的方法下载这个发行包，我这里提供一个[百度网盘](http://pan.baidu.com/s/1c0s5c56)。pythonxy其实上是一个Python科学计算包集合，里面提供了很多Python的开发库，这些库很多都是有C的拓展，不过pythoxy已经帮我们编译好了，而且还集成了其他有用的Python库，包括Scrapy依赖的库。 Linux和Unix可以很方便的通过pip命令安装Scrapy，大部分*nix发行版中的Python都含有Scrapy依赖的库，所以我们可以直 接使用pip。当然，如果在Windows中安装了pythonxy，我们也可以通过pip来安装。

`$ pip install scrapy`

这样我们就装好了Scrapy库。

### 创建Scrapy Project

我们可以通过Scrapy提供的命令行工具轻松地创建初始项目，这个对我们开发者来说相当的友好，就像[django](https://www.djangoproject.com/)提供的命令行工具一样。我们通过观察初始项目的目录结构和文件命名，可以大致上了解整个项目的结构。

`$ scrapy startproject spixiv`

通过这条命令，我们可以初始化一个scrapy项目。这里的spixiv是项目的名称。我们看看Scrapy为我们生成了哪些东西。

```
  spixiv
    ├── scrapy.cfg
    └── spixiv
      ├── __init__.py
      ├── items.py
      ├── pipelines.py
      ├── settings.py
      └── spiders
        └── __init__.py
```

我们可以看到顶级目录是spixiv，下面有一个scrapy.cfg文件和一个spixiv目录(这个目录也是一个Python包)。scrapy.cfg这个文件一般不用去理会它，他只是向Scrapy提供了项目的配置信息(真正的配置信息其实在settings.py文件中，scrapy.cfg只是在其中有一个字段指向了这个settings.py文件)。

```python
# Automatically created by: scrapy startproject
#
# For more information about the [deploy] section see:
# http://doc.scrapy.org/en/latest/topics/scrapyd.html

[settings]
default = spixiv.settings

[deploy]
#url = http://localhost:6800/
project = spixiv
```

以上是scrapy.cfg文件中的所有内容。我们代码的编写主要在spixiv目录下。

### 分析初始项目结构

个人非常喜欢框架提供startproject这种类似的工具，因为这往往就是这个框架下项目的最佳组织结构。这里不得不提一下django提供的初始项目结构，非常的模块化，从中我们也可以窥探到这些框架自身的组织结构和运行流程。下面我们来分析分析Scrapy为我们创建的初始项目

##### scrapy.cfg

这个文件在上一节中已经提到过，它只是给Scrapy命令行工具提供一个基础的配置信息(这也是为什么我们后面运行scrapy命令时必须和这个文件在同一目录下)。里面的`default`字段提供了项目配置的文件。

##### spixiv/settings.py

```
BOT_NAME = 'spixiv'

SPIDER_MODULES = ['spixiv.spiders']
NEWSPIDER_MODULE = 'spixiv.spiders'
```

这个文件里才是真正的项目配置(其实也没有多少东西)，BOT_NAME指明我们的项目名称(爬虫机器人？)，SPIDER_MODULES告诉Scrapy框架应该在哪些模块中寻找我们编写的爬虫。NEWSPIDER_MODULE这个字段其实可有可无，如果你需要Scrapy为你生成Spider模板的话，那么Scrapy生成的代码就会被写在这里设置的模块下。

##### spixiv/items.py

```python
import scrapy

class SpixivItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass
```

这个文件中主要用来编写我们需要爬取的信息，我们将对抓取到的信息抽象并包装为一个一个的item，而这些item的定义就可以放在这个文件中。后面谈到Scrapy整个框架的流程时，我们可以看到这样做的好处。

##### spixiv/pipelines.py

```python
class SpixivPipeline(object):
    def process_item(self, item, spider):
        return item
```

这个文件里定义了对item的处理行为，默认没有做处理。如果要对item做额外的处理，可以在这里编写代码逻辑，还要在settings.py中添加相应的字段让Scrapy框架来加载我们的处理逻辑(默认不会加载)。

以上就是Scrapy为我们创建的项目结构，非常简洁的结构，下面我们就可以开始编写Spider了。
