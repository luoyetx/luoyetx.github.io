Title: Scrapy 使用小记 2
Date: 2014-12-31
Slug: using-scrapy-2
Category: Technology


### 编写Spider

有了Scrapy为我们创建的初始项目，在这个基础上，我们就可以开始编写spider了。我们编写的spider将放在settings.py中指定的模块中，默认是在spiders模块下。我们需要创建一个文件来写我们的spider，Scrapy启动时会查找相应的spider并加载。

```python
import scrapy

class DmozSpider(scrapy.Spider):
    name = "dmoz"
    allowed_domains = ["dmoz.org"]
    start_urls = [
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Books/",
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Resources/"
    ]

    def parse(self, response):
        filename = response.url.split("/")[-2]
        with open(filename, 'wb') as f:
            f.write(response.body)
```

这是[Scrapy官方教程](http://doc.scrapy.org/en/latest/intro/tutorial.html)中的一个spider例子。name表示了爬虫的名字，allowed_domains里存放这个爬虫允许访问的域名，start_urls用来生成url请求，并将请求结果送入爬虫的parse方法作处理。这个类必须继承自scrapy.Spider，这样，Scrapy才知道这个类代表了爬虫。

### 编写Item

Item代表了从网页中提取出来的信息，我们可以从一个网页中提取到多个item，也可能是多种item，这取决于我们想要从页面中提取的信息。

```python
import scrapy

class DmozItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    desc = scrapy.Field()
```

这段代码同样来自Scrapy的教程，我们自定义的Item需要继承自scrapy.item，每一个字段都是Field类型，可以保存任意类型的Python对象。其实我们可以把Item当作Python的dict使用，而在这里定义的属性名就是它的关键字。

### Item与Spider相结合

有了Item，我们就可以在Spider中编写处理页面的逻辑了，也就是从页面中提取信息并包装成Item，然后抛给Scrapy框架作后续处理。在这里，我们不使用Scrapy教程中的例子，我们自己动手写写，来抓取想要的页面。

我们来抓取[Pixiv](http://www.pixiv.net/)网站上当天排名前50的插画信息。Pixiv是一家日本的插画交流网站，聚集了很多一流的绘画高手，当然插画的内容主要都是二次元的。不管这么多了，我们就来抓抓看。在动手写之前，我们需要分析分析如何抓取，具体来说就是要向哪个url发出请求，在得到请求结果之后得分析页面来提取我们想要的信息。其实P站(即Pixiv)已经有一个[url](http://www.pixiv.net/ranking.php?mode=daily&amp;content=illust&amp;format=json)可以直接从这里获取当天插画排名的json数据。

http://www.pixiv.net/ranking.php?mode=daily&amp;content=illust&amp;format=json。

我们可以访问下面这个链接来看看今天的插画排名。

http://www.pixiv.net/ranking.php?mode=daily&amp;content=illust

通过这里的链接，我们看到的直接是一个网页了，而不是得到json格式的数据。方便起见，我们就直接抓json格式的数据。有时候我们不一定能够直接得到json格式的数据，而是要通过html文件，通过分析html源码来分析出数据，Scrapy也提供了相应的方法来帮助我们分析页面。在简单情况下，我们可以使用Python标准库中的正则表达式模块re，直接从html中提取信息，当这个过程很复杂时，我推荐使用Scrapy为我们提供的工具来分析页面，或者使用第三方页面分析库，比如[Beautifuloup](http://www.crummy.com/software/BeautifulSoup/)。

我们先来写Item，简单起见，我们只提取插画的标题和插画的url地址。

```python
import scrapy
from scrapy import Field

class IllustrationItem(scrapy.Item):
    """Item for Illustration on Pixiv
    """
    title = Field()
    url = Field()
```

就是这么的简单，而我们需要抓取的页面上面已经提到了，再得到这个json数据后，我们可以直接用Python的json模块格式化数据，并一个一个提取信息，包装成Illustrationtem，抛给外层Scrapy框架。

```python
import json
import scrapy
from spixiv.items import IllustrationItem

class PixivSpider(scrapy.Spider):
    """Spider for daily top illustrations on Pixiv
    """
    name = 'pixiv'
    allowed_domains = ['pixiv.net']
    start_urls = ['http://www.pixiv.net/ranking.php?mode=daily&amp;content=illust&amp;format=json']

    def parse(self, response):
        jsondata = json.loads(response.body)

        date = jsondata['date']
        for one in jsondata['contents']:
            item = IllustrationItem()
            item['title'] = one['title']
            item['url'] = one['url']
            yield item
```

我们使用Python标准库中的json模块来操作json格式的数据。这里的Item使用就像Python的dict一样，我们也可以给它的字段赋值复杂类型的对象，比如序列，当然，在后续有处理Item的代码必须得知道每个字段的类型。我们使用yield向外层框架抛出IllustrationItem，以便Scrapy对其作后续处理。

### 开始抓取数据

Scrapy提供了一个简单的命令来启动我们的爬虫。

`$ scrapy crawl [options] spider`

这里的options可以来配置scrapy的行为，而spider则是我们爬虫的名字。直接在项目的根目录下(含有scrapy.cfg的目录)运行`scrapy crawl pixiv`，Scrapy默认会把Item的信息输出到控制台中。

```
2015-01-01 00:04:48+0800 [pixiv] DEBUG: Scraped from &lt;200 http://www.pixiv.net/ranking.php?mode=daily&amp;content=illust&amp;format=json&gt;
        {'title': u'\u5bb6\u65cf\u3068\u592b\u5a66\u3068',
         'url': u'http://i2.pixiv.net/c/240x480/img-master/img/2014/12/29/23/08/03/47845529_p0_master1200.jpg'}
2015-01-01 00:04:48+0800 [pixiv] DEBUG: Scraped from &lt;200 http://www.pixiv.net/ranking.php?mode=daily&amp;content=illust&amp;format=json&gt;
        {'title': u'\u843d\u66f8\u304d\u307e\u3068\u3081 No.8',
         'url': u'http://i1.pixiv.net/c/240x480/img-master/img/2014/12/29/00/44/17/47829688_p0_master1200.jpg'}
2015-01-01 00:04:48+0800 [pixiv] DEBUG: Scraped from &lt;200 http://www.pixiv.net/ranking.php?mode=daily&amp;content=illust&amp;format=json&gt;
        {'title': u'BB\u3061\u3083\u3093',
         'url': u'http://i3.pixiv.net/c/240x480/img-master/img/2014/12/29/17/16/52/47839178_p0_master1200.jpg'}
2015-01-01 00:04:48+0800 [pixiv] DEBUG: Scraped from &lt;200 http://www.pixiv.net/ranking.php?mode=daily&amp;content=illust&amp;format=json&gt;
        {'title': u'-\u6708\u306e\u60f3\u3044-',
         'url': u'http://i1.pixiv.net/c/240x480/img-master/img/2014/12/29/00/00/08/47828516_p0_master1200.jpg'}
```

我们也可以通过options参数将Item保存成json格式的数据。

`$ scrapy crawl pixiv -o illustration.json`

这样 ，Item的数据就会以json格式保存到illustration.json这个文件中了。

### 使用ItemPipeline对Item做进一步处理

Scrapy默认会帮我们建一个ItemPipeline，它什么也没有处理。而且在项目中使用Pipeline还必须配置一下settings.py这个文件

```python
ITEM_PIPELINES = {
    'spixiv.pipelines.SpixivPipeline': 300,
}
```

这里指出了Pipeline定义的位置，是pipelines下的一个类。300是一个优先级数，因为可能不只一个Pipeline想要处理Item，它的取值范围是0～1000，值越小，优先级越高。

```python
class SpixivPipeline(object):
    def process_item(self, item, spider):
        return item
```

这个类是创建项目时Scrapy自动帮我们建的，这也表示我们的Pipleline需要实现process_item这个方法，方法的参数item表示了一个Item实例，spider表示了一个Spider实例，两个代表从spider抛出了一个item。这样我们可以根据spider和item的类型来处理item。比如将item存入数据库中做持久的保存。
