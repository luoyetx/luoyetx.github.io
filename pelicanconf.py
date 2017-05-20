#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'zhangjie'
SITENAME = u"luoyetx's blog"
SITELOGO = '/images/avatar.jpg'
SITESUBTITLE = u"Computer Vision and Machine Learning"
SITEURL = 'https://luoyetx.github.io'
GITHUB_URL = 'https://github.com/luoyetx'

PATH = 'content'

TIMEZONE = 'Asia/Shanghai'

DEFAULT_LANG = u'English'
LOCALE = 'C'

# Datetime
DEFAULT_DATE_FORMAT = '%Y/%m/%d'

# Theme
THEME = './Flex'

# Plugins
PLUGIN_PATHS = ['./pelican-plugins']
PLUGINS = ["render_math"]

# Static Paths
STATIC_PATHS = ['images']

# Article and Page URL
ARTICLE_URL = '{date:%Y}/{date:%m}/{slug}/'
ARTICLE_SAVE_AS = '{date:%Y}/{date:%m}/{slug}/index.html'
PAGE_URL = '{slug}/'
PAGE_SAVE_AS = '{slug}/index.html'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
#LINKS = (('About', '/pages/about/'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True

# Theme Configure for Flex
SOCIAL = (('github', 'https://github.com/luoyetx'),
          ('twitter', 'https://twitter.com/luoyetx'),
          ('envelope-o', 'mailto://luoyetx@gmail.com'),
          ('weibo', 'http://weibo.com/luoyetx'),
          ('steam', 'http://steamcommunity.com/id/luoyetx'),)

MAIN_MENU = True
MENUITEMS = (('Archives', '/archives.html'),
             ('Categories', '/categories.html'),)

CC_LICENSE = {
    'name': 'Creative Commons Attribution-ShareAlike',
    'version': '4.0',
    'slug': 'by-sa'
}

DISQUS_SITENAME = 'luoyetxblog'
