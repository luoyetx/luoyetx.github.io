#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

# AUTHOR = u'zhangjie'
SITENAME = u'zhangjie'
# SITELOGO = '/images/avatar.jpg'
# SITESUBTITLE = u"Computer Vision and Machine Learning"
SITEURL = 'https://luoyetx.github.io'
# GITHUB_URL = 'https://github.com/luoyetx'

# PATH = 'content'

TIMEZONE = 'Asia/Shanghai'

DEFAULT_LANG = u'English'
LOCALE = 'C'

# # Datetime
# DEFAULT_DATE_FORMAT = '%Y/%m/%d'

# # Theme
THEME = './resume'

# # Plugins
# PLUGIN_PATHS = ['./pelican-plugins']
# PLUGINS = ["render_math"]

# # Static Paths
# STATIC_PATHS = ['images']

# # Article and Page URL
# ARTICLE_URL = '{date:%Y}/{date:%m}/{slug}/'
# ARTICLE_SAVE_AS = '{date:%Y}/{date:%m}/{slug}/index.html'
# PAGE_URL = '{slug}/'
# PAGE_SAVE_AS = '{slug}/index.html'

# # Feed generation is usually not desired when developing
# FEED_ALL_ATOM = None
# CATEGORY_FEED_ATOM = None
# TRANSLATION_FEED_ATOM = None
# AUTHOR_FEED_ATOM = None
# AUTHOR_FEED_RSS = None

# # Blogroll
# #LINKS = (('About', '/pages/about/'),)

# DEFAULT_PAGINATION = 10

# # Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True

# # Theme Configure for Flex
# SOCIAL = (('github', 'https://github.com/luoyetx'),
#           ('twitter', 'https://twitter.com/luoyetx'),
#           ('envelope-o', 'mailto://luoyetx@gmail.com'),
#           ('steam', 'http://steamcommunity.com/id/luoyetx'),)

# MAIN_MENU = True
# MENUITEMS = (('Archives', '/archives.html'),
#              ('Categories', '/categories.html'),)

# CC_LICENSE = {
#     'name': 'Creative Commons Attribution-ShareAlike',
#     'version': '4.0',
#     'slug': 'by-sa'
# }

# DISQUS_SITENAME = 'luoyetxblog'



### Resume

NAME = 'Jie Zhang'
TAGLINE = 'Algorithm Engineer'
PIC = 'avatar.jpeg'

EMAIL = 'luoyetx@gmail.com'
#PHONE = '(+91) 9560038966'
WEBSITE = 'luoyetx.github.io'
#LINKEDIN = 'suhebk'
GITHUB = 'luoyetx'
#TWITTER = '@iamsuheb'

CAREER_SUMMARY = 'I am currectly working at Shopee.'

# SKILLS = [
# 	{
# 		'title': 'C/C++',
#    		'level': '80'
#    	},
#   	{
#   		'title': 'Python',
#    		'level': '80'
#    	}
# ]

PROJECT_INTRO = 'Some projects I have worked on.'

PROJECTS = [
	{
		'title': 'luoyetx/mini-caffe',
        'url': 'https://github.com/luoyetx/mini-caffe',
		'tagline': 'Minimal runtime core of Caffe, Forward only, GPU support and Memory efficiency.'
	},
	{
		'title': 'Pixivly/Pixivly',
        'url': 'https://github.com/Pixivly/Pixivly',
		'tagline': 'Daily Top Illustrations On Pixiv.net.'
	},
	{
		'title': 'luoyetx/face-alignment-at-3000fps',
        'url': 'https://github.com/luoyetx/face-alignment-at-3000fps',
		'tagline': 'C++ implementation of Face Alignment at 3000 FPS via Regressing Local Binary Features.'
	},
    {
		'title': 'luoyetx/deep-landmark',
        'url': 'https://github.com/luoyetx/deep-landmark',
		'tagline': 'Predict facial landmarks with Deep CNNs powered by Caffe.'
	},
    {
        'title': 'luoyetx/mx-lsoftmax',
        'url': 'https://github.com/luoyetx/mx-lsoftmax',
        'tagline': 'mxnet version of Large-Margin Softmax Loss for Convolutional Neural Networks.'
    }
]

# LANGUAGES = [
# ]

INTERESTS = [
	'Gaming',
	'Coding',
]

EXPERIENCES = [
	{
		'job_title': 'Search Architecture Engineer',
		'time': 'May 2022 - Present',
		'company': 'Shopee, Search & Recommend',
		'details': 'Search for Shopee'
	},
	{
		'job_title': 'Search Algorithm Engineer',
		'time': 'July 2018 - April 2022',
		'company': 'ByteDance, Search Group',
		'details': 'Search for TikTok && Douyin'
	},
	{
		'job_title': 'Research Intern',
		'time': 'June 2017 - Aug 2017',
		'company': 'SenseTime, OCR Team',
		'details': 'Text recognition and unsupervised data generation.'
	},
]

EDUCATIONS = [
	{
		'degree': 'Master',
		'meta': 'Huazhong University of Science and Technology (HUST)',
		'time': '2015 - 2018'
	},
	{
		'degree': 'Bachelor',
		'meta': 'Huazhong University of Science and Technology (HUST)',
		'time': '2011 - 2015'
	}
]
