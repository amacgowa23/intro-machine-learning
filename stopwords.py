# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:30:09 2016

@author: AMacGowan
"""
### English stopwords for text processing"

import nltk
nltk.download()

from nltk.corpus import stopwords

sw = stopwords.words("english")

print len(sw)
