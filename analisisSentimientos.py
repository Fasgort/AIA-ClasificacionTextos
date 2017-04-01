# -*- coding: utf-8 -*-

import json
import re

tweets = [] # Subject, classification, tweetId
f = open('AnalisisSentimientos\\corpus.csv', 'r')
for line in f:
    line = re.sub('["\\n]', '', line)
    tweets.append(line.split(","))


