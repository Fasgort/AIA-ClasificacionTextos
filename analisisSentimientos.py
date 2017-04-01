# -*- coding: utf-8 -*-

import json
import re

tweets = [] # Subject, classification, tweetId
f = open('AnalisisSentimientos\\corpus.csv', 'r')
for line in f:
    line = re.sub('["\\n]', '', line)
    tweet = line.split(",")
    try:
        jsonf = open('AnalisisSentimientos\\rawdata\\' + str(tweet[2]) + '.json' )
        json_data = json.load(jsonf)
        if json_data["lang"] != "en":
            continue
        tweet[2] = json_data["text"]
    except:
        continue
    tweets.append(tweet)


