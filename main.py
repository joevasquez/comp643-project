# private class definitions
from enrich_rdd import url_to_string
from gen_wordcloud import create_wordcloud
from plot_chart import plot_chart

from textblob import TextBlob
from bs4 import BeautifulSoup
import requests
import re

import spacy
from spacy import displacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")

import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from wordcloud import WordCloud

import pyspark
sc = pyspark.SparkContext()

#Array of file names
files = [
    ("ABCNews", "data/data/abc_news_86680728811.csv"),
    ("FoxNews", "data/data/fox_news_15704546335.csv"),
    ("BBC", "data/data/bbc_228735667216.csv"),
    ("NBCNews", "data/data/nbc_news_155869377766434.csv"),
    ("CBSNews", "data/data/cbs_news_131459315949.csv"),
    ("NPR", "data/data/npr_10643211755.csv"),
    ("CNN", "data/data/cnn_5550296508.csv"),
    ("LATimes", "data/data/the_los_angeles_times_5863113009.csv")
]

files2 = [
    ("ABCNews", "data/data/abc_news_86680728811.csv"),
    ("FoxNews", "data/data/fox_news_15704546335.csv")
]

results_array = []
polarity_array = []
av_polarity_array = []

def find_entities(string):
    doc = nlp(string)
    array = [(X.text, X.label_) for X in doc.ents]
    result = filter(lambda x: x[1]=="PERSON", array)
    return list(result)

# Sentiment(polarity, subjectivity)
def find_sentiment(string):
    testimonial = TextBlob(string)
    return (testimonial.sentiment.polarity, testimonial.sentiment.subjectivity) 

def main_function(url):
    #Read the data file into an RDD (starting with one for now)
    datafile = sc.textFile(url[1])

    #Split each line by , to extract field values¶
    values = datafile.map(lambda x: x.split(','))

    #Filter out RDD rows where row contain a NULL value
    values = values.filter(lambda row: 'NULL' not in row)

    #Clean up the RDD to have the id, caption, likes, and comments
    values = values.map(lambda x: (x[0].replace('"\ufeff""', '').replace('"""',''), (x[3], x[4], x[8], x[9], x[10], x[17])))

    #ENRICHMENT
    def get_article (link):
        if "com" not in link:
            return ""
        else:
            return url_to_string(link)

    #Apply entity recognition via find_entities(string) function
    values = values.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], find_entities(x[1][0]+x[1][1]), find_sentiment(x[1][0]+x[1][1]))))

    # Build up string of all recognized entities in corpus
    entity_rdd = values.map(lambda x: x[1][5])
    entity_array = entity_rdd.collect()
    
    concat_string = ""
    for i in entity_array:
        for j in i:
            concat_string = concat_string + " " + j[0]
    
    print("NEW ENTITY: ", concat_string)
    
    # Create a word cloud!
    create_wordcloud(url[0], concat_string)    

    all_people = values.flatMap(lambda x: ((j[0], 1) for j in x[1][5]))
    people_count = all_people.reduceByKey(lambda a, b: a + b)
    top_people = people_count.top(200, lambda x : x[1])
    results_array.append((url[0], top_people[:50]))

    print("TOP PEOPLE: ", top_people[:50])
    plot_chart(url[0], top_people[:50])

    polarity_rdd = values.flatMap(lambda x: ((j[0], x[1][6][0]) for j in x[1][5]))
    polarity_rdd = polarity_rdd.reduceByKey(lambda a, b: a + b)
    top_polarity = polarity_rdd.top(1000, lambda x : x[1])
    polarity_array.append((url[0], top_polarity[:50]))

    temp_array = []
    for i in top_people:
        for j in top_polarity:
            if j[0] == i[0]:
                temp_array.append((i[0], i[1], j[1]/i[1]))
    
    av_polarity_array.append((url[0], temp_array))

for i in files:
    main_function(i)

print(av_polarity_array)

#print("TEST: ", url_to_string('https://www.nbcnews.com/politics/donald-trump/trumps-handling-secret-documents-fbi-mar-a-lago-search-rcna42935'))

"""
NOTE: RDD is structured as follows...
    [(ID,  (Message, 
            Description, 
            Likes, 
            Comments, 
            Shares, 
            [ER Count Array]))
    ]

"""