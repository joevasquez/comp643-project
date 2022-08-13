
import nltk
import pyspark
import re

#Use first time to download key packages
"""nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')"""

#JAVA_HOME = /usr/lib/jvm/java-8-openjdk-amd64
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

sc = pyspark.SparkContext()

#Array of file names
files = [
    "data/data/abc_news_86680728811.csv",
    "data/data/fox_news_15704546335.csv",
    "data/data/bbc_228735667216.csv",
    "data/data/nbc_news_155869377766434.csv",
    "data/data/cbs_news_131459315949.csv",
    "data/data/npr_10643211755.csv",
    "data/data/cnn_5550296508.csv",
    "data/data/the_los_angeles_times_5863113009.csv"
]

#Read the data file into an RDD (starting with one for now)
datafile = sc.textFile('data/data/abc_news_86680728811.csv')
print("Rows in the csv: ", datafile.count())

#Split each line by , to extract field values¶
values = datafile.map(lambda x: x.split(','))
columns = values.take(1)
print("Columns: ", columns)

#Filter RDD where rows contain a NULL value
values = values.filter(lambda row: 'NULL' not in row)
print(values.take(1))

#Find the named entities and split into an array
def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
            if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            if current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                            continuous_chunk.append(named_entity)
                            current_chunk = []
            else:
                    continue
    return continuous_chunk

#Clean up the RDD to have the id, caption, likes, and comments
values = values.map(lambda x: (x[0].replace('"\ufeff""', '').replace('"""',''), (x[3], x[4], x[8], x[9], x[10])))
print("Updated RDD ", values.take(2))

#Apply named entity recognition to RDD
values = values.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], get_continuous_chunks(x[1][1]))))
print("Values with ER: ", values.take(8))