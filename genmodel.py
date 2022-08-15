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

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession\
    .builder\
    .appName("DecisionTreeRegressionExample")\
    .getOrCreate()

#import pyspark
#sc = pyspark.SparkContext()
#sc.stop()

datafile = spark.read.csv("data/data/abc_news_86680728811.csv").rdd
values = datafile.map(lambda x: x.split(','))
values = values.filter(lambda row: 'NULL' not in row)
values = values.map(lambda x: (x[0].replace('"\ufeff""', '').replace('"""',''), (x[3], x[4], x[8], x[9], x[10], x[17])))

def find_entities(string):
    doc = nlp(string)
    array = [(X.text, X.label_) for X in doc.ents]
    result = filter(lambda x: x[1]=="PERSON", array)
    return list(result)

def find_sentiment(string):
    blob_text = TextBlob(string)
    return (blob_text.sentiment.polarity, blob_text.sentiment.subjectivity) 

values = values.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], find_entities(x[1][0]+x[1][1]), find_sentiment(x[1][0]+x[1][1]))))

def input_array(entities, polarity, subjectivity):
    trump_count = 0;
    clinton_count = 0;
    obama_count = 0;
    for i in entities:
        if "obama" in i[0].lower():
            obama_count = obama_count + 1
        if "clinton" in i[0].lower():
            clinton_count = clinton_count + 1
        if "trump" in i[0].lower():
            trump_count = trump_count + 1
    return([obama_count, clinton_count, trump_count, polarity, subjectivity])                   

values = values.map(lambda x: (input_array(x[1][5], x[1][6][1], x[1][6][0]), x[1][3]))
data = values.toDF()
print(data.show(4))

#data = spark.read.csv("data/data/abc_news_86680728811.csv")
#print(data.show(4))
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="_c1", outputCol="_c2", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

treeModel = model.stages[1]
# summary only
print(treeModel)