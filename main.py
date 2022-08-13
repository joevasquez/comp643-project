# Import Spark NLP
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline# Start Spark Session with Spark NLP
spark = sparknlp.start()
spark = SparkSession.builder\
    .appName("BBC Text Categorization")\
    .config("spark.driver.memory","8G")\
    .config("spark.memory.offHeap.enabled",True)\
    .config("spark.memory.offHeap.size","8G") \
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5")\
    .config("spark.kryoserializer.buffer.max", "1000M")\
    .config("spark.network.timeout","3600s")\
    .getOrCreate()

# File location and type
file_location = r'data/bbc-text.csv'
file_type = "csv"# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)
  
df.count()

(trainingData, testData) = df.randomSplit([0.7, 0.3], seed = 100)

from pyspark.ml.feature import HashingTF, IDF, StringIndexer, SQLTransformer,IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
 
# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)# stems tokens to bring it to root form
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")# Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)# To generate Term Frequency
hashingTF = HashingTF(inputCol="token_features", outputCol="rawFeatures", numFeatures=1000)# To generate Inverse Document Frequency
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)# convert labels (string) to integers. Easy to process compared to string.
label_stringIdx = StringIndexer(inputCol = "category", outputCol = "label")# define a simple Multinomial logistic regression model. Try different combination of hyperparameters and see what suits your data. You can also try different algorithms and compare the scores.
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0)# To convert index(integer) to corresponding class labels
label_to_stringIdx = IndexToString(inputCol="label", outputCol="article_class")# define the nlp pipeline
nlp_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            stemmer, 
            finisher,
            hashingTF,
            idf,
            label_stringIdx,
            lr,
            label_to_stringIdx])

# fit the pipeline on training data
pipeline_model = nlp_pipeline.fit(trainingData)

# perform predictions on test data
predictions =  pipeline_model.transform(testData)

# import evaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % (accuracy))
print("Test Error = %g " % (1.0 - accuracy))

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % (accuracy))
print("Test Error = %g " % (1.0 - accuracy))

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % (accuracy))
print("Test Error = %g " % (1.0 - accuracy))

pipeline_model.save('/model')