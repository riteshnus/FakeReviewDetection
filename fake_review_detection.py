# Databricks notebook source
# File location and type
file_location = "/FileStore/tables/classify_data.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

df.show()

# COMMAND ----------

# Create a view or table

temp_table_name = "classify_data_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `classify_data_csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "classify_data_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.options.display.mpl_style = 'default'

from pyspark.ml import *
from pyspark.ml.classification import *
from pyspark.ml.feature import *
from pyspark.ml.param import *
from pyspark.ml.tuning import *
from pyspark.ml.evaluation import *
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import rand 
from sklearn.metrics import classification_report
from time import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row

# COMMAND ----------

df = df.filter(df.text.isNotNull())
df = df.filter(df.cat.isNotNull())
df = df[['cat', 'text']]
df = df.withColumn("cat", df.cat.cast(DoubleType()))
df = df.filter(df.cat.isin([0.000, 1.000]))
df.groupby('cat').count().toPandas()
df = df.selectExpr("cat as label", "text as review")
df.show()

# COMMAND ----------

# convert the distinct labels in the input dataset to index values
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
# tokenizer 
tokenizer = RegexTokenizer(inputCol="review", outputCol="words", pattern="\W")##'\w' remove none-word letters
df_tokenized = tokenizer.transform(df)
# remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
df_removed = remover.transform(df_tokenized)
# Convert to TF words vector
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures")
df_TF = hashingTF.transform(df_removed)
# Convert to TF*IDF words vector
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(df_TF)
df_idf = idfModel.transform(df_TF)
for features_label in df_idf.select("features", "label").take(3):
    print(features_label)

# COMMAND ----------



# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=20)

# from pyspark.mllib.clustering import LDA, LDAModel
# ldaModel = LDA.train(df['review'], k=3)

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, kmeans])

model = pipeline.fit(df)

results = model.transform(df)
results.cache()

model.params
display(results.groupBy("prediction").count()) 

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# COMMAND ----------

# Split data aproximately into training (80%) and test (20%)
(train, test)=df.randomSplit([0.8,0.2], seed = 0)
# Cache the train and test data in-memory 
train = train.cache()
test = test.cache()
print('Sample number in the train set : {}'.format(train.count()))
print('Sample number in the test set : {}'.format(test.count()))
train.groupby('label').count().toPandas()

# COMMAND ----------

def grid_search(p1,p2,p3,p4):
    lr = LogisticRegression()
    pipeline = Pipeline(stages=[labelIndexer,tokenizer, remover, hashingTF, idfModel, lr])
  
    #Create ParamGrid for Cross Validation
    paramGrid = (ParamGridBuilder()
                 .addGrid(hashingTF.numFeatures, [p1])
                 .addGrid(lr.regParam, [p2])
                 .addGrid(lr.elasticNetParam, [p3])
                 .addGrid(lr.maxIter, [p4])
                 .build())
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=4)
    
    ########  Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(train)
    # average cross-validation accuracy metric/s on all folds
    average_score = cvModel.avgMetrics
    print('average cross-validation accuracy = {}'.format(average_score[0]))
    return average_score[0]

# COMMAND ----------

score=0.0
for p1 in [45000,50000,55000]:
    for p2 in [0.09,0.10,0.11]:
        for p3 in [0.09,0.10,0.11]:
            for p4 in [9,10,11]:
                t0 = time()
                print('(numFeatures,regParam,elasticNetParam,maxIter)=({},{},{},{})'.format(p1,p2,p3,p4))
                average_score=grid_search(p1,p2,p3,p4)
                tt = time() - t0
                print("Classifier trained in {} seconds".format(round(tt,3)))
                if average_score > score:
                    print('################ Best score ######################')
                    params=(p1,p2,p3,p4)
                    score=average_score
print('Best score is {} at params ={}'.format(score, params))

# COMMAND ----------

def Data_modeling(train, test, pipeline, paramGrid):
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=4)
    
    ########  Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(train)
    
    ########  Make predictions on on the test data
    prediction = cvModel.transform(test)
    average_score = cvModel.avgMetrics
    print('average cross-validation accuracy = {}'.format(average_score[0]))
    ######## Calculate accuracy of the prediction of the test data
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy_score=evaluator.evaluate(prediction)
    # another way to calculate accuracy 
    #correct=prediction.filter(prediction['label']== prediction['prediction']).select("label","prediction")
    #accuracy_score = correct.count() / float(test.count())  
    print('Accuracy in the test data = {}'.format(accuracy_score))
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score=evaluator.evaluate(prediction)
    print('F1 score in the test data = {}'.format(f1_score))
    
    
    ######## Print classification_report
    prediction_and_labels=prediction.select("label","prediction")
    y_true = []
    y_pred = []
    for x in prediction_and_labels.collect():
        xx = list(x)
        try:
            tt = int(xx[1])
            pp = int(xx[0])
            y_true.append(tt)
            y_pred.append(pp)
        except:
            continue

    target_names = ['fake', 'not_fake']
    print(classification_report(y_true, y_pred, target_names=target_names))
    return 

# COMMAND ----------

# trained by a logistic regression 
lr = LogisticRegression()
# Build a pipeline
pipeline = Pipeline(stages=[labelIndexer,tokenizer, remover, hashingTF, idfModel, lr])

# Create ParamGrid for Cross Validation 
paramGrid = (ParamGridBuilder()
             .addGrid(hashingTF.numFeatures, [50000])
             .addGrid(lr.regParam, [0.10])
             .addGrid(lr.elasticNetParam, [0.10])
             .addGrid(lr.maxIter, [10])
             .build())
# Execute 4-folds cross validation for hyperparameter tuning, model prediction and model evaluation.
Data_modeling(train, test, pipeline, paramGrid)

# COMMAND ----------

# trained by a Naïve Bayes 
nb = NaiveBayes()
# Build a pipeline
pipeline = Pipeline(stages=[labelIndexer,tokenizer, remover, hashingTF, idfModel, nb])
# Create ParamGrid for Cross Validation 
paramGrid = (ParamGridBuilder()
             .addGrid(hashingTF.numFeatures, [40000])
             .addGrid(nb.smoothing, [1.0])
             .build())
# Execute 4-folds cross validation for hyperparameter tuning, model prediction and model evaluation.
Data_modeling(train, test, pipeline, paramGrid)

# COMMAND ----------

# trained by a Decision Tree 
dt = DecisionTreeClassifier(labelCol="indexedLabel",impurity="entropy")
# Build a pipeline
pipeline = Pipeline(stages=[labelIndexer,tokenizer, remover, hashingTF, idfModel, dt])
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(hashingTF.numFeatures, [70000])
             .addGrid(dt.maxDepth, [25])
             .addGrid(dt.minInstancesPerNode, [4])
             .build())
# Execute 4-folds cross validation for hyperparameter tuning, model prediction and model evaluation.
Data_modeling(train, test, pipeline, paramGrid)

# COMMAND ----------

rf = RandomForestClassifier(labelCol="indexedLabel",impurity="entropy", seed=5043)
# Build a pipeline
pipeline = Pipeline(stages=[labelIndexer,tokenizer, remover, hashingTF, idfModel, rf])

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(hashingTF.numFeatures, [50000])
             .addGrid(rf.numTrees, [31])
             .addGrid(rf.maxDepth, [29])
             .addGrid(rf.minInstancesPerNode, [1])
             .build())
# Execute 4-folds cross validation for hyperparameter tuning, model prediction and model evaluation.
Data_modeling(train, test, pipeline, paramGrid)

# COMMAND ----------

# trained by a Gradient Boosted Tree 
gbt = GBTClassifier(labelCol="indexedLabel")
# Build a pipeline
pipeline = Pipeline(stages=[labelIndexer,tokenizer, remover, hashingTF, idfModel, gbt])
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(hashingTF.numFeatures, [60000])
             .addGrid(gbt.maxIter, [25]) #(default: 20)
             .addGrid(gbt.maxDepth, [19])
             .addGrid(gbt.minInstancesPerNode, [2])
             .build())
# Execute 4-folds cross validation for hyperparameter tuning, model prediction and model evaluation.
Data_modeling(train, test, pipeline, paramGrid)
