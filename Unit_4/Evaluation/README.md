# Evaluation #04

## Decision Tree

Import libraries

```r
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.log4j._
```

Handle errors in code run

```r
Logger.getLogger("org").setLevel(Level.ERROR)
```

We create a spark session and load the CSV data into a datraframe

```r
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank.csv")
```

Display the schema and the first 5 lines

```r
df.printSchema()
df.show(5)
```

Change the columns "y" by one with binary data

```r
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
```

Create a new VectorAssembler object called assembler for the features

```r
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
```

Rename column "y" to label

```r
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
```

Declare a new StringIndexer object to rename and index the column label to indexedLabel

```r
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)
```

Create a new VectorIndexer object called featureIndexer to index all the data contained in features, creating a new column called indexedFeatures

```r
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4) 
```

Divide the data set into training and test arrays

```r
val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3))
```

Model instance

```r
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
```

Convert the indexed data to its original type

```r
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
```

Create a PIpeline object to unite the classification process into a single object

```r
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
// AJustar el modelo a los datos de entrenamiento
val model = pipeline.fit(trainingData)
```

Perform Classification Prediction with Test Data

```r
val predictions = model.transform(testData)
predictions.select("predictedLabel", "label", "features").show(5)
```

Show model accuracy

```r
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Accuracy = ${accuracy}")
```

## Logistic Regression

Import libraries

```r
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
```

Handle errors in code run

```r
Logger.getLogger("org").setLevel(Level.ERROR)
```

We create a spark session and load the CSV data into a datraframe

```r
val spark = SparkSession.builder().getOrCreate()
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank.csv")
```

Change the columns "y" by one with binary data

```r
val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val clean = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val cleanData = clean.withColumn("y",'y.cast("Int"))
```

Create a new Array with the selected data

```r
val featureCols = Array("age","previous","balance","duration")
```

Create a new VectorAssembler object called assembler for the features

```r
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
val df2 = assembler.transform(cleanData)
```

Rename column "y" to label

```r
val featuresLabel = df2.withColumnRenamed("y", "label")
val dataI = featuresLabel.select("label","features")
```

Divide the data set into training and test arrays

```r
val Array(training, test) = dataI.randomSplit(Array(0.7, 0.3), seed = 12345)
```

Model instance

```r
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
```

Fit the model to the training data

```r
val lrModel = lr.fit(training)
```

Perform the prediction with the test data

```r
var results = lrModel.transform(test)
```

Printing the coefficients and interceptions

```r
println(s"Coefficients: \n${lrModel.coefficientMatrix}")
println(s"Intercepts: \n${lrModel.interceptVector}")
```

Convert the results to RDD using .as and .rdd

```r
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
```

Show model accuracy

```r
val metrics = new MulticlassMetrics(predictionAndLabels)
println(s"Accuracy = ${metrics.accuracy}")
```

## SVM

We import the necessary libraries with which we are going to work

```r
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j._
```

Remove warnings

```r
Logger.getLogger("org").setLevel(Level.ERROR)
```

We create a spark session and load the CSV data into a datraframe

```r
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank.csv")
```

We change the column "y" for one with binary data.

```r
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
```

We generate the features table

```r
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
```

We change the column "y" to the label column

```r
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
```

SVM: It is required to change the numerical categorical values to 0 and 1 respectively

```r
val c1 = feat.withColumn("label",when(col("label").equalTo("1"),0).otherwise(col("label")))
val c2 = c1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val c3 = c2.withColumn("label",'label.cast("Int"))
```

The data is prepared for training and the test

```r
val Array(trainingData, testData) = c3.randomSplit(Array(0.7, 0.3))
```

Model instance using the label and features as predominant values

```r
val linsvc = new LinearSVC().setLabelCol("label").setFeaturesCol("features")
```

Model fit

```r
val linsvcModel = linsvc.fit(trainingData)
```

Transformation of the model with the test data

```r
val lnsvc_prediction = linsvcModel.transform(testData)
lnsvc_prediction.select("prediction", "label", "features").show(10)
```

Show Accuracy

```r
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val lnsvc_accuracy = evaluator.evaluate(lnsvc_prediction)
print("Accuracy of Support Vector Machine is = " + (lnsvc_accuracy))
```

Result:

```r
lnsvc_prediction: org.apache.spark.sql.DataFrame = [label: int, features: vector ... 2 more fields]
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       1.0|    0|[-3058.0,17.0,882...|
|       1.0|    0|[-1944.0,7.0,623....|
|       1.0|    0|[-970.0,4.0,489.0...|
|       1.0|    0|[-839.0,16.0,1018...|
|       1.0|    0|[-754.0,9.0,727.0...|
|       1.0|    0|[-639.0,15.0,585....|
|       1.0|    0|[-537.0,21.0,1039...|
|       1.0|    0|[-477.0,21.0,1532...|
|       1.0|    0|[-468.0,8.0,534.0...|
|       1.0|    0|[-466.0,15.0,901....|
+----------+-----+--------------------+
only showing top 10 rows

evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_051e12751017
lnsvc_accuracy: Double = 0.8830457003184949
Accuracy of Support Vector Machine is = 0.8830457003184949
```

## Decission Tree

## Logistic Regression

## Multilayer Perceptron

We import the necessary libraries with which we are going to work

```r
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline
import org.apache.log4j._
```

Remove warnings

```r
Logger.getLogger("org").setLevel(Level.ERROR)
```

We create a spark session and load the CSV data into a datraframe

```r
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank.csv")
```

We change the column "y" for one with binary data

```r
val df1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val df2 = df1.withColumn("y",when(col("y").equalTo("no"),0).otherwise(col("y")))
val newcolumn = df2.withColumn("y",'y.cast("Int"))
```

We generate the features field with VectorAssembler

```r
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val newDF = assembler.transform(newcolumn)
```

We modify the field "y" by label

```r
val cambio = newDF.withColumnRenamed("y", "label")
```

We select a new df with the fields of 'label' and 'features'

```r
val finalDF = cambio.select("label","features")
```

We change the main label with categorical data from string to an Index

```r
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(finalDF)
```

New variables to define an index to the vectors of the "features" field

```r
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(finalDF)
```

Split the data into train and test

```r
val splits = c3.randomSplit(Array(0.7, 0.3), seed = 1234L)
val trainingData = splits(0)
val testData = splits(1)
```

Specify layers for the neural network.
Input layer of size 5 (features), two intermediate of size 6 and 5 and output of size 2 (classes).

```r
val layers = Array[Int](5, 6, 5, 2)
```

Create an instance of the classification library method with the input field "indexedLabel" and the characteristics of the field "indexedFeatures"

```r
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234L).setMaxIter(100)
```

For demonstration purposes, the prediction is inverted to the string type of the label.

```r
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
```

We join the data created to have a new df with the new fields

```r
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))
```

Let's create a model with the training data

```r
val model = pipeline.fit(trainingData)
```

We generate the prediction with the test data

```r
val prediction = model.transform(testData)
prediction.select("prediction", "label", "features").show(5)
```

We finished a test to know the accuracy of the model and its efficiency.

```r
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(prediction)
```

Result

```r
print("Accuracy of Support Vector Machine is = " + (accuracy))

Found labels: [0, 1]
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       1.0|    0|[-3058.0,17.0,882...|
|       1.0|    0|[-1944.0,7.0,623....|
|       1.0|    0|[-1206.0,15.0,382...|
|       1.0|    0|[-770.0,18.0,618....|
|       1.0|    0|[-556.0,8.0,682.0...|
+----------+-----+--------------------+
only showing top 5 rows

Accuracy of Multilayer Perceptron Classifier is = 0.886684037558685
```
