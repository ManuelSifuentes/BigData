# Evaluation #01

In the evaluation of Big Data Unit 2, different actions and operations will be carried out with a provided Dataset. The Spark syntax will be used in order to achieve all the expected results.

1. Load into a dataframe Iris.csv

The SparkSession library is imported so that later you can use the methods to start a session in Spark.

```r
val iris_df=spark.read.format("csv").option("header","true").load("iris.csv")
```

- Libraries:

```r
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
```

Elaborate the necessary data cleaning to be processed by the algorithm.

```r
val data = iris_df.withColumn("sepal_length", $"sepal_length".cast(DoubleType)).withColumn("sepal_width", $"sepal_width".cast(DoubleType)).withColumn("petal_length", $"petal_length".cast(DoubleType)).withColumn("petal_width", $"petal_width".cast(DoubleType))
```

2. What are the names of the columns?

The CSV to be used is imported, specifying through the option function that header is true, this means that the first line is identified as the "title" of each column and is saved in the constant df.

```r
data.columns
```

- Result:

```r
res0: Array[String] = Array(sepal_length, sepal_width, petal_length, petal_width, species)
```

3. What is the scheme like?

Using the columns function, you can display the name of the columns that the dataset contains.

```r
data.printSchema()
```

- Result:

```r
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
```

4. Print the first 5 columns.

```r
data.show(5)
```

- Result:

The schema allows you to see the type of data that each column of the CSV has.

```r
+------------+-----------+------------+-----------+-------+
|sepal_length|sepal_width|petal_length|petal_width|species|
+------------+-----------+------------+-----------+-------+
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
|         4.7|        3.2|         1.3|        0.2| setosa|
|         4.6|        3.1|         1.5|        0.2| setosa|
|         5.0|        3.6|         1.4|        0.2| setosa|
+------------+-----------+------------+-----------+-------+
only showing top 5 rows
```

5. Use the describe () method to learn more about the data in the DataFrame.

```r
data.describe().show()
```

- Result:

The schema allows you to see the type of data that each column of the CSV has.

```r
+-------+------------------+-------------------+------------------+------------------+---------+
|summary|      sepal_length|        sepal_width|      petal_length|       petal_width|  species|
+-------+------------------+-------------------+------------------+------------------+---------+
|  count|               150|                150|               150|               150|      150|
|   mean| 5.843333333333335| 3.0540000000000007|3.7586666666666693|1.1986666666666672|     null|
| stddev|0.8280661279778637|0.43359431136217375| 1.764420419952262|0.7631607417008414|     null|
|    min|               4.3|                2.0|               1.0|               0.1|   setosa|
|    max|               7.9|                4.4|               6.9|               2.5|virginica|
+-------+------------------+-------------------+------------------+------------------+---------+
```

6. Make the pertinent transformation for the categorical data which will be our labels to be classified.

The schema allows you to see the type of data that each column of the CSV has.

```r
import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).setOutputCol("features")
val features = assembler.transform(data)
features.show(5)

import org.apache.spark.ml.feature.StringIndexer
val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("indexedLabel").fit(features)
println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)

val splits = features.randomSplit(Array(0.6, 0.4))
val trainingData = splits(0)
val testData = splits(1)

val layers = Array[Int](4, 5, 5, 3)
```

- Result:

```r
features: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 4 more fields]
+------------+-----------+------------+-----------+-------+-----------------+
|sepal_length|sepal_width|petal_length|petal_width|species|         features|
+------------+-----------+------------+-----------+-------+-----------------+
|         5.1|        3.5|         1.4|        0.2| setosa|[5.1,3.5,1.4,0.2]|
|         4.9|        3.0|         1.4|        0.2| setosa|[4.9,3.0,1.4,0.2]|
|         4.7|        3.2|         1.3|        0.2| setosa|[4.7,3.2,1.3,0.2]|
|         4.6|        3.1|         1.5|        0.2| setosa|[4.6,3.1,1.5,0.2]|
|         5.0|        3.6|         1.4|        0.2| setosa|[5.0,3.6,1.4,0.2]|
+------------+-----------+------------+-----------+-------+-----------------+
only showing top 5 rows

import org.apache.spark.ml.feature.StringIndexer
labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_c48c889842df
Found labels: [versicolor, virginica, setosa]
```

7. Build the classification model and explain its architecture.

```r
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(System.currentTimeMillis).setMaxIter(200)

import org.apache.spark.ml.feature.IndexToString
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

val model = pipeline.fit(trainingData)
```

8. Print the model results

The schema allows you to see the type of data that each column of the CSV has.

```r
val predictions = model.transform(testData)

predictions.show(5)

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
```

- Result:

As can be seen, the prediction shows a certainty of 0.9696, showing its efficiency when predicting the categorical of the irises when there are random or unknown samples of its category.

```r
predictions: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 10 more fields]
+------------+-----------+------------+-----------+-------+-----------------+------------+-----------------+--------------------+--------------------+----------+--------------+
|sepal_length|sepal_width|petal_length|petal_width|species|         features|indexedLabel|  indexedFeatures|       rawPrediction|         probability|prediction|predictedLabel|
+------------+-----------+------------+-----------+-------+-----------------+------------+-----------------+--------------------+--------------------+----------+--------------+
|         4.3|        3.0|         1.1|        0.1| setosa|[4.3,3.0,1.1,0.1]|         2.0|[4.3,3.0,1.1,0.1]|[6.20977900230933...|[2.27331552505356...|       2.0|        setosa|
|         4.4|        2.9|         1.4|        0.2| setosa|[4.4,2.9,1.4,0.2]|         2.0|[4.4,2.9,1.4,0.2]|[6.08106085219172...|[1.49549216984019...|       2.0|        setosa|
|         4.4|        3.2|         1.3|        0.2| setosa|[4.4,3.2,1.3,0.2]|         2.0|[4.4,3.2,1.3,0.2]|[6.86084770281904...|[1.89616850698231...|       2.0|        setosa|
|         4.6|        3.2|         1.4|        0.2| setosa|[4.6,3.2,1.4,0.2]|         2.0|[4.6,3.2,1.4,0.2]|[6.83415294074120...|[1.73795463395268...|       2.0|        setosa|
|         4.6|        3.4|         1.4|        0.3| setosa|[4.6,3.4,1.4,0.3]|         2.0|[4.6,3.4,1.4,0.3]|[7.24100106273056...|[6.60366530948576...|       2.0|        setosa|
+------------+-----------+------------+-----------+-------+-----------------+------------+-----------------+--------------------+--------------------+----------+--------------+
only showing top 5 rows

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_98213606e44a
accuracy: Double = 0.9016393442622951
Test Error = 0.09836065573770492
```
