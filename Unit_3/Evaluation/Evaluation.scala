//1. Load into a dataframe Iris.csv
val iris_df=spark.read.format("csv").option("header","true").load("iris.csv")

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
//import org.apache.spark.sql.types.DoubleType

val data = iris_df.withColumn("sepal_length", $"sepal_length".cast(DoubleType)).withColumn("sepal_width", $"sepal_width".cast(DoubleType)).withColumn("petal_length", $"petal_length".cast(DoubleType)).withColumn("petal_width", $"petal_width".cast(DoubleType))

//2. What are the names of the columns?
data.columns

//3. What is the scheme like?
data.printSchema()

//4. Print the first 5 columns.
data.show(5)

//5. Use the describe () method to learn more about the data in the DataFrame.
data.describe().show()

//6. Make the pertinent transformation for the categorical data which will be our labels to be classified.
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

//7. Build the classification model and explain its architecture.
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(System.currentTimeMillis).setMaxIter(200)

import org.apache.spark.ml.feature.IndexToString
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

val model = pipeline.fit(trainingData)

//8. Print the model results
val predictions = model.transform(testData)

predictions.show(5)

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

