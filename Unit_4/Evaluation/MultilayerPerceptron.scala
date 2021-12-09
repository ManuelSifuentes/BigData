//Practice MultilayerPerceptron
for(i <- 0 to 30)
{

// We import the necessary libraries with which we are going to work
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

// Remove warnings
Logger.getLogger("org").setLevel(Level.ERROR)

// We create a spark session and load the CSV data into a datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank.csv")

// // We display the data types
// df.printSchema()
// // We show first line
// df.show(1)

// We change the column "y" for one with binary data
val df1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val df2 = df1.withColumn("y",when(col("y").equalTo("no"),0).otherwise(col("y")))
val newcolumn = df2.withColumn("y",'y.cast("Int"))

// // We display the new column
// newcolumn.show(1)

// We generate the features field with VectorAssembler
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val newDF = assembler.transform(newcolumn)

// // We show the features field
// newDF.show(1)

// We modify the field "y" by label
val cambio = newDF.withColumnRenamed("y", "label")
// We select a new df with the fields of 'label' and 'features'
val finalDF = cambio.select("label","features")
// finalDF.show(1)

// We change the main label with categorical data from string to an Index
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(finalDF)

// // We show the category of the data
// println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

// New variables to define an index to the vectors of the "features" field
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(finalDF)

// Split the data into train and test
val splits = c3.randomSplit(Array(0.7, 0.3), seed = 1234L)
val trainingData = splits(0)
val testData = splits(1)

// Specify layers for the neural network:
// Input layer of size 5 (features), two intermediate of size 6 and 5 and output of size 2 (classes)
val layers = Array[Int](5, 6, 5, 2)

// Create an instance of the classification library method with the input field "indexedLabel" and the characteristics of the field "indexedFeatures"
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234L).setMaxIter(100)

// For demonstration purposes, the prediction is inverted to the string type of the label.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// We join the data created to have a new df with the new fields
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

// Let's create a model with the training data
val model = pipeline.fit(trainingData)

// We generate the prediction with the test data
val prediction = model.transform(testData)
prediction.select("prediction", "label", "features").show(5)

// We finished a test to know the accuracy of the model and its efficiency.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(prediction)

// Result
print("Accuracy of Support Vector Machine is = " + (accuracy))

}

//References
//https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3741049972324885/1019862370390522/4413065072037724/latest.html