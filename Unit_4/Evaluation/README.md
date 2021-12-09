# Evaluation #04

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
