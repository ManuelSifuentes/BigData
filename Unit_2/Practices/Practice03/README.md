# Practice #03

Libraries:

```r
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
```

Load and parse the data file, converting it to a DataFrame.

```r
val data = spark.read.format("libsvm").load("../sample_libsvm_data.txt")
```

Index labels, adding metadata to the label column.
Fit on whole dataset to include all labels in index.

```r
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
```

Automatically identify categorical features, and index them.
Set maxCategories so features with > 4 distinct values are treated as continuous.

```r
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
```

Split the data into training and test sets (30% held out for testing).

```r
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
```

Train a RandomForest model.

```r
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
```

Convert indexed labels back to original labels.

```r
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
```

Chain indexers and forest in a Pipeline.

```r
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
```

Train model. This also runs the indexers.

```r
val model = pipeline.fit(trainingData)
```

Make predictions.

```r
val predictions = model.transform(testData)
```

Select example rows to display.

```r
predictions.select("predictedLabel", "label", "features").show(5)
```

Result:

```r
predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[98,99,100,1...|
|           1.0|  0.0|(692,[100,101,102...|
|           0.0|  0.0|(692,[122,123,124...|
|           0.0|  0.0|(692,[122,123,148...|
|           0.0|  0.0|(692,[124,125,126...|
+--------------+-----+--------------------+
only showing top 5 rows
```

Select (prediction, true label) and compute test error.

```r
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")
```

Result:

```r
accuracy: Double = 0.875
Test Error = 0.125
```

```r
val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
```

Result:

```r
Learned classification forest model:
 RandomForestClassificationModel (uid=rfc_d58c33307ff3) with 10 trees
  Tree 0 (weight 1.0):
    If (feature 552 <= 5.5)
     If (feature 356 <= 19.0)
      Predict: 0.0
     Else (feature 356 > 19.0)
      Predict: 1.0
    Else (feature 552 > 5.5)
     If (feature 323 <= 23.0)
      Predict: 1.0
     Else (feature 323 > 23.0)
      Predict: 0.0
  Tree 1 (weight 1.0):
    If (feature 567 <= 8.0)
     If (feature 456 <= 31.5)
      Predict: 0.0
     Else (feature 456 > 31.5)
      Predict: 1.0
    Else (feature 567 > 8.0)
     If (feature 317 <= 8.0)
      Predict: 0.0
     Else (feature 317 > 8.0)
      Predict: 1.0
  Tree 2 (weight 1.0):
    If (feature 385 <= 4.0)
     If (feature 317 <= 158.0)
      Predict: 0.0
     Else (feature 317 > 158.0)
      Predict: 1.0
    Else (feature 385 > 4.0)
     Predict: 1.0
  Tree 3 (weight 1.0):
    If (feature 328 <= 24.0)
     If (feature 439 <= 28.0)
      Predict: 0.0
     Else (feature 439 > 28.0)
      Predict: 1.0
    Else (feature 328 > 24.0)
     Predict: 1.0
  Tree 4 (weight 1.0):
    If (feature 429 <= 11.5)
     If (feature 358 <= 17.5)
      Predict: 0.0
     Else (feature 358 > 17.5)
      Predict: 1.0
    Else (feature 429 > 11.5)
     Predict: 1.0
  Tree 5 (weight 1.0):
    If (feature 462 <= 63.0)
     If (feature 240 <= 253.5)
      Predict: 1.0
     Else (feature 240 > 253.5)
      If (feature 600 <= 5.5)
       Predict: 0.0
      Else (feature 600 > 5.5)
       Predict: 1.0
    Else (feature 462 > 63.0)
     Predict: 0.0
  Tree 6 (weight 1.0):
    If (feature 512 <= 8.0)
     If (feature 289 <= 28.5)
      Predict: 0.0
     Else (feature 289 > 28.5)
      Predict: 1.0
    Else (feature 512 > 8.0)
     Predict: 1.0
  Tree 7 (weight 1.0):
    If (feature 512 <= 8.0)
     If (feature 510 <= 6.5)
      Predict: 0.0
     Else (feature 510 > 6.5)
      Predict: 1.0
    Else (feature 512 > 8.0)
     Predict: 1.0
  Tree 8 (weight 1.0):
    If (feature 462 <= 63.0)
     If (feature 324 <= 253.5)
      Predict: 1.0
     Else (feature 324 > 253.5)
      Predict: 0.0
    Else (feature 462 > 63.0)
     Predict: 0.0
  Tree 9 (weight 1.0):
    If (feature 385 <= 4.0)
     If (feature 298 <= 224.5)
      Predict: 0.0
     Else (feature 298 > 224.5)
      Predict: 1.0
    Else (feature 385 > 4.0)
     Predict: 1.0
```
