# Practice #01

We declare the libraries to use to carry out the example with decision tree classification.

```r
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
```

Load the data stored in LIBSVM format as a DataFrame.

```r
val data = spark.read.format("libsvm").load("../sample_libsvm_data.txt")
```

Index labels, adding metadata to the label column.
Fit on whole dataset to include all labels in index.

```r
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
```

Automatically identify categorical features, and index them.

```r
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data) // features with > 4 distinct values are treated as continuous..fit(data)
```

Split the data into training and test sets (30% held out for testing and the other 70% to make the model).

```r
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
```

Train a DecisionTree model.

```r
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
```

Convert indexed labels back to original labels defining as label the "indexLabe" and the features the column "indexedFeatures".

```r
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
```

Chain indexers and tree in a Pipeline.

```r
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
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

```r
predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[98,99,100,1...|
|           0.0|  0.0|(692,[122,123,124...|
|           0.0|  0.0|(692,[123,124,125...|
|           0.0|  0.0|(692,[123,124,125...|
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
accuracy: Double = 0.9722222222222222
Test Error = 0.02777777777777779
```

We finish by showing the conditional logic that represents the elements of the decision tree.

```r
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
```

Result:

```r
Learned classification tree model:
 DecisionTreeClassificationModel (uid=dtc_a688c0785eea) of depth 2 with 5 nodes
  If (feature 406 <= 22.0)
   If (feature 99 in {2.0})
    Predict: 0.0
   Else (feature 99 not in {2.0})
    Predict: 1.0
  Else (feature 406 > 22.0)
   Predict: 0.0
```
