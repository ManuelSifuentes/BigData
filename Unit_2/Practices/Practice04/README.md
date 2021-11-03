# Practice 04

Libraries:

```r
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
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

Train a GBT model.

```r
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")
```

Convert indexed labels back to original labels.

```r
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
```

Chain indexers and GBT in a Pipeline.

```r
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
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
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[98,99,100,1...|
|           0.0|  0.0|(692,[100,101,102...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[125,126,127...|
|           0.0|  0.0|(692,[126,127,128...|
+--------------+-----+--------------------+
only showing top 5 rows
```

Select (prediction, true label) and compute test error.

```r
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")
```

Result:

```r
accuracy: Double = 1.0
Test Error = 0.0
```

We finish by showing the conditional logic that represents the elements of the trees

```r
val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
```

Result:

```r
Learned classification GBT model:
 GBTClassificationModel (uid=gbtc_b5eac4259fd7) with 10 trees
  Tree 0 (weight 1.0):
    If (feature 434 <= 88.5)
     If (feature 99 in {2.0})
      Predict: -1.0
     Else (feature 99 not in {2.0})
      Predict: 1.0
    Else (feature 434 > 88.5)
     Predict: -1.0
  Tree 1 (weight 0.1):
    If (feature 490 <= 29.0)
     If (feature 568 <= 253.5)
      If (feature 154 <= 57.5)
       Predict: 0.4768116880884702
      Else (feature 154 > 57.5)
       Predict: 0.47681168808847024
     Else (feature 568 > 253.5)
      Predict: -0.4768116880884694
    Else (feature 490 > 29.0)
     If (feature 323 <= 112.5)
      Predict: -0.47681168808847024
     Else (feature 323 > 112.5)
      Predict: -0.47681168808847035
  Tree 2 (weight 0.1):
    If (feature 434 <= 88.5)
     If (feature 626 <= 2.5)
      Predict: -0.4381935810427206
     Else (feature 626 > 2.5)
      Predict: 0.43819358104272055
    Else (feature 434 > 88.5)
     If (feature 323 <= 13.0)
      Predict: -0.4381935810427206
     Else (feature 323 > 13.0)
      Predict: -0.43819358104272066
  Tree 3 (weight 0.1):
    If (feature 462 <= 62.5)
     If (feature 213 <= 85.5)
      Predict: -0.4051496802845983
     Else (feature 213 > 85.5)
      Predict: 0.40514968028459825
    Else (feature 462 > 62.5)
     If (feature 432 <= 4.0)
      Predict: -0.40514968028459825
     Else (feature 432 > 4.0)
      Predict: -0.4051496802845983
  Tree 4 (weight 0.1):
    If (feature 490 <= 29.0)
     If (feature 548 <= 253.5)
      If (feature 578 <= 56.5)
       Predict: 0.3765841318352991
      Else (feature 578 > 56.5)
       Predict: 0.3765841318352993
     Else (feature 548 > 253.5)
      Predict: -0.3765841318352994
    Else (feature 490 > 29.0)
     If (feature 432 <= 58.5)
      If (feature 377 <= 88.5)
       Predict: -0.3765841318352991
      Else (feature 377 > 88.5)
       Predict: -0.37658413183529926
     Else (feature 432 > 58.5)
      Predict: -0.3765841318352994
  Tree 5 (weight 0.1):
    If (feature 434 <= 88.5)
     If (feature 99 in {2.0})
      Predict: -0.35166478958101005
     Else (feature 99 not in {2.0})
      If (feature 156 <= 9.0)
       Predict: 0.35166478958101005
      Else (feature 156 > 9.0)
       Predict: 0.3516647895810101
    Else (feature 434 > 88.5)
     If (feature 351 <= 176.5)
      If (feature 181 <= 2.0)
       Predict: -0.35166478958101005
      Else (feature 181 > 2.0)
       Predict: -0.3516647895810101
     Else (feature 351 > 176.5)
      Predict: -0.35166478958101005
  Tree 6 (weight 0.1):
    If (feature 490 <= 29.0)
     If (feature 548 <= 253.5)
      If (feature 233 <= 198.5)
       Predict: 0.32974984655529926
      Else (feature 233 > 198.5)
       Predict: 0.3297498465552994
     Else (feature 548 > 253.5)
      Predict: -0.32974984655530015
    Else (feature 490 > 29.0)
     Predict: -0.32974984655529915
  Tree 7 (weight 0.1):
    If (feature 434 <= 88.5)
     If (feature 239 <= 253.5)
      If (feature 211 <= 160.0)
       Predict: 0.3103372455197956
      Else (feature 211 > 160.0)
       If (feature 456 <= 31.5)
        Predict: 0.3103372455197956
       Else (feature 456 > 31.5)
        Predict: 0.3103372455197957
     Else (feature 239 > 253.5)
      Predict: -0.31033724551979525
    Else (feature 434 > 88.5)
     If (feature 294 <= 99.5)
      If (feature 463 <= 53.5)
       Predict: -0.3103372455197956
      Else (feature 463 > 53.5)
       Predict: -0.3103372455197957
     Else (feature 294 > 99.5)
      If (feature 323 <= 112.5)
       Predict: -0.3103372455197956
      Else (feature 323 > 112.5)
       If (feature 211 <= 72.0)
        Predict: -0.3103372455197956
       Else (feature 211 > 72.0)
        Predict: -0.3103372455197957
  Tree 8 (weight 0.1):
    If (feature 434 <= 88.5)
     If (feature 243 <= 4.0)
      Predict: -0.2930291649125432
     Else (feature 243 > 4.0)
      If (feature 210 <= 221.0)
       Predict: 0.2930291649125433
      Else (feature 210 > 221.0)
       Predict: 0.2930291649125434
    Else (feature 434 > 88.5)
     If (feature 462 <= 136.0)
      Predict: -0.2930291649125433
     Else (feature 462 > 136.0)
      Predict: -0.2930291649125434
  Tree 9 (weight 0.1):
    If (feature 490 <= 29.0)
     If (feature 239 <= 253.5)
      If (feature 577 <= 10.0)
       Predict: 0.27750666438358246
      Else (feature 577 > 10.0)
       Predict: 0.27750666438358257
     Else (feature 239 > 253.5)
      Predict: -0.2775066643835826
    Else (feature 490 > 29.0)
     If (feature 377 <= 132.5)
      Predict: -0.2775066643835825
     Else (feature 377 > 132.5)
      Predict: -0.27750666438358257
```
