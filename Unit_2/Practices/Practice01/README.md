# Practice #01

First, the libraries to be used must be imported, they contain the methods that will later allow to carry out all the procedures and actions to be able to perform the classification by the Decision tree method.

```r
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
```

Afterwards, the data set to be analyzed must be loaded, so the format is specified, in this case "libsvm" and the load method specifies the path where the file is hosted, since it is in the same directory , only the name sample_libsvm_data.txt is specified.

```r
val data = spark.read.format("libsvm").load("../sample_libsvm_data.txt")
```

Once the data is loaded, the next step is to declare the following indexing vectors, which do not modify the current data, they are only indexed, this allows optimizing the procedures to be carried out.

In the setInputCol method the name of the data column is specified, and in setOutputCol how the indexed column will be temporarily called.

Index labels, adding metadata to the label column.
Fit on whole dataset to include all labels in index.

```r
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
```

Automatically identify categorical features, and index them.

```r
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data) // features with > 4 distinct values are treated as continuous..fit(data)
```

Already having the two vectors, now the data must be separated between the test and training data (test and training), for this another arrangement is declared and with the help of the randomSplit method, the data is separated, specifying that 70% of the data will go for training, and the rest for testing.

```r
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
```

In this step, the model object is generated, specifying how the column that will carry the label that will be used for the predictions will be called, and then the characteristics that will be used to predict.

```r
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
```

The new IndexToString object will take care of converting existing columns to indices, all storing in the labelConverter variable.

```r
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
```

Inside the Pipeline is where everything created so far is related, it is specified that the label indexer (labelIndexer), characteristics (featureIndexer), the model (dt) and finally the converter (labelConverter) will be used, which is the one that removes the indexes that were previously created.

```r
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
```

The training stage follows, here the created pipeline is used, and with the help of the fit method, the data source to be used is specified. This data will go through the pipeline stages, which were declared in the previous step.

```r
val model = pipeline.fit(trainingData)
```

Once the training is done, you already have data to be able to make predictions, so the declared variable is specified to store the training data, and with the transform method, the previously separated data is passed as a parameter to make the predictions. These will go through the same steps as the previous ones and in the end the predictions will return.

```r
val predictions = model.transform(testData)
```

Here the predictions data is selected, and the parameters are the name of the columns contained in the predictions variable, which are the label that was predicted, the real label and the characteristics and at the end, with the show method it is specified that only the first 5 rows are taken.

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
|           0.0|  0.0|(692,[122,123,124...|
|           0.0|  0.0|(692,[123,124,125...|
|           0.0|  0.0|(692,[123,124,125...|
|           0.0|  0.0|(692,[124,125,126...|
+--------------+-----+--------------------+
only showing top 5 rows
```

The evaluator, which is what is declared in the following code, provides the declared metrics, which is through the setMetricName method, specifying that the accuracy is wanted, then the error obtained by the model is printed, subtracting completely (1) , the percentage of accuracy.

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

Finally, the visualization of the decision tree model is explicitly shown, specifying that only 2 stages are carried out and also with the help of the asInstanceOf method, passing as a parameter the type of model to be carried out, in this case DecisionTreeClassificationModel.

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
