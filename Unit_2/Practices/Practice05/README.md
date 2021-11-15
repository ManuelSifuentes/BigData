# Practice #05

Libraries:

In order to carry out this type of classification, the respective libraries for the creation of the object must be imported, so the MultilayerPerceptronClassifier and MulticlassClassificationEvaluator libraries are imported.

```r
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
```

Load the data stored in LIBSVM format as a DataFrame.
The dataset must be loaded, and this is done with the read method, first specifying the format in which the data comes, and then, with the load method, the directory where the file is located is placed, the data is saved in the variable data.

```r
val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")
```

Split the data into train and test.
Having already loaded the data set in memory, they must be divided randomly to be able, first, to train the model, and later to do the tests to see what results it yields after having performed the classification procedures. The random split method is used, specifying that 60% will go to training and the rest to test. The seed parameter is used to indicate a pseudo-randomization pattern so that, each time the code is executed, different results are obtained.

```r
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
```

Specify layers for the neural network: input layer of size 4 (features), two intermediate of size 5 and 4 and output of size 3 (classes).
Here the fields for the neural network are specified, which are the input, output and intermediate fields, so they are declared in this way, with the help of a vector.

```r
val layers = Array[Int](4, 5, 4, 3)
```

Create the trainer and set its parameters.
Here you create the MultilayerPerceptronClassifier object, and with the help of different methods you specify the characteristics you want it to have. The layers declared in the previous step are passed as a parameter with the help of the setLayers method, with setBlockSize the number of bits to use is defined, setSeed to grant randomness and finally setMaxIter is found, which is the maximum number of iterations. To make.

```r
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
```

Train the model.
Here the model is adjusted to the data, it is trained to give better results for the subsequent prediction, what is saved in the trainer variable is what will be executed in the data set that is passed as a parameter, which results in the model of MultilayerPerceptronClassifier.

```r
val model = trainer.fit(train)
```

Compute accuracy on the test set.
Being the final part of the process, the data set saved for the test is passed as a parameter in the transform method, these will be executed in the model and will return the result obtained after going through the MultilayerPerceptronClassifier process. Once the data has been saved in the result variable, it is accessed with the help of the select method, specifying the columns to be displayed. In the evaluator variable the percentage of precision that the classification method had will be saved.

```r
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
```

At the end, the precision obtained by the MultilayerPerceptron classification process is displayed.

```r
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```
