# Practice 01: Logistic Regression Project

In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
This data set contains the following features:

- 'Daily Time Spent on Site': consumer time on site in minutes
- 'Age': cutomer age in years
- 'Area Income': Avg. Income of geographical area of ​​consumer
- 'Daily Internet Usage': Avg. Minutes a day consumer is on the internet
- 'Ad Topic Line': Headline of the advertisement
- 'City': City of consumer
- 'Male': Whether or not consumer was male
- 'Country': Country of consumer
- 'Timestamp': Time at which consumer clicked on Ad or closed window
- 'Clicked on Ad': 0 or 1 indicated clicking on Ad

### Take the data

#### Import a SparkSession with the Logistic Regression library:

These are the libraries to be able to carry out subsequent actions of the practice, LogisticRegression to be able to create an object of this that will be used in the Pipeline, and SparkSession, which is what allows us to create a session in the current practice.

```r
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
```

#### Optional: Use the Error reporting code.

To control errors in running the code, the log4j library is imported.

```r
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```

#### Create a Spark Session:

Here the session is declared, having provided the respective library, it is saved in the spark variable.

```r
val spark = SparkSession.builder().getOrCreate()
```

#### Use Spark to read the csv file Advertising:

The dataset to be used is named advertising, in order to load it into memory the load method is used, which specifies the path where it is located. There are more parameters, such as "format", for the type of file format and "option", so that the first line of the data set is taken as the headers of these.

```r
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
```

#### Print the Schema of the DataFrame:

The schema shows the types of data in which the columns of the dataset come, this is important to know whether or not to perform a data cleanup.

```r
data.printSchema()
```

Result:

```r
root
 |-- Daily Time Spent on Site: double (nullable = true)
 |-- Age: integer (nullable = true)
 |-- Area Income: double (nullable = true)
 |-- Daily Internet Usage: double (nullable = true)
 |-- Ad Topic Line: string (nullable = true)
 |-- City: string (nullable = true)
 |-- Male: integer (nullable = true)
 |-- Country: string (nullable = true)
 |-- Timestamp: timestamp (nullable = true)
 |-- Clicked on Ad: integer (nullable = true)
```

### Display the data

#### Print an example row:

There are two ways to print the first row of the dataset, the simplest is to use the head method, this returns the row, but without the columns to which each one belongs, to be able to visualize these columns you must go through the set of data with the for loop.

```r
data.head(1)

val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}
```

Result:

```r
res2: Array[org.apache.spark.sql.Row] = Array([68.95,35,61833.9,256.09,Cloned 5thgeneration orchestration,Wrightburgh,0,Tunisia,2016-03-27 00:53:11.0,0])
colnames: Array[String] = Array(Daily Time Spent on Site, Age, Area Income, Daily Internet Usage, Ad Topic Line, City, Male, Country, Timestamp, Clicked on Ad)
firstrow: org.apache.spark.sql.Row = [68.95,35,61833.9,256.09,Cloned 5thgeneration orchestration,Wrightburgh,0,Tunisia,2016-03-27 00:53:11.0,0]

Example data row
Age
35

Area Income
61833.9

Daily Internet Usage
256.09

Ad Topic Line
Cloned 5thgeneration orchestration

City
Wrightburgh

Male
0

Country
Tunisia

Timestamp
2016-03-27 00:53:11.0

Clicked on Ad
0
```

Prepare the DataFrame for Machine Learning:

Do the next:

1. Rename the column "Clicked on "Ad" to "label"
2. Take the following columns as features "Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Timestamp", "Male"
3. Create a new column called "Hour" from the Timestamp containing the "Hour of the click"

The "withColumn" method is used to transform data types, so by specifying the column, the required data type is changed, and the name change of a column is done with the "as" method.

```r
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
```

#### Import VectorAssembler and Vectors:

These libraries are the ones that help to be able to make the transformation of data to the characteristics with which Logistic Regression is going to work.

```r
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```

#### Create a new VectorAssembler object called assembler for the features:

In this step, all the columns are grouped into the same one, which bears the name of features, this result is stored in the “assembler” variable.

```r
val assembler = (new VectorAssembler().setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male")).setOutputCol("features"))
```

#### Use randomSplit to create 70/30 split test and train data:

Two different vectors must be created, one for the training data, which is the one with which the regression algorithm will be adjusted, and another which is the test algorithm, which is the one that will show the effectiveness of the success achieved in linear regression, all this while maintaining a degree of randomness with the seed.

```r
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
```

### Configure a Pipeline

#### Pipeline amount

The Pipeline library must be imported, which is the one that will help to gather in a single variable, the steps through which the data must pass.

```r
import org.apache.spark.ml.Pipeline
```

#### Create a new LogisticRegression object called lr

The LogisticRegression object is created, and declaring a new Pipeline object, the “assembler” and “lr” are passed as parameters, which will be the steps through which the data will pass to perform the linear regression. At the end the training data is passed to fit the linear regression to the data.

```r
val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(assembler, lr))
```

#### Fit (fit) the pipeline for the training set:

Once the model has been trained, the test data is passed, so that the results are predicted with the linear regression.

```r
val model = pipeline.fit(training)
```

Take the Result in the Test set with transform

```r
val results = model.transform(test)
```

### Model Evaluation

#### For Metrics and Evaluation import MulticlassMetrics:

Having already obtained the results, they must be evaluated to verify the precision of the prediction, and for that MulticlassMetrics is imported.

```r
import org.apache.spark.mllib.evaluation.MulticlassMetrics
```

#### Convert test results to RDD using .as and .rdd:

The results of the prediction and "label" columns are converted as type double, and these are stored in the variable "predictionAndLabels".

```r
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
```

#### Initialize a MulticlassMetrics object:

Once the data has been transformed, it is evaluated with the help of the MulticlassMetrics object and stored in the metrics variable.

```r
val metrics = new MulticlassMetrics(predictionAndLabels)
```

#### Print the Confussion Matrix:

The confusion matrix is used to evaluate false positives and negatives, with true positives and negatives, and these results are displayed, as well as the precision obtained from the linear regression using the accuracy method.

```r
println("Confusion matrix:")
println(metrics.confusionMatrix)

metrics.accuracy
```

Result:

```r
Confusion matrix:
146.0  7.0
1.0    161.0
res8: Double = 0.9746031746031746
```
