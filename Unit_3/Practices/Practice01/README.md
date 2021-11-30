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

Import a SparkSession with the Logistic Regression library:

```r
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
```

Optional: USe the Error reporting code

```r
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```

Create a Spark Session

```r
val spark = SparkSession.builder().getOrCreate()
```

Use Spark to read the csv file Advertising

```r
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
```

Print the Schema of the DataFrame:

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

#### Display the data

Print an example row:

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

Prepare the DataFrame for Machine Learning

Do the next:

1. Rename the column "Clicked on "Ad" to "label"
2. Take the following columns as features "Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Timestamp", "Male"
3. Create a new column called "Hour" from the Timestamp containing the "Hour of the click"

```r
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
```

Import VectorAssembler and Vectors

```r
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```

Create a new VectorAssembler object called assembler for the features

```r
val assembler = (new VectorAssembler().setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male")).setOutputCol("features"))
```

Use randomSplit to create 70/30 split test and train data

```r
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
```

#### Configure a Pipeline

Pipeline amount

```r
import org.apache.spark.ml.Pipeline
```

Create a new LogisticRegression object called lr

```r
val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(assembler, lr))
```

Fit (fit) the pipeline for the training set

```r
val model = pipeline.fit(training)
```

Take the Result in the Test set with transform

```r
val results = model.transform(test)
```

#### Model Evaluation

For Metrics and Evaluation import MulticlassMetrics

```r
import org.apache.spark.mllib.evaluation.MulticlassMetrics
```

Convert test results to RDD using .as and .rdd

```r
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
```

Initialize a MulticlassMetrics object

```r
val metrics = new MulticlassMetrics(predictionAndLabels)
```

Print the Confussion Matrix

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
