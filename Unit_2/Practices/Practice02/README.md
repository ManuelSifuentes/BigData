# Practice #02

LINEAR REGRESSION EXERCISE

Import linear regression

```r
import org.apache.spark.ml.regression.LinearRegression
```

Start a simple Spark session

```r
import org.apache.spark.sql.SparkSession
var spark = SparkSession.builder().getOrCreate()
```

Use Spark for the Clean-Ecommerce csv file.

```r
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Clean-Ecommerce.csv")
```

Print the schema in the DataFrame and print a sample row from the DataFrame.

```r
data.printSchema()
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}
```

Result:

```r
data: org.apache.spark.sql.DataFrame = [Email: string, Avatar: string ... 5 more fields]
root
 |-- Email: string (nullable = true)
 |-- Avatar: string (nullable = true)
 |-- Avg Session Length: double (nullable = true)
 |-- Time on App: double (nullable = true)
 |-- Time on Website: double (nullable = true)
 |-- Length of Membership: double (nullable = true)
 |-- Yearly Amount Spent: double (nullable = true)

colnames: Array[String] = Array(Email, Avatar, Avg Session Length, Time on App, Time on Website, Length of Membership, Yearly Amount Spent)
firstrow: org.apache.spark.sql.Row = [mstephenson@fernandez.com,Violet,34.49726772511229,12.65565114916675,39.57766801952616,4.0826206329529615,587.9510539684005]

Example Data Row
Avatar
Violet

Avg Session Length
34.49726772511229

Time on App
12.65565114916675

Time on Website
39.57766801952616

Length of Membership
4.0826206329529615

Yearly Amount Spent
587.9510539684005
```

### Configure the data frame for machine learning

Transform the data frame to take the form of ("tag", "features")
Import VectorAssembler and Vectors

```r
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```

Rename the Annual Amount Spent column as "label", also from the data take only the numeric column and leave all of this as a new DataFrame called df

```r
val df = data.select(data("Yearly Amount Spent").as("label"),
$"Avg Session Length",$"Time on App",$"Time on Website",$"Length of Membership")
```

Let the assembler object convert the input values ​​to a vector

Use the VectorAssembler object to convert the input columns of the df
to a single output column of an array named "features"
Set the input columns from where we are supposed to read the values.
Call this new raider.

```r
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length","Time on App","Time on Website","Length of Membership")).setOutputCol("features")
```

Use the assembler to transform our DataFrame into two columns: label and characteristics

```r
val output = assembler.transform(df).select($"label",$"features")
```

Create an object for a linear regression model.

```r
val lr = new LinearRegression()
```

Fit the model for the data and call this model lrModel

```r
val lrModel = lr.fit(output)
```

Print the coefficients and intercept for linear regression

```r
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
```

Summarize the model in the training set print the result of some metrics!
Use the .summary method of our model to create an object, called trainingSummary

```r
val trainingSummary = lrModel.summary
```

Shows the values ​​of the residuals, the RMSE, the MSE and also the R ^ 2.

```r
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")
```

Result:

```r
Coefficients: [25.734271084670716,38.709153810828816,0.43673883558514964,61.57732375487594] Intercept: -1051.5942552990748
trainingSummary: org.apache.spark.ml.regression.LinearRegressionTrainingSummary = org.apache.spark.ml.regression.LinearRegressionTrainingSummary@39dc0a19
+-------------------+
|          residuals|
+-------------------+
| -6.788234090018818|
| 11.841128565326073|
| -17.65262700858966|
| 11.454889631178617|
| 7.7833824373080915|
|-1.8347332184773677|
|  4.620232401352382|
| -8.526545950978175|
| 11.012210896516763|
|-13.828032682158891|
| -16.04456458615175|
|  8.786634365463442|
| 10.425717191807507|
| 12.161293785003522|
|  9.989313714461446|
| 10.626662732649379|
|  20.15641408428496|
|-3.7708446586326545|
| -4.129505481591934|
|  9.206694655890487|
+-------------------+
only showing top 20 rows

RMSE: 9.923256785022229
MSE: 98.47102522148971
r2: 0.9843155370226727
```
