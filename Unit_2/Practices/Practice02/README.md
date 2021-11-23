# Practice #02

LINEAR REGRESSION EXERCISE

Import linear regression.

In order to create the linear regression object, the respective library must first be imported, so this is where the library is imported.

```r
import org.apache.spark.ml.regression.LinearRegression
```

Optional: use the following code to configure errors.

The following library is called Log4j and it is used to write log messages, whose purpose is to record a certain action carried out during the execution of the code. The next line shows the registry declaration without doing any configuration.

```r
import org.apache.log4j._
Logger.getLogger ("org"). SetLevel (Level.ERROR)
```

Start a simple Spark session.

The library is imported in order to establish a session in spark, and then it is declared by saving it in the variable named spark.

```r
import org.apache.spark.sql.SparkSession
var spark = SparkSession.builder().getOrCreate()
```

Use Spark for the Clean-Ecommerce csv file.

After the initial preliminary configuration, the data to be used is loaded, this is done with the read method, adding more parameters through the option method, specifying that the headers are read, the type of format that will be csv and the path where the data file will be read.

```r
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Clean-Ecommerce.csv")
```

Print the schema in the DataFrame and print a sample row from the DataFrame.

The scheme allows you to see the type of data that each column of the CSV has, and what is done next is to print the first line of the dataset, as well as the name of its columns, going through it with the help of a loop for.

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

Import VectorAssembler and Vectors.

In order to create a VectorAssembler object, the library must first be imported, so the libraries are imported here in order to carry out this action.

```r
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```

Rename the Annual Amount Spent column as "label", also from the data take only the numeric column and leave all of this as a new DataFrame called df.

With the help of the select method, it is specified that the Yearly Amount Spent column will be taken and renamed by the name label, also including the rest of the columns and saving it in a variable called df.

```r
val df = data.select(data("Yearly Amount Spent").as("label"),
$"Avg Session Length",$"Time on App",$"Time on Website",$"Length of Membership")
```

Let the assembler object convert the input values ​​to a vector.

Use the VectorAssembler object to convert the input columns of the df to a single output column of an array named "features". Set the input columns from where we are supposed to read the values.
Call this new raider.

A new VectorAssembler object is declared, converting all the columns of the dataset to a single output column called features, all of this will be saved in the assembler variable. The newly created variable is used in conjunction with the transform method, in this the set of data saved in df is passed as a parameter to transform it to have two columns, which are label and features.

```r
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length","Time on App","Time on Website","Length of Membership")).setOutputCol("features")
```

Use the assembler to transform our DataFrame into two columns: label and characteristics.

```r
val output = assembler.transform(df).select($"label",$"features")
```

Create an object for a linear regression model.

What has been done so far is the cleaning and transformation of the data, now it is time to create the LinearRegression object, and this is saved in the variable lr.

```r
val lr = new LinearRegression()
```

Fit the model for the data and call this model lrModel.

Once the LinearRegression object has been created, the model must be adjusted to the data contained in the dataset stored in the output variable. This is where the diagonal is generated, which would be what it would look like graphically in the linear regression.

```r
val lrModel = lr.fit(output)
```

Print the coefficients and intercept for linear regression.

With the help of println, the coefficients contained in the variable lrModel are printed.

```r
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
```

Summarize the model in the training set print the result of some metrics!
Use the .summary method of our model to create an object, called trainingSummary.

Here the summary of the trained model is made, using the variable lrModel, this with the help of the summary method.

```r
val trainingSummary = lrModel.summary
```

Shows the values ​​of the residuals, the RMSE, the MSE and also the R ^ 2.

Finally, the results obtained using the linear regression method are shown. The residuals is the difference that exists between the predicted value and against what is contained in the current label column, the RMSE is the distance between the points and the r2 is how variant the model is

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
