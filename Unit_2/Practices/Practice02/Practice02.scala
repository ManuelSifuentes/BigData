//Practice 02

//// LINEAR REGRESSION EXERCISE

// Import linear regression
import org.apache.spark.ml.regression.LinearRegression

// Optional: use the following code to configure errors
import org.apache.log4j._
Logger.getLogger ("org"). SetLevel (Level.ERROR)


// Start a simple Spark session
import org.apache.spark.sql.SparkSession
var spark = SparkSession.builder().getOrCreate()

// Use Spark for the Clean-Ecommerce csv file.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Clean-Ecommerce.csv")

// Print the schema in the DataFrame.
// Print a sample row from the DataFrame.
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


///////////////////////////////////////////////////////
//// Configure the data frame for machine learning ////
/////////////////////////////////////////////////////// 

// Transform the data frame to take the form of
// ("label", "features")

// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Rename the Annual Amount Spent column as "label"
// Also from the data take only the numeric column
// Leave all of this as a new DataFrame called df
val df = data.select(data("Yearly Amount Spent").as("label"),
$"Avg Session Length",$"Time on App",$"Time on Website",$"Length of Membership")

// Let the assembler object convert the input values ​​to a vector


// Use the VectorAssembler object to convert the input columns of the df
// to a single output column of an array named "features"
// Set the input columns from where we are supposed to read the values.
// Call this new raider.
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length","Time on App","Time on Website","Length of Membership")).setOutputCol("features")
// Use the assembler to transform our DataFrame into two columns: label and characteristics
val output = assembler.transform(df).select($"label",$"features")

// Create an object for a linear regression model.
val lr = new LinearRegression()

// Fit the model for the data and call this model lrModel
val lrModel = lr.fit(output)

// Print the coefficients and intercept for linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model in the training set print the result of some metrics!
// Use the .summary method of our model to create an object
// called trainingSummary
val trainingSummary = lrModel.summary

// Shows the values ​​of the residuals, the RMSE, the MSE and also the R ^ 2.
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")