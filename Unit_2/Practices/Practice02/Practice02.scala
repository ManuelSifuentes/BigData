//Practice 02

//// LINEAR REGRESSION EXERCISE

// Import linear regression

// Optional: use the following code to configure errors
import org.apache.log4j._
Logger.getLogger ("org"). SetLevel (Level.ERROR)


// Start a simple Spark session

// Use Spark for the Clean-Ecommerce csv file.

// Print the schema in the DataFrame.


// Print a sample row from the DataFrame.



//////////////////////////////////////////////////// // // ////
//// Configure the data frame for machine learning ////
//////////////////////////////////////////////////// // // ////

// Transform the data frame to take the form of
// ("tag", "features")

// Import VectorAssembler and Vectors

// Rename the Annual Amount Spent column as "label"
// Also from the data take only the numeric column
// Leave all of this as a new DataFrame called df

// Let the assembler object convert the input values ​​to a vector


// Use the VectorAssembler object to convert the input columns of the df
// to a single output column of an array named "features"
// Set the input columns from where we are supposed to read the values.
// Call this new raider.

// Use the assembler to transform our DataFrame into two columns: label and characteristics


// Create an object for a linear regression model.


// Fit the model for the data and call this model lrModel


// Print the coefficients and intercept for linear regression

// Summarize the model in the training set print the result of some metrics!
// Use the .summary method of our model to create an object
// called trainingSummary

// Shows the values ​​of the residuals, the RMSE, the MSE and also the R ^ 2.