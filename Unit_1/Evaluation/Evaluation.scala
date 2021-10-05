// Answer the following questions with Spark DataFrames and Scala using the "CSV"
// Netflix_2011_2016.csv found in the spark-dataframes folder.

// 1. Start a simple Spark session.
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

// 2. Upload Netflix Stock CSV file, have Spark infer the data types.
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv") 

// 3. What are the names of the columns?
df.columns

// 4. What is the scheme like?
df.printSchema()

// 5. Print the first 5 columns.
for(row <- df.head(5)){
    println(row)
}

// 6. Use describe () to learn about the DataFrame.
df.describe().show()

// 7. Create a new dataframe with a new column called “HV Ratio” which is the relationship between the price in the “High” column versus the “Volume” column of shares traded for a day. Hint is an operation
val df2 = df.withColumn("HV Ratio",df("High")/df("Volume"))

// 8. What day had the highest peak in the “Open” column?
df.orderBy($"Open".desc).show(1)

// 9. What is the meaning of the Close column “Close” in the context of financial information, explain it, there is nothing to code?
// It is the price at which the stock sales ended on the respective day.

// 10. What is the maximum and minimum in the “Volume” column?
df.select(max("Volume")).show()
df.select(min("Volume")).show()

//11. With Scala / Spark $ Syntax answer the following:

// a. How many days was the “Close” column less than $ 600?
df.filter($"Close"<600).count()
// b. What percentage of the time was the “High” column greater than $ 500?
(df.filter($"High">500).count()*1.0/df.count())*100
// c. What is the Pearson correlation between column "High" and column "Volume"?
df.select(corr("High","Volume")).show()
// d. What is the maximum in the “High” column per year?
val df_year = df.withColumn("Year",year(df("Date")))
val df_max = df_year.select($"Year",$"High").groupBy("Year").max()
df_max.select($"Year",$"max(High)").show()
// e. What is the average “Close” column for each calendar month?
val df_month = df.withColumn("Month",month(df("Date")))
val month_avgs = df_month.select($"Month",$"Close").groupBy("Month").mean()
month_avgs.select($"Month",$"avg(Close)").show()
