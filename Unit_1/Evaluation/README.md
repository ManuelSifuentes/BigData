# Evaluation #01

#### a. How many days was the “Close” column less than $ 600?

The spark method is used, which allows filtering data from a column with a specific condition to see those less than 600.

```r
df.filter($"Close"<600).count()

res1: Long = 1218

```

#### b. What percentage of the time was the “High” column greater than $ 500?

The same as the previous one, but now first the ones greater than 500 are searched, and from this it is multiplied by 1 and divided by the total of records in data frame and multiplied by 100, to obtain the required percentage.

```r
(df.filter($"High">500).count()*1.0/df.count())*100

res2: Double = 4.924543288324067
```

#### c. What is the Pearson correlation between column "High" and column "Volume"?

We use the spark method of corr to obtain the Pearson correlation.

```r
df.select(corr("High","Volume")).show()

+--------------------+
|  corr(High, Volume)|
+--------------------+
|-0.20960233287942157|
+--------------------+

```

#### d. What is the maximum in the “High” column per year?

A new dataframe is obtained from the date column and then the required year is obtained. Once again another dataframe is created that has the highest values of each year, ending with showing the maximum value of each year.

```r
val df_year = df.withColumn("Year",year(df("Date")))
val df_max = df_year.select($"Year",$"High").groupBy("Year").max()
df_max.select($"Year",$"max(High)").show()

+----+------------------+
|Year|         max(High)|
+----+------------------+
|2015|        716.159996|
|2013|        389.159988|
|2014|        489.290024|
|2012|        133.429996|
|2016|129.28999299999998|

```
