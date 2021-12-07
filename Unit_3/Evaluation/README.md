# Evaluation #03

The goal of this practical test is to try to group customers from specific regions of a wholesaler. This based on the sales of some product categories.

1. Import a simple Spark session.

```r
import org.apache.spark.sql.SparkSession
```

2. Use the lines of code to minimize errors

```r
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```

3. Create an instance of the Spark session

```r
val spark = SparkSession.builder().getOrCreate()
```

4. Import the Kmeans library for the clustering algorithm.

```r
import org.apache.spark.ml.clustering.KMeans
```

5. Load the Wholesale Customers Data dataset

```r
val dataset = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesale customers data.csv")
```

6. Select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data

```r
val feature_data = (dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen"))
```

7. Import Vector Assembler and Vector

```r
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```

8. Create a new Vector Assembler object for the feature columns as a input set, remembering that there are no labels

```r
val assembler = (new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features"))
```

9. Use the assembler object to transform feature_data

```r
val training_data = assembler.transform(feature_data).select($"features")
```

10. Create a Kmeans model with K = 3

```r
val kmeans = new KMeans().setK(3).setSeed(1L)
```

- Result

```r
WSSSE: Double = 8.095172370767671E10
Within Set Sum of Squared Errors = 8.095172370767671E10
```

10.1 10.1 Fit that model to the training_data

```r
val model = kmeans.fit(training_data)
```

11. Evaluate the groups using Within Set Sum of Squared Errors WSSSE and print the centroids.

```r
val WSSSE = model.computeCost(training_data)
println(s"Within Set Sum of Squared Errors = $WSSSE")
```

11.1 Shows the result

```r
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
```

- Result

```r
Cluster Centers: 
[7993.574780058651,4196.803519061584,5837.4926686217,2546.624633431085,2016.2873900293255,1151.4193548387098]
[9928.18918918919,21513.081081081084,30993.486486486487,2960.4324324324325,13996.594594594595,3772.3243243243246]
```



