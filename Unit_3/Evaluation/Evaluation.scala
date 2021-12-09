// Evaluation #03

// The goal of this practical test is to try to group customers from specific regions of a wholesaler. This based on the sales of some product categories.

// 1. Import a simple Spark session.
import org.apache.spark.sql.SparkSession

// 2. Use the lines of code to minimize errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// 3. Create an instance of the Spark session
val spark = SparkSession.builder().getOrCreate()

// 4. Import the Kmeans library for the clustering algorithm.
import org.apache.spark.ml.clustering.KMeans

// 5. Load the Wholesale Customers Data dataset
val dataset = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesale customers data.csv")

// 6. Select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data
val feature_data = (dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen"))

// 7. Import Vector Assembler and Vector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// 8. Create a new Vector Assembler object for the feature columns as a input set, remembering that there are no labels
val assembler = (new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features"))

// 9. Use the assembler object to transform feature_data
val training_data = assembler.transform(feature_data).select($"features")

// 10. Create a Kmeans model with K = 3
val kmeans = new KMeans().setK(3).setSeed(1L)

// 10.1 Fit that model to the training_data
val model = kmeans.fit(training_data)

// 11.Evaluate the groups using Within Set Sum of Squared Errors WSSSE and print the centroids.
val WSSSE = model.computeCost(training_data)
println(s"Within Set Sum of Squared Errors = $WSSSE")

//11.1 Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)