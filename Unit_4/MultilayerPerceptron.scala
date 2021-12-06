//Practice MultilayerPerceptron
for(i <- 0 to 30)
{
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline
import org.apache.log4j._

//Quita los warnings
Logger.getLogger("org").setLevel(Level.ERROR)

//Creamos una sesion de spark y cargamos los datos del CSV en un datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank.csv")

//Desblegamos los tipos de datos
df.printSchema()
//Mostramos primer renglon
df.show(1)

//Cambiamos la columna "y" por una con datos binarios
val df1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val df2 = df1.withColumn("y",when(col("y").equalTo("no"),0).otherwise(col("y")))
val newcolumn = df2.withColumn("y",'y.cast("Int"))

//Desplegamos la nueva columna
newcolumn.show(1)

//Generamos el campo features con VectoAssembler
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val newDF = assembler.transform(newcolumn)

//Mostramos el campo de features
newDF.show(1)

//Modificamos el campo "y" por label
val cambio = newDF.withColumnRenamed("y", "label")
//Seleccionamos un nuevo df con los campor de 'label' y 'features'
val finalDF = cambio.select("label","features")
finalDF.show(1)

//Cambiamos el label principal con datos caterogicos de string a un Index
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(finalDF)
//val labelIndexer2 = new StringIndexer().setInputCol("job").setOutputCol("indexedLabelJob").fit(newcolumn)
//Mostramos la categoria de los datos
println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

//Nueva variablea para definirle un index a los vectores del campo "features"
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(finalDF)

// Split the data into train and test
val splits = c3.randomSplit(Array(0.7, 0.3), seed = 1234L)
val trainingData = splits(0)
val testData = splits(1)

//Specify layers for the neural network:
//Input layer of size 5 (features), two intermediate of size 6 and 5 and output of size 2 (classes)
val layers = Array[Int](5, 6, 5, 2)

//Creamos instancia del metodo de la libreria de clasificacion con el campo de entrada "indexedLabel" y los caracteristicas del campo "indexedFeatures"
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234L).setMaxIter(100)

//Para efectos de demostracion se invierte la prediccion a tipo string del label
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

//Juntamos los datos creados para tener un nuevo df con los nuevos campos
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

//Cremos modelo con los datos de entramiento
val model = pipeline.fit(trainingData)

//Generamos la prediccion con los datos de prueba
val prediction = model.transform(testData)
prediction.select("prediction", "label", "features").show(5)

//Finalizamos realizado una prueba para conocer el accuracy del modelo y conocer su eficiencia.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(prediction)

//Resultado
print("Accuracy of Support Vector Machine is = " + (accuracy))

}

//References
//https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3741049972324885/1019862370390522/4413065072037724/latest.html