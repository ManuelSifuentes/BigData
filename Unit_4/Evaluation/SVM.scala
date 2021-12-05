/*Importamos las librerias necesarias con las que vamos a trabajar*/
for(i <- 0 to 30)
{
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j._

/*Quita los warnings*/
Logger.getLogger("org").setLevel(Level.ERROR)

/*Creamos una sesion de spark y cargamos los datos del CSV en un datraframe*/
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank.csv")

/*Desblegamos los tipos de datos.*/
// df.printSchema()
// df.show(5)

/*Cambiamos la columna y por una con datos binarios.*/
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))

/*Desplegamos la nueva columna*/
// newcolumn.show(5)

/*Generamos la tabla features*/
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)

/*Mostramos la nueva columna*/
// fea.show(1)

/*Cambiamos la columna y a la columna label*/
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
// feat.show(5)

/*SVM: Se requiere cambiar los valores categoricos numericos a 0 y 1 respectivamente*/
val c1 = feat.withColumn("label",when(col("label").equalTo("1"),0).otherwise(col("label")))
val c2 = c1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val c3 = c2.withColumn("label",'label.cast("Int"))

// c3.show(5)

//The data is prepared for training and the test
val Array(trainingData, testData) = c3.randomSplit(Array(0.7, 0.3))

//Instancia del modelo
val linsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// //Usando como valores predominantes el label y features
// val linsvc = new LinearSVC().setLabelCol("label").setFeaturesCol("features")

/* Fit del modelo*/
val linsvcModel = linsvc.fit(trainingData)

//Transformacion del modelo con los datos de test
val lnsvc_prediction = linsvcModel.transform(testData)
// lnsvc_prediction.select("prediction", "label", "features").show(10)

/*Imprimimos linea de intercepcion*/
// println(s"Coefficients: ${linsvcModel.coefficients} Intercept: ${linsvcModel.intercept}")

//Mostrar Accuracy
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val lnsvc_accuracy = evaluator.evaluate(lnsvc_prediction)
print("Accuracy of Support Vector Machine is = " + (lnsvc_accuracy))
// print(" and Test Error of Support Vector Machine = " + (1.0 - lnsvc_accuracy))
}