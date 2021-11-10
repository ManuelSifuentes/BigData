import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}

// Read the titanic dataset
val spark = SparkSession.builder().getOrCreate()
val titanic_df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("Test.csv")
titanic_df.show()

// Delete unnecessary columns
val data = titanic_df.drop("PassengerId","Name","Ticket","Cabin","Embarked","Sex","SibSp","Parch")
data.show()

//We use VectorAssembler to merge the multi-column features into one vector column
val feature = new VectorAssembler().setInputCols(Array("Pclass","Age","Fare","Sex_index")).setOutputCol("features")
val feature_vector= feature.transform(data)
feature_vector.select("Survived","Pclass","Age","Fare","Sex_index","features").show()

//The data is prepared for training and the test
val Array(trainingData, testData) = feature_vector.randomSplit(Array(0.7, 0.3))

// Fits the model to the input data
val lnsvc = new LinearSVC().setLabelCol("Survived").setFeaturesCol("features")
val lnsvc_model = lnsvc.fit(trainingData)

// Prediction is made with the test data
val lnsvc_prediction = lnsvc_model.transform(testData)
lnsvc_prediction.select("prediction", "Survived", "features").show()

// Select (prediction, the value to compare and the accuracy) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("Survived").setPredictionCol("prediction").setMetricName("accuracy")

val lnsvc_accuracy = evaluator.evaluate(lnsvc_prediction)
print("Accuracy of Support Vector Machine is = " + (lnsvc_accuracy))
print(" and Test Error of Support Vector Machine = " + (1.0 - lnsvc_accuracy))


//Referens:
//https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/5722190290795989/3865595167034368/8175309257345795/latest.html
//https://techblog-history-younghunjo1.tistory.com/156