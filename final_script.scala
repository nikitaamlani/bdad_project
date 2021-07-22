import org.apache.spark.ml.feature.Imputer
import org.apache.spark.sql.types._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.mllib.util.MLUtils
import sys.process._  
import scala.language.postfixOps
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator  
import org.apache.spark.mllib.evaluation.MulticlassMetrics  
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics  
import org.apache.spark.ml.classification.RandomForestClassifier  
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, CrossValidator}  
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoderEstimator}  
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

import org.apache.spark.ml.Pipeline  
import org.apache.log4j._  
Logger.getLogger("org").setLevel(Level.ERROR)  
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

import spark.implicits._


import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

					
val creditDf = spark.read.option("header","true").
    option("inferSchema","true").
    format("csv").
    load("/user/xw2447/project/MyData_6.csv")
	
// val cols = Array("ft1", "ft2", "ft3", "ft4", "ft5", "ft6")
					

val cols = Array("sound","s","number","lat","l","long","weight","wing","color","temp_min","temp_max","beak")

val assembler = new VectorAssembler().setHandleInvalid("skip").setInputCols(cols).setOutputCol("features")
val featureDf = assembler.transform(creditDf)


// StringIndexer define new 'label' column with 'result' column
// val indexer = new StringIndexer().setInputCol("creditability").setOutputCol("label")
val indexer = new StringIndexer().setInputCol("variety").setOutputCol("label")

val labelDf = indexer.fit(featureDf).transform(featureDf)


val seed = 5043
val Array(trainingData, testData) = labelDf.randomSplit(Array(0.7, 0.3), seed)

// train Random Forest model with training data set
val randomForestClassifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(3).setNumTrees(20).setFeatureSubsetStrategy("auto").setSeed(seed)
val randomForestModel = randomForestClassifier.fit(trainingData)

// run model with test data set to get predictions
// this will add new columns rawPrediction, probability and prediction
val predictionDf = randomForestModel.transform(testData)
predictionDf.show(10)
// measure the accuracy
val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setMetricName("areaUnderROC").setRawPredictionCol("rawPrediction")

val accuracy = evaluator.evaluate(predictionDf)
println("The accuracy for collected model is")
println(accuracy)
///////////////////////////////////////////////////////////////////////////////////////////////////
println("Data cleaning script excecuting")

val df = spark.read.option("header","true").
 option("inferSchema","true").
 format("csv").
 load("/user/xw2447/project/MyData_6.csv")

// Drop time column
val df1 = df.drop("time")

// Convert 'variety' data type to Double
val toDouble = udf[Double, Int]( _.toDouble)
val df2 = df1.withColumn("variety", toDouble(df("variety")))

// Replace Nan value with mean of the columns
val imputer = new Imputer().setInputCols(df2.columns).setOutputCols(df2.columns.map(c => s"${c}")).setStrategy("mean")

val dfClean = imputer.fit(df2).transform(df2)
println("The dataset cleaned is as follows")
dfClean.show(20)

println("The cleaned dataset is saved as csv file")
dfClean.coalesce(1).write.csv("/user/xw2447/project/cleaned")
///////////////////////////////////////////////////////////////////////////////////////////////////
println("The Noise reduction using PCA Executing")
import org.apache.spark.mllib.linalg.Vectors
val rdd = sc.textFile("/user/xw2447/project/cleaned").map(line => line.split(","))

val data = rdd.map(row => 
    new LabeledPoint(
          row(0).toDouble, 
          Vectors.dense(row.takeRight(row.length - 1).map(str => str.toDouble))
    )
  ).cache()



// Compute the top 5 principal components.
val pca = new PCA(8).fit(data.map(_.features))

// Project vectors to the linear space spanned by the top 5 principal
// components, keeping the label
val projected = data.map(p => p.copy(features = pca.transform(p.features)))
//projected.saveAsObjectFile("/user/xw2447/obj/")

//projected.coalesce(1).saveAsTextFile("/user/xw2447/out3/")
println("After running dataset through PCA")
projected.take(10).foreach(println)

MLUtils.saveAsLibSVMFile(projected.coalesce(1), "/user/xw2447/project/pca_op")

///////////////////////////////////////////////////////////////////////////////////////////////////
println("The Noise reduction using PCA Executing")
//projected.coalesce(1).write.c("/user/xw2447/out4/")
val data = spark.read.format("libsvm").load("/user/xw2447/project/pca_op")
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// // Select example rows to display.
// predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

println("The accuracy after the noise reduction is")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")


val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n ${rfModel.toDebugString}")