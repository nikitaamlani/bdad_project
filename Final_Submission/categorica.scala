import sys.process._  
import scala.language.postfixOps
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator  
import org.apache.spark.mllib.evaluation.MulticlassMetrics  
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics  
import org.apache.spark.ml.classification.RandomForestClassifier  
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, CrossValidator}  
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoderEstimator}  
import org.apache.spark.ml.linalg.Vectors  
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
    load("/user/nga261/project/MyData_1.csv")
				

val cols = Array("sound","s","number","lat","l", "long", "weight", "wing", "color")

val assembler = new VectorAssembler().setHandleInvalid("skip").setInputCols(cols).setOutputCol("features")
val featureDf = assembler.transform(creditDf)
featureDf.printSchema()
featureDf.show(10)

// StringIndexer define new 'label' column with 'result' column
// val indexer = new StringIndexer().setInputCol("creditability").setOutputCol("label")
val indexer = new StringIndexer().setInputCol("variety").setOutputCol("label")

val labelDf = indexer.fit(featureDf).transform(featureDf)
labelDf.printSchema()
labelDf.show(10)

val seed = 5043
val Array(trainingData, testData) = labelDf.randomSplit(Array(0.7, 0.3), seed)

// train Random Forest model with training data set
val randomForestClassifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(3).setNumTrees(20).setFeatureSubsetStrategy("auto").setSeed(seed)
val randomForestModel = randomForestClassifier.fit(trainingData)
println(randomForestModel.toDebugString)

// run model with test data set to get predictions
// this will add new columns rawPrediction, probability and prediction
val predictionDf = randomForestModel.transform(testData)
predictionDf.show(10)
// measure the accuracy
val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setMetricName("areaUnderROC").setRawPredictionCol("rawPrediction")

val accuracy = evaluator.evaluate(predictionDf)
println(accuracy)

