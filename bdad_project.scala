
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

import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header","true").
    option("inferSchema","true").
    format("csv").
    load("/user/nga261/project/mooc.csv")


val df = (data.select(data("certified").as("label"), 
          $"registered", $"viewed", $"explored", 
          $"final_cc_cname_DI", $"gender", $"nevents", 
          $"ndays_act", $"nplay_video", $"nchapters", $"nforum_posts"))

val indexer1 = new StringIndexer().
    setInputCol("final_cc_cname_DI").
    setOutputCol("countryIndex").
    setHandleInvalid("keep") 
val indexed1 = indexer1.fit(df).transform(df)

val indexer2 = new StringIndexer().
    setInputCol("gender").
    setOutputCol("genderIndex").
    setHandleInvalid("keep")
val indexed2 = indexer2.fit(indexed1).transform(indexed1)

// one hot encoding
val encoder = new OneHotEncoderEstimator().
  setInputCols(Array("countryIndex", "genderIndex")).
  setOutputCols(Array("countryVec", "genderVec"))
val encoded = encoder.fit(indexed2).transform(indexed2)

val nanEvents = encoded.groupBy("nevents").count().orderBy($"count".desc)
for (line <- nanEvents){
    println(line)
}

val neventsMedianArray = encoded.stat.approxQuantile("nevents", Array(0.5), 0)
val neventsMedian = neventsMedianArray(0)

val ndays_actMedianArray = encoded.stat.approxQuantile("ndays_act", Array(0.5), 0)
val ndays_actMedian = ndays_actMedianArray(0)

val nplay_videoMedianArray = encoded.stat.approxQuantile("nplay_video", Array(0.5), 0)
val nplay_videoMedian = nplay_videoMedianArray(0)

val nchaptersMedianArray = encoded.stat.approxQuantile("nchapters", Array(0.5), 0)
val nchaptersMedian = nchaptersMedianArray(0)
val filled = encoded.na.fill(Map(
  "nevents" -> neventsMedian, 
  "ndays_act" -> ndays_actMedian, 
  "nplay_video" -> nplay_videoMedian, 
  "nchapters" -> nchaptersMedian))

val assembler = (new VectorAssembler().setInputCols(Array(
  "viewed", "explored", "nevents", "ndays_act", "nplay_video", 
  "nchapters", "nforum_posts", "countryVec", "genderVec")).
   setOutputCol("features"))

val output = assembler.transform(filled).select($"label",$"features")

val Array(training, test) = output.select("label","features").
                            randomSplit(Array(0.7, 0.3), seed = 12345)
val rf = new RandomForestClassifier()
val paramGrid = new ParamGridBuilder().
  addGrid(rf.numTrees,Array(20,50,100)).
  build()


val cv = new CrossValidator().
  setEstimator(rf).
  setEvaluator(new MulticlassClassificationEvaluator().setMetricName("weightedRecall")).
  setEstimatorParamMaps(paramGrid).
  setNumFolds(3).
  setParallelism(2)


val model = cv.fit(training)

val results = model.transform(test).select("features", "label", "prediction")

val predictionAndLabels = results.
    select($"prediction",$"label").
    as[(Double, Double)].
    rdd

val bMetrics = new BinaryClassificationMetrics(predictionAndLabels)
val mMetrics = new MulticlassMetrics(predictionAndLabels)
val labels = mMetrics.labels


println("Confusion matrix:")
println(mMetrics.confusionMatrix)

labels.foreach { l =>
  println(s"Precision($l) = " + mMetrics.precision(l))
}
labels.foreach { l =>
  println(s"Recall($l) = " + mMetrics.recall(l))
}


labels.foreach { l =>
  println(s"FPR($l) = " + mMetrics.falsePositiveRate(l))
}


labels.foreach { l =>
  println(s"F1-Score($l) = " + mMetrics.fMeasure(l))
}

val precision = bMetrics.precisionByThreshold
precision.foreach { case (t, p) =>
  println(s"Threshold: $t, Precision: $p")
}


val recall = bMetrics.recallByThreshold
recall.foreach { case (t, r) =>
  println(s"Threshold: $t, Recall: $r")
}


val PRC = bMetrics.pr

val f1Score = bMetrics.fMeasureByThreshold
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 1")
}

val beta = 0.5
val fScore = bMetrics.fMeasureByThreshold(beta)
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 0.5")
}


val auPRC = bMetrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)


val thresholds = precision.map(_._1)
val roc = bMetrics.roc


val auROC = bMetrics.areaUnderROC
println("Area under ROC = " + auROC)

