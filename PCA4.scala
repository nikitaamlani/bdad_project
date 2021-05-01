import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.util.MLUtils
// val conf = new SparkConf().setAppName("linearRegressionWine").setMaster("local[2]")
// val sc = new SparkContext(conf)
val rdd = sc.textFile("/user/xw2447/project/c1.csv").map(line => line.split(","))

val data = rdd.map(row => 
    new LabeledPoint(
          row(0).toDouble, 
          Vectors.dense(row.takeRight(row.length - 1).map(str => str.toDouble))
    )
  ).cache()



// Compute the top 5 principal components.
val pca = new PCA(6).fit(data.map(_.features))

// Project vectors to the linear space spanned by the top 5 principal
// components, keeping the label
val projected = data.map(p => p.copy(features = pca.transform(p.features)))
//projected.saveAsObjectFile("/user/xw2447/obj/")

//projected.coalesce(1).saveAsTextFile("/user/xw2447/out3/")
projected.take(10).foreach(println)
MLUtils.saveAsLibSVMFile(projected.coalesce(1), "/user/xw2447/out4/")
//projected.coalesce(1).write.c("/user/xw2447/out4/")
