import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


// val conf = new SparkConf().setAppName("linearRegressionWine").setMaster("local[2]")
// val sc = new SparkContext(conf)
val rdd = sc.textFile("/user/xw2447/project/MyData_11f.csv").map(line => line.split(","))

val data = rdd.map(row =>
    new LabeledPoint(
          row.last.toDouble,
          Vectors.dense(row.take(row.length - 1).map(str => str.toDouble))
    )
  ).cache()



// Compute the top 5 principal components.
val pca = new PCA(5).fit(data.map(_.features))

// Project vectors to the linear space spanned by the top 5 principal
// components, keeping the label
val projected = data.map(p => p.copy(features = pca.transform(p.features)))
projected.saveAsObjectFile("/user/xw2447/obj/")
projected.saveAsTextFile("/user/xw2447/text/")
projected.take(10).foreach(println)