import org.apache.spark.ml.feature.Imputer
import org.apache.spark.sql.types._


val df = spark.read.option("header","true").
 option("inferSchema","true").
 format("csv").
 load("/user/youraccout/project/MyData_6.csv")

// Drop time column
val df1 = df.drop("time")

// Convert 'variety' data type to Double
val toDouble = udf[Double, Int]( _.toDouble)
val df2 = df1.withColumn("variety", toDouble(df("variety")))

// Replace Nan value with mean of the columns
val imputer = new Imputer().setInputCols(df2.columns).setOutputCols(df2.columns.map(c => s"${c}")).setStrategy("mean")

val dfClean = imputer.fit(df2).transform(df2)
dfClean.show(20)
//dfClean.saveAsTextFile("/user/xw2447/cleaned/")
dfClean.coalesce(1).write.csv("/user/youraccout/cleaned/c1")