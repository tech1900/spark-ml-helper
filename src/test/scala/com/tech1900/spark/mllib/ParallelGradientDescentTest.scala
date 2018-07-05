package com.tech1900.spark.mllib

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{LogisticGradient, LogisticRegressionWithParallelSGD, ParallelGradientDescent, SimpleUpdater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession
import org.scalatest.FunSuite

class ParallelGradientDescentTest extends FunSuite{
  test("applyGradient"){
    val noOfCores = 5
    val spark = SparkSession.builder().master(s"local[$noOfCores]")
      .config("spark.sql.shuffle.partitions", noOfCores)
      .getOrCreate()
    val regression = new LogisticRegressionWithParallelSGD()
    val model = regression.run(spark.range(0, Short.MaxValue*5).rdd.map(x => {
      new LabeledPoint((x%2).doubleValue(), Vectors.dense(Array(x.doubleValue())))
    }))
    println(model,model.weights)
  }
}
