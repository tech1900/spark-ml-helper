package org.apache.spark.mllib.optimization {

  import java.util.UUID
  import java.util.concurrent.atomic.{AtomicInteger, AtomicReference}

  import breeze.linalg.norm
  import org.apache.spark.annotation.{DeveloperApi, Since}
  import org.apache.spark.internal.Logging
  import org.apache.spark.mllib.linalg.{Vector, Vectors}
  import breeze.linalg.{norm, DenseVector => BDV}
  import org.apache.spark._
  import org.apache.spark.mllib.classification.LogisticRegressionModel
  import org.apache.spark.mllib.optimization.GradientDescent.{logInfo, logWarning}
  import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
  import org.apache.spark.mllib.util.DataValidators
  import org.apache.spark.rdd.RDD
  import org.apache.spark.rpc.{RpcCallContext, RpcEndpoint, RpcEndpointRef, RpcEnv}
  import org.apache.spark.scheduler.DAGScheduler
  import org.apache.spark.util.random.BernoulliSampler

  import scala.collection.mutable.ArrayBuffer

  class LogisticRegressionWithParallelSGD private[mllib] (
                                                   private var stepSize: Double,
                                                   private var numIterations: Int,
                                                   private var regParam: Double,
                                                   private var miniBatchFraction: Double)
    extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

    private val gradient = new LogisticGradient()
    private val updater = new SquaredL2Updater()
    @Since("0.8.0")
    override val optimizer = new ParallelGradientDescent(gradient, updater)
      .setStepSize(stepSize)
      .setNumIterations(numIterations)
      .setRegParam(regParam)
      .setMiniBatchFraction(miniBatchFraction)
    override protected val validators = List(DataValidators.binaryLabelValidator)

    /**
      * Construct a LogisticRegression object with default parameters: {stepSize: 1.0,
      * numIterations: 100, regParm: 0.01, miniBatchFraction: 1.0}.
      */
    @Since("0.8.0")
    @deprecated("Use ml.classification.LogisticRegression or LogisticRegressionWithLBFGS", "2.0.0")
    def this() = this(1.0, 100, 0.01, 1.0)

    override protected[mllib] def createModel(weights: Vector, intercept: Double) = {
      new LogisticRegressionModel(weights, intercept)
    }
  }

  class ParallelGradientDescent(private var gradient: Gradient, private var updater: Updater) extends GradientDescent(gradient,updater) {
    private var stepSize: Double = 1.0
    private var numIterations: Int = 100
    private var regParam: Double = 0.0
    private var miniBatchFraction: Double = 1.0
    private var convergenceTol: Double = 0.001

    /**
      * Set the initial step size of SGD for the first step. Default 1.0.
      * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
      */
    override def setStepSize(step: Double): this.type = {
      require(step > 0,
        s"Initial step size must be positive but got ${step}")
      this.stepSize = step
      this
    }

    /**
      * Set fraction of data to be used for each SGD iteration.
      * Default 1.0 (corresponding to deterministic/classical gradient descent)
      */
    override def setMiniBatchFraction(fraction: Double): this.type = {
      require(fraction > 0 && fraction <= 1.0,
        s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
      this.miniBatchFraction = fraction
      this
    }

    /**
      * Set the number of iterations for SGD. Default 100.
      */
    override def setNumIterations(iters: Int): this.type = {
      require(iters >= 0,
        s"Number of iterations must be nonnegative but got ${iters}")
      this.numIterations = iters
      this
    }

    /**
      * Set the regularization parameter. Default 0.0.
      */
    override def setRegParam(regParam: Double): this.type = {
      require(regParam >= 0,
        s"Regularization parameter must be nonnegative but got ${regParam}")
      this.regParam = regParam
      this
    }

    /**
      * Set the convergence tolerance. Default 0.001
      * convergenceTol is a condition which decides iteration termination.
      * The end of iteration is decided based on below logic.
      *
      *  - If the norm of the new solution vector is greater than 1, the diff of solution vectors
      *    is compared to relative tolerance which means normalizing by the norm of
      *    the new solution vector.
      *  - If the norm of the new solution vector is less than or equal to 1, the diff of solution
      *    vectors is compared to absolute tolerance which is not normalizing.
      *
      * Must be between 0.0 and 1.0 inclusively.
      */
    override def setConvergenceTol(tolerance: Double): this.type = {
      require(tolerance >= 0.0 && tolerance <= 1.0,
        s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
      this.convergenceTol = tolerance
      this
    }

    /**
      * Set the gradient function (of the loss function of one single data example)
      * to be used for SGD.
      */
    override def setGradient(gradient: Gradient): this.type = {
      this.gradient = gradient
      this
    }


    /**
      * Set the updater function to actually perform a gradient step in a given direction.
      * The updater is responsible to perform the update from the regularization term as well,
      * and therefore determines what kind or regularization is used, if any.
      */
    override def setUpdater(updater: Updater): this.type = {
      this.updater = updater
      this
    }

    @DeveloperApi
    override def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
      val (weights, _) = ParallelGradientDescent.runMiniBatchSGD(
        data,
        gradient,
        updater,
        stepSize,
        numIterations,
        regParam,
        miniBatchFraction,
        initialWeights,
        convergenceTol)
      weights
    }
  }

  object ParallelGradientDescent extends Logging{
    /**
      * Run stochastic gradient descent (SGD) in parallel using mini batches.
      * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
      * in order to compute a gradient estimate.
      * Sampling, and averaging the subgradients over this subset is performed using one standard
      * spark map-reduce in each iteration.
      *
      * @param data Input data for SGD. RDD of the set of data examples, each of
      *             the form (label, [feature values]).
      * @param gradient Gradient object (used to compute the gradient of the loss function of
      *                 one single data example)
      * @param updater Updater function to actually perform a gradient step in a given direction.
      * @param stepSize initial step size for the first step
      * @param numIterations number of iterations that SGD should be run.
      * @param regParam regularization parameter
      * @param miniBatchFraction fraction of the input data set that should be used for
      *                          one iteration of SGD. Default value 1.0.
      * @param convergenceTol Minibatch iteration will end before numIterations if the relative
      *                       difference between the current weight and the previous weight is less
      *                       than this value. In measuring convergence, L2 norm is calculated.
      *                       Default value 0.001. Must be between 0.0 and 1.0 inclusively.
      * @return A tuple containing two elements. The first element is a column matrix containing
      *         weights for every feature, and the second element is an array containing the
      *         stochastic loss computed for every iteration.
      */
    def runMiniBatchSGD(
                         data: RDD[(Double, Vector)],
                         gradient: Gradient,
                         updater: Updater,
                         stepSize: Double,
                         numIterations: Int,
                         regParam: Double,
                         miniBatchFraction: Double,
                         initialWeights: Vector,
                         convergenceTol: Double=0.001): (Vector, Array[Double]) = {

      // convergenceTol should be set with non minibatch settings
      if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
        logWarning("Testing against a convergenceTol when using miniBatchFraction " +
          "< 1.0 can be unstable because of the stochasticity in sampling.")
      }

      if (numIterations * miniBatchFraction < 1.0) {
        logWarning("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
          s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
      }


      // Record previous weight and current one to calculate solution vector difference

      var previousWeights: Option[Vector] = None
      var currentWeights: Option[Vector] = None

      val numExamples = data.count()

      // if no data, return initial weights to avoid NaNs
      if (numExamples == 0) {
        logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
        return (initialWeights, Array.empty[Double])
      }

      if (numExamples * miniBatchFraction < 1) {
        logWarning("The miniBatchFraction is too small")
      }

      // Initialize weights as a column vector
      var weights = Vectors.dense(initialWeights.toArray)
      val n = weights.size

      val combOp: ((BDV[Double], Double, Long), (BDV[Double], Double, Long)) => (BDV[Double], Double, Long) = (c1, c2) => {
        // c: (grad, loss, count)
        (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
      }

      /**
        * Create RpcEndPoint to track task results on the Driver
        *
        * @param data
        * @return
        */
      def getRpcEndPoint(data: RDD[(Double, Vector)]): (AtomicReference[(Vector, Array[Double], Array[String], Int)], RpcEndpointRef) = {
        val tasksCounter = Array.fill[AtomicInteger](numIterations + 1)(new AtomicInteger(0))
        val partialResult = Array.fill[(BDV[Double], Double, Long)](numIterations + 1)((BDV.zeros[Double](n), 0.0, 0L))
        val finalResult = new AtomicReference[(Vector, Array[Double], Array[String], Int)]()
        val numPartitions = data.getNumPartitions
        def action:PartialFunction[Any,Any] ={
          case UpdatePartialStepResult(stepId: Int, pResult: (BDV[Double], Double, Long)) =>
            partialResult.synchronized({
              partialResult(stepId) = combOp(partialResult(stepId), pResult)
            })
            tasksCounter(stepId).incrementAndGet() <= numPartitions
          case HasStepCompleted(stepId: Int) =>
            if (data.context.defaultParallelism != numPartitions) {
              throw new IllegalStateException(s"Executors went down Current: ${data.context.defaultParallelism} Expected:$numPartitions")
            }
            tasksCounter(stepId).get() == numPartitions
          case GetStepResult(stepId: Int) =>
            Option(partialResult(stepId))
          case ConvergedResult(gResult) =>
            if (finalResult.get() != null) { // Check if all threads are ended at the same iteration
              finalResult.get()._4 == gResult._4
            } else {
              finalResult.set(gResult)
              true
            }
        }

        val endpointRef = data.context.env.rpcEnv.setupEndpoint(UUID.randomUUID().toString, new RpcEndpoint {
          override val rpcEnv: RpcEnv = data.context.env.rpcEnv

          override def receiveAndReply(context: RpcCallContext): PartialFunction[Any, Unit] = {
            action.andThen[Unit](x => {
              context.reply(x)
            })
          }
        })
        (finalResult, endpointRef)
      }
      /**
        * Execute parallel aggregate coordinating with the driver
        *
        * @param data
        * @param rpcEndpointRef
        */
      def treeAggregate(data: RDD[(Double, Vector)],rpcEndpointRef: RpcEndpointRef): Unit = {
        val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
        data.repeat(1, numIterations + 1).foreachPartition((iterator: Iterator[(Partition, Int, Iterator[(Double, Vector)])]) => {
          val driver = SparkEnv.get.rpcEnv.setupEndpointRef(rpcEndpointRef.address, rpcEndpointRef.name)
          previousWeights = Some(weights)
          currentWeights = Some(weights)

          /**
            * For the first iteration, the regVal will be initialized as sum of weight squares
            * if it's L2 updater; for L1 updater, the same logic is followed.
            */
          var regVal = updater.compute(
            weights, Vectors.zeros(weights.size), 0, 1, regParam)._2

          val sampler: BernoulliSampler[Void] = new BernoulliSampler[Void](miniBatchFraction)
          var converged = false // indicates whether converged based on convergenceTol
          val log = new ArrayBuffer[String]()
          var count = 0
          while (!converged && iterator.hasNext) {
            val tuple = iterator.next()
            val i = tuple._2
            count += 1
            // Sample a subset (fraction miniBatchFraction) of the total data
            // compute and sum up the subgradients on this subset (this is one map-reduce)
            sampler.setSeed(42 + i)
            val partialResult =
              tuple._3.filter(_ => {
                sampler.sample() > 0
              }).aggregate((BDV.zeros[Double](n), 0.0, 0L))(
                seqop =(c, v) => {
                  // c: (grad, loss, count), v: (label, features)
                  val l = gradient.compute(v._2, v._1, weights, Vectors.fromBreeze(c._1))
                  (c._1, c._2 + l, c._3 + 1)
                }, combop =combOp)
            if (!driver.askSync[Boolean](UpdatePartialStepResult(i, partialResult))) {
              throw new IllegalStateException(s"${tuple._1} : Not able to send results to driver")
            }
            while (!driver.askSync[Boolean](HasStepCompleted(i))) {
              Thread.sleep(0, 200)
            }
            val optFullResult = driver.askSync[Option[(BDV[Double], Double, Long)]](GetStepResult(i))
            if (optFullResult.isEmpty) {
              throw new IllegalStateException(s"${tuple._1} : Not able to get task result driver")
            }
            val (gradientSum, lossSum, miniBatchSize) = optFullResult.get

            if (miniBatchSize > 0) {
              /**
                * lossSum is computed using the weights from the previous iteration
                * and regVal is the regularization value computed in the previous iteration as well.
                */
              stochasticLossHistory += lossSum / miniBatchSize + regVal
              val update = updater.compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble),
                stepSize, i, regParam)
              weights = update._1
              regVal = update._2

              previousWeights = currentWeights
              currentWeights = Some(weights)
              if (previousWeights.isDefined && currentWeights.isDefined) {
                converged = isConverged(previousWeights.get,
                  currentWeights.get, convergenceTol)
              }
            } else {
              log+= s"Iteration ($i/$numIterations). The size of sampled batch is zero"
            }
          }
          if (!driver.askSync[Boolean](ConvergedResult((weights, stochasticLossHistory.toArray, log.toArray, count)))) {
            throw new IllegalStateException("Not able to send results")
          }
        })
      }

      var rdd: RDD[(Double, Vector)] = null
      var rpcEndPoint: RpcEndpointRef = null
      var maxRetries = data.context.getConf.getInt("spark.stage.maxConsecutiveAttempts"
                                                          ,DAGScheduler.DEFAULT_MAX_CONSECUTIVE_STAGE_ATTEMPTS)
      var result: (Vector, Array[Double], Array[String], Int) = null
      import scala.util.control.Breaks._
      // Retry job similar to Tasks. Each retry, repartition to number of executors
      breakable {
        for (i <- 1 to maxRetries) {
          try {
            if (rdd != null) {
              logInfo(s"Retry [$i] runParallelMiniBatchSGD")
            }
            rdd = data.repartition(data.context.defaultParallelism).cache()
            val tuple = getRpcEndPoint(rdd)
            rpcEndPoint = tuple._2
            treeAggregate(rdd, rpcEndPoint)
            result = tuple._1.get()
          } catch {
            case e: Throwable => e.printStackTrace()
          } finally {
            data.context.env.rpcEnv.stop(rpcEndPoint)
            rdd.unpersist()
          }
          if (result != null) {
            break
          }
        }
      }
      if (result != null) {
        result._3.foreach(logInfo(_))
        logInfo("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
          result._2.takeRight(10).mkString(", ")))
        (result._1, result._2)
      } else {
        throw new SparkException(s"runParallelMiniBatchSGD failed after $maxRetries retries")
      }
    }

    private def isConverged(
                             previousWeights: Vector,
                             currentWeights: Vector,
                             convergenceTol: Double): Boolean = {
      // To compare with convergence tolerance.
      val previousBDV = previousWeights.asBreeze.toDenseVector
      val currentBDV = currentWeights.asBreeze.toDenseVector

      // This represents the difference of updated weights in the iteration.
      val solutionVecDiff: Double = norm(previousBDV - currentBDV)

      solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
    }

    private case class GetStepResult(stepId:Int)
    private case class HasStepCompleted(stepId:Int)
    private case class UpdatePartialStepResult(stepId:Int, weight:(BDV[Double],Double,Long))
    private case class ConvergedResult(result:(Vector,Array[Double],Array[String],Int))

    implicit class RepeatRDDHelper[T](rdd: RDD[T]){
      def repeat(start: Int, end: Int): RDD[(Partition,Int,Iterator[T])] ={
        new RepeatRDD(rdd,start,end)
      }
    }

    /**
      * Repeat the parent RDD upto end-start iterations
      * @param rdd
      * @param start
      * @param end
      * @tparam U
      */
    private class RepeatRDD[U](rdd: RDD[U],start: Int, end: Int) extends RDD[(Partition,Int,Iterator[U])](rdd) {
      override def compute(split: Partition, context: TaskContext): Iterator[(Partition,Int,Iterator[U])] = {
        Stream.range[Int](start, end).iterator.map(x => (split,x,rdd.compute(split, context)))
      }

      override protected def getPartitions: Array[Partition] = {
        rdd.partitions
      }
    }
  }

}
