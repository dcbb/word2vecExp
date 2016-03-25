import java.io.File

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.models.word2vec.Word2Vec.Builder
import org.deeplearning4j.text.sentenceiterator.{LineSentenceIterator, SentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory

/**
  * Created by davidblumenthal on 24.03.16.
  */
object Dl4jMain {

  def getFile(paths: Seq[String], fileName: String) : File = {
    paths.map { p => new File(p+fileName) }
      .filter( _.exists() )
      .head
  }

  def run( config: Config ) = {

    val sentenceIter = new LineSentenceIterator(config.dataFile)
    val tokenizer = new DefaultTokenizerFactory()

    val word2vec = new Builder()
      .batchSize(config.batchSize) //# words per minibatch.
      .minWordFrequency(5) //
      .useAdaGrad(false) //
      .layerSize(config.layerSize) // word feature vector size
      .iterations(config.iterations) // # iterations to train
      .learningRate(0.025) //
      .minLearningRate(1e-3) // learning rate decays wrt # words. floor learning
      .negativeSample(10) // sample size 10 words
      .iterate(sentenceIter) //
      .tokenizerFactory(tokenizer)
      .build()

    println("Training")
    word2vec.fit()

    println("Saving")
    WordVectorSerializer.writeWordVectors(word2vec,config.outFile)
  }

  case class Config( dataFile: File = new File("."),
                     outFile: String = "words.txt",
                     batchSize: Int = 1000,
                     iterations: Int = 3,
                     layerSize: Int = 150 )

  def main( args: Array[String] ) = {

    val parser = new scopt.OptionParser[Config]("dl4jexp") {
      opt[String]("input").required()
        .action { (x, conf) => conf.copy(dataFile = new File(x)) }
        .validate { x => if (new File(x).exists()) success else failure("input file not found") }
      opt[String]("output")
        .action { (x, conf) => conf.copy(outFile=x) }
      opt[Int]("batchSize").action( (x,c) => c.copy(batchSize = x) )
      opt[Int]("iterations").action( (x,c) => c.copy(iterations = x) )
      opt[Int]("layerSize").action( (x,c) => c.copy(layerSize = x) )
    }

    parser.parse(args, Config()) match {
      case Some(config) => run(config)
      case None =>
    }
  }

}
