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

  def fileExists(path: String) = new File(path).exists()

  def main( args: Array[String] ) = {

    val batchSize = 1000
    val iterations = 3
    val layerSize = 150

    val path = "/Users/davidblumenthal/data/full-ads-india/processed/"
    val altPath = "~/data/"

    val dataFile = "sentences_6M.csv"

    val dataPath = if (fileExists(path+dataFile)) path+dataFile else altPath+dataFile

    val sentenceIter = new LineSentenceIterator(new File(dataPath))
    val tokenizer = new DefaultTokenizerFactory()

    val word2vec = new Builder()
      .batchSize(batchSize) //# words per minibatch.
      .minWordFrequency(5) //
      .useAdaGrad(false) //
      .layerSize(layerSize) // word feature vector size
      .iterations(iterations) // # iterations to train
      .learningRate(0.025) //
      .minLearningRate(1e-3) // learning rate decays wrt # words. floor learning
      .negativeSample(10) // sample size 10 words
      .iterate(sentenceIter) //
      .tokenizerFactory(tokenizer)
      .build()

    println("Training")
    word2vec.fit()

    println("Saving")
    WordVectorSerializer.writeWordVectors(word2vec,"words.txt")
  }

}
