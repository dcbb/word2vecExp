name := "wordVecExp"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc3.8"
libraryDependencies += "org.nd4j" % "nd4j-x86" % "0.4-rc3.8"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-ui" % "0.4-rc3.8"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.4.0"

resolvers += Resolver.sonatypeRepo("public")

fork := true

javaOptions += "-Xmx55G"
