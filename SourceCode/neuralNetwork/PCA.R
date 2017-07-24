
# import training data
trainingData <- read.csv("features.csv", header = FALSE)

#change input data as training data and label
trainingLabel <- data.frame(trainingData[,3073])
trainingData <- trainingData[, c(1:3072)]

#normalization
trainingData <- (trainingData-mean(unlist(trainingData)))/sd(unlist(trainingData))

#pca
pca <- prcomp(trainingData, scale. = T, center. = T)
summary(pca)


dim(pca$x)
summary(pca)

pca$x
pov = pca$sdev^2/sum(pca$sdev^2)
pov
x = c(1:1186)
plot(x,pov, ylim = c(0,0.01))

y <- data.frame(pca$x[,1:2])
colnames(y) <- 