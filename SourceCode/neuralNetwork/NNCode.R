library(caret)

computeHidden <- function(x) {
  val <- as.double(1/(1+exp(-x)))
  return(val)
}

initialWeight <- function(x) {
  noise_factor <- 0.55
  return(2.0 * (x - 0.5 ) * noise_factor)
}

computeTotalError <- function(expectedOutput, actualOutput) {
  eTotal <- 0
  total <- ncol(expectedOutput)
  for(i in c(1:total)) {
    diff <- as.double(expectedOutput[,i] - actualOutput[,i])
    eTotal <- eTotal + (1/2)*(diff * diff)
  }
  return(eTotal)
}

# import training data
tdata <- read.table("pcaData_10.txt")
tLabel <- read.table("trainingLabel.txt")
colnames(tLabel) <- c("class")
data <- data.frame(cbind(class = tLabel, tdata))

#code to create partition of data[ the p value determines the %]
index <- createDataPartition(data$class, p=0.1, list=FALSE)
trainD <- data[index,]

newTestD <- data[-index,]
index1 <- createDataPartition(newTestD$class, p=0.1, list=FALSE)

testD <- newTestD[index1,]
validationD <- newTestD[-index1,]

trainingData <- trainD[,2:11]
trainingLabel <- data.frame(trainD[,1])

trainingData <- cbind(bias = rep(1,nrow(trainingData)), trainingData)

#set parameters
numHiddenUnits <- 15
numHiddenLayers <- 1
numIterations <- 10
learningRate <- 0.15
alpha <- 0

numTuples <- nrow(trainingData)
numInput <- ncol(trainingData)
numOutput <- max(unique(trainingLabel[,1]))
print(numOutput)
#set initial weights from input to hidden layer
weightIH <- matrix(, nrow = numInput, ncol = numHiddenUnits) 
weightIH <- matrix(runif(numInput*numHiddenUnits, 0, 1), numInput, numHiddenUnits)
weightIH <- apply(weightIH, MARGIN=2, FUN=initialWeight)

#set initial weights from hidden to output layer
weightHO <- matrix(, nrow = (numHiddenUnits+1), ncol = numOutput)
weightHO <- matrix(runif((numHiddenUnits+1)*numOutput, 0, 1), (numHiddenUnits+1), numOutput)
weightHO <- apply(weightHO, MARGIN=2, FUN=initialWeight)

totalError <- data.frame(Epoch= integer(), Error=double())
deltaWeightIH <- matrix(0, nrow = numInput, ncol = numHiddenUnits+1)
deltaWeightHO <- matrix(0, nrow = (numHiddenUnits+2), ncol = numOutput)



for(x in c(1:numIterations)) {
  cat("Epoch - ", x, "\n")
  
  errorTotal <- 0
  for(p in c(1:numTuples)) {

    cat("Row - ", p, "\n")
    input <- as.matrix(trainingData[p,])
    
    #forward pass input to hidden layer
    sum <- input %*% weightIH
    hidden <- as.matrix(sapply(sum, computeHidden))
    hidden <- t(hidden)
    hidden <- cbind(bias = rep(1,nrow(hidden)), hidden)
    
    #forward pass hidden to output
    sumO <- hidden %*% weightHO
    actualOutput <- as.matrix(sapply(sumO, computeHidden))
    actualOutput <- t(actualOutput)
    
    #compute mean squared error
    outputLabel <- trainingLabel[p,]
    expectedOutput <- matrix(rep(0), nrow=1, ncol=numOutput)
    #   print(expectedOutput)
    colnames(expectedOutput) <- c(0:(numOutput-1))
    expectedOutput[,(outputLabel-1)] <- 1
    errorTotal <- errorTotal + computeTotalError(expectedOutput, actualOutput)
    
    #back propagation output to hidden and hidden to input -
    deltaO <- matrix(, nrow = 1, ncol = numOutput)
    for(i in c(1:numOutput)) {
      deltaValue <- - (expectedOutput[,i] - actualOutput[,i]) * actualOutput[,i] * (1-actualOutput[,i])
      deltaO[,i] <- deltaValue
    }
    
    deltaH <- matrix(, nrow = 1, ncol = numHiddenUnits)
    for(i in c(2:(numHiddenUnits+1))) {
      sumH <- matrix(0, nrow =1, ncol= numHiddenUnits+1)
      for(j in c(1:numOutput)) {
        sumH[,i] <- sumH[,i] + weightHO[i,j]*deltaO[j]
      }
      deltaH[,i-1] <- sumH[,i] * hidden[1,i] * (1 - hidden[1, i])
    }
    
    
    for(i in c(1:numHiddenUnits)) {
      for(j in c(1:numInput)) {
        deltaWeightIH[j,i] <- input[1,j] * deltaH[,i] 
        weightIH[j,i] <- weightIH[j,i] - (learningRate * deltaWeightIH[j,i])+ (alpha * deltaWeightIH[j,i])
      }
    }
    
    
    for(i in c(1:numOutput)) {
      for(j in c(1:(numHiddenUnits+1))) {
        deltaWeightHO[j,i] <- hidden[1,j] * deltaO[,i]
        weightHO[j,i] <- weightHO[j,i] - (learningRate * deltaWeightHO[j,i]) + (alpha * deltaWeightHO[j,i])
      }
    }
  }
  #compute total average mean squared error for each epoch
  avgErrorTotal <- errorTotal/numTuples
  print("Error:")
  print(avgErrorTotal)
  totalError <- rbind(totalError, c(x, avgErrorTotal))
}
colnames(totalError) <- c("SNo", "Error")


#testing

testData <- testD[,2:11]
testLabel <- data.frame(testD[,1])
testData <- cbind(bias = rep(1,nrow(testData)), testData)

testNumTuples <- nrow(testData)
testResult <- data.frame(RowId= character(), ActualOutput=character(), stringsAsFactors = F)

#run forward pass [prediction] to all rows in the test data
for(tRow in c(1:testNumTuples)) {
  testInput <- as.matrix(testData[tRow,])
  testSum <- testInput %*% weightIH
  testHidden <- as.matrix(sapply(testSum, computeHidden))
  testHidden <- t(testHidden)
  testHidden <- cbind(bias = rep(1,nrow(testHidden)), testHidden)
  
  testSumO <- testHidden %*% weightHO
  testActualOutput <- as.matrix(sapply(testSumO, computeHidden))
  testActualOutput <- t(testActualOutput)
  colnames(testActualOutput) <- c(0:(numOutput-1))
  
  #convert [0,1] vector to predicted diget
  testActualOutput <- ifelse(testActualOutput == max(testActualOutput), 1, 0)
  actualLabel <- colnames(testActualOutput)[apply(testActualOutput, 2, function(x) any(x==1))]
  
  if(length(actualLabel) == 0 || length(actualLabel) > 1) {
    actualLabel <- 1585
  }
  
  #print(actualLabel)
  testResult <- rbind(testResult, data.frame(RowId =tRow, ActualOutput=actualLabel, stringsAsFactors = F))
  
}

testLabel$RowId <- seq.int(nrow(testLabel))

testResult <- merge(testResult, testLabel, by.x="RowId", by.y = "RowId")
colnames(testResult) <- c("RowId", "ActualOutput", "ExpectedOutput")
testResult$correct <- ifelse(testResult$ActualOutput == testResult$ExpectedOutput,1,0)

accuratelyPredicted <- nrow(data.frame(testResult[which(testResult$correct == 1), ]))
totalPredicted <- nrow(testResult)
accuracyPercent <- (accuratelyPredicted/totalPredicted) * 100

print("The accuracy Percent for the Test data is :")
print(accuracyPercent)
