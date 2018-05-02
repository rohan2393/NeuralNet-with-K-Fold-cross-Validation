library(data.table)
## Load H2O library
library(h2o)

## Connect to H2O cluster
h2o.init(nthreads = -1)

## Define data paths
base_path = normalizePath("/Users/rohan/Desktop/MLP2")
mnist_path = paste0(base_path,"/mnist.csv")

## Ingest data
mnist.hex = h2o.importFile(path = mnist_path,destination_frame = "mnist.hex",header=NA)
dim(mnist.hex)

## Converting h2o hex file to a data frame
mnist.data<-as.data.frame(mnist.hex)
dim(mnist.data)

## Remonving first column from the dataframe
mnist.data.modified<-mnist.data[,-1]
dim(mnist.data.modified)

## Running the PCA model
pca.model<-prcomp(mnist.data.modified)

## Value of the rotated data is taken 
mnist_pca_x<-as.data.frame(pca.model$x[,1:261])
dim(mnist_pca_x)
## View(head(mnist_pca_x))

## Combining the data
mnist.data.complete <- as.data.frame(cbind(mnist.data[,1],mnist_pca_x))
#View(head(mnist.data.complete))

## Naming the first column
colnames(mnist.data.complete)[1] <- "C1"

## conveting the dataframe back to h2o hex file
mnist.hex<-as.h2o(mnist.data.complete,destination_frame = "mnist.hex")

## Splitting the data into two sets 1st set = 75% and sencond set = 25%
mnist.data.split <- h2o.splitFrame(data=mnist.hex, ratios=0.90)

## Cheching the dimensions of the split data
print(dim(mnist.data.split[[1]]))
print(dim(mnist.data.split[[2]]))

## Splitting the data and giving it names
train <- mnist.data.split[[1]]
## train
test <- mnist.data.split[[2]]
## test

## Specify the response and predictor columns
y<-"C1"
## View(y)
## Removing the response variable from 
x<-setdiff(names(train),y)
## View(x)

## Encode the response column as categorical for multinomial classification
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])

## Perform 5-fold cross-validation on training_frame using Kfold
system.time(
train_model <- h2o.deeplearning(
  x = x,
  y = y,
  training_frame = train,
  distribution = "multinomial",
  activation = "RectifierWithDropout",
  hidden = c(100,100,100),
  input_dropout_ratio = 0.2,
  sparse = TRUE,
  l1 = 1e-5,
  epochs = 10,
  nfolds = 10)
)

## View specified parameters of the deep learning model
train_model@parameters

## Examine the performance of the trained model model # display all performance metrics
h2o.performance(train_model) # training metrics 

## Cross validated Mean Square Error
h2o.mse(train_model, xval = TRUE)

## Obtaining the variable importance
head(as.data.table(h2o.varimp(train_model)))

## validation accuracy
h2o.hit_ratio_table(train_model)[1, 2]

## Classify the test set (predict class labels)
## This also returns the probability for each class 
pred <- h2o.predict(train_model, newdata = test)
list(Predicted_model_Accuracy_Test = mean(pred$predict == test$C1))

?prcomp()
?h2o.deeplearning()