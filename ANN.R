##### Neural Network -------------------

##### Part 1: Neural Networks -------------------
## Example: Modeling the Strength of Concrete  ----

## Step 2: Exploring and preparing the data ----
#the concrete dataset contains 1,030 examples of concrete
#with eight features describing the components used in the mixture. These features
#are thought to be related to the final compressive strength and they include the
#amount (in kilograms per cubic meter) of cement, slag, ash, water, superplasticizer,
#coarse aggregate, and fine aggregate used in the product in addition to the aging
#time (measured in days).

# read in data and examine structure
concrete <- read.csv("concrete.csv")
str(concrete)

# custom normalization function

#The nine variables in the data frame correspond to the eight features and one
#outcome we expected, although a problem has become apparent. Neural networks
#work best when the input data are scaled to a narrow range around zero, and here,
#we see values ranging anywhere from zero up to over a thousand.
#Typically, the solution to this problem is to rescale the data with a normalizing or
#standardization function.

normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
concrete_norm <- as.data.frame(lapply(concrete, normalize))

# confirm that the range is now between zero and one
summary(concrete_norm$strength)

# compared to the original minimum and maximum
summary(concrete$strength)

# create training and test data

#we will partition the data into
#a training set with 75 percent of the examples and a testing set with 25 percent. The
#CSV file we used was already sorted in random order, so we simply need to divide it
#into two portions 

#We'll use the training dataset to build the neural network
#and the testing dataset to
#evaluate how well the model generalizes to future results.
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

## Step 3: Training a model on the data ----
# train the neuralnet model

#To model the relationship between the ingredients used in concrete and the strength
#of the finished product, we will use a multilayer feedforward neural network. The
#neuralnet package by Stefan Fritsch and Frauke Guenther provides a standard
#and easy-to-use implementation of such networks. It also offers a function to plot
#the network topology.
install.packages('neuralnet')
library(neuralnet)

# simple ANN with only a single hidden neuron
set.seed(12345) # to guarantee repeatable results
#We'll begin by training the simplest multilayer feedforward network with only a
#single hidden node:
concrete_model <- neuralnet(formula = strength ~ cement + slag +
                              ash + water + superplastic + 
                              coarseagg + fineagg + age,
                              data = concrete_train)

# visualize the network topology

#there is one input node for each of the eight features, followed
#by a single hidden node and a single output node that predicts the concrete strength.
#The weights for each of the connections are also depicted, as are the bias terms
#(indicated by the nodes labeled with the number 1). The bias terms are numeric
#constants that allow the value at the indicated nodes to be shifted upward or
#downward, much like the intercept in a linear equation.

plot(concrete_model)

#at the bottom R reports the number of training steps and an error
#measure called the Sum of Squared Errors (SSE), which as you might expect, is
#the sum of the squared predicted minus actual values. A lower SSE implies better
#predictive performance. This is helpful for estimating the model's performance on
#the training data, but tells us little about how it will perform on unseen data.

## Step 4: Evaluating model performance ----
# obtain model results

#To generate predictions on the test dataset, we can use the compute()

#The compute() function works a bit differently from the predict() functions
#we've used so far. It returns a list with two components: $neurons, which stores the
#neurons for each layer in the network, and $net.result, which stores the predicted
#values.
model_results <- compute(concrete_model, concrete_test[1:8])
# obtain predicted strength values



predicted_strength <- model_results$net.result

# examine the correlation between predicted and actual values

#Because this is a numeric prediction problem rather than a classification problem, we
#cannot use a confusion matrix to examine model accuracy. Instead, we must measure
#the correlation between our predicted concrete strength and the true value. This
#provides insight into the strength of the linear association between the two variables.

#Correlations close to 1 indicate strong linear relationships between two variables.
#Therefore, the correlation here of about 0.806 indicates a fairly strong relationship. 
#This implies that our model is doing a fairly good job, 
#even with only a single hidden node.

#No need to be alarmed if your result differs. Because the neural network
#begins with random weights, the predictions can vary from model
#to model.  to match these results exactly, try using set.
#seed(12345) before building the neural network.

cor(predicted_strength, concrete_test$strength)

## Step 5: Improving model performance ----


# a more complex neural network topology with 5 hidden neurons

#Given that we only used one hidden node, it is likely that we can improve the
#performance of our model.

set.seed(12345) # to guarantee repeatable results

#let's see what happens when we increase the number of hidden nodes to
#five. We use the neuralnet() function as before, but add the hidden = 5 parameter

concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic + 
                               coarseagg + fineagg + age,
                               data = concrete_train, hidden = 5)

# plot the network

#Plotting the network again, we see a drastic increase in the number of connections.

#the reported error (measured again by SSE) has been reduced from
#5.08 in the previous model to 1.63 here. Additionally, the number of training steps
#rose from 4,882 to 86,849, which should come as no surprise given how much more
#complex the model has become.

plot(concrete_model2)

# evaluate the results as we did before
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)

#try using different numbers of hidden nodes, applying different activation functions, and so on. The
#?neuralnet help page provides more information on the various parameters that
#can be adjusted.

