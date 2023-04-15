# Paige Reynolds
# STAT 4020
# Lab 3: Classification

rm(list= ls())
library(ISLR) # Contains desired data
library(MASS) # Contains f-ns for discriminant analysis
library(class) # Contains the k-nn classifier function

head(Smarket)

# Look at the correlation among the variables:
# cor(Smarket) throws an error since Direction is qualitative
cor(Smarket[, -9])
plot(Smarket[, -9])

# Observe volume vs year
plot(x= Smarket$Year, y= Smarket$Volume, pch= 20, xlab= "Year", ylab= "Volume",
     main= "")

# Plot over each day to see a clearer picture
plot(Smarket$Volume, pch= 20, xlab= "Days", ylab= "Volume", main= "")

# LOGISTIC REGRESSION #
# Model the day's direction using the previous give lags and the trading volume
dir.mod <- glm(Direction ~ ., data= Smarket[, -c(1,8)],
               family = binomial(link= logit))
summary(dir.mod)

# Observe how the response was coded for proper interpretation
contrasts(Smarket$Direction)

# Create a vector of predicted outcomes:
pred.probs <- predict(dir.mod, type= "response")
# type= response indicates that we want the actual probability, not the logit-transform of the
# probability (say)
head(pred.probs)

# Initialize the same length as the number of records
labeled.preds <- rep("Down", times= dim(Smarket)[1])
labeled.preds[pred.probs > .5] <- "Up"
head(labeled.preds)

# Use table() to create the confusion matrix
cmat <- table(labeled.preds, Smarket$Direction)
# Predictions are the rows, observed
print(cmat)
# Outcomes are the columns
(cmat[1, 1] + cmat[2, 2]) / dim(Smarket)[1]

# USING A HOLD-OUT TESTING SET #
# Create a training set and a testing set
my.test <- Smarket[Smarket$Year == 2005, ]
my.train <- Smarket[Smarket$Year < 2005, ]
head(my.test)
head(my.train)

# Fit the model and compare to the test data
dir.mod2 <- glm(Direction ~ ., data= Smarket[, -c(1,8)], family= binomial,
                subset= (Smarket$Year < 2005))
summary(dir.mod2)

pred.probs2 <- predict(dir.mod2, my.test, type= "response")
lab.preds2 <- rep("Down", times= dim(my.test)[1])
lab.preds2[pred.probs2 > .5] <- "Up"

cmat2 <- table(lab.preds2, my.test$Direction)
print(cmat2)

(cmat2[1, 1] + cmat2[2, 2]) / dim(my.test)[1]

# a dumb prediction rule: Always pick up
dumb <- rep("Up", times= dim(my.test)[1])
table(dumb, my.test$Direction)

141 / dim(my.test)[1]

# LINEAR DISCRIMINANT ANALYSIS #
# Same problem as above, but use LDA instead of logistic regression
(lda.mod <- lda(Direction ~ Lag1 + Lag2, data= Smarket,
                subset= Smarket$Year < 2005))
plot(lda.mod)
# Provides class predictions (in class), Bauesian posterior group
# probabilities (posterior)
lda.preds <- predict(lda.mod, my.test)
table(lda.preds$class, my.test$Direction)
# (35+106) / 252 = .559 prediction accuracy

# QUATRATIC DISCRIMINANT ANALYSIS #
# K-Nearest Neighbors Classification #
# Pull out (lag1, lag2) for the testing and training data
train.X <- cbind(my.train$Lag1, my.train$Lag2)
test.X <- cbind(my.test$Lag1, my.test$Lag2)

# Get labels and try it
train.labs <- my.train$Direction
# Use if you want to reproduce the results
set.seed(1)
knn.res <- knn(train.X, test.X, train.labs, k=1)
knn.tab <- table(knn.res, my.test$Direction)
print(knn.tab)

(knn.tab[1, 1] + knn.tab[2, 2]) / dim(my.test)[1]

# k=3 neighbors
knn.res <- knn(train.X, test.X, train.labs, k=3)
knn.tab <- table(knn.res, my.test$Direction)
print(knn.tab)

(knn.tab[1, 1] + knn.tab[2, 2]) / dim(my.test)[1]

# AN APPLICATION TO CARAVAN INSURANCE DATA #

dim(Caravan)
attach(Caravan)
summary(Purchase)
348 / 5822 # Only 6% of people purchased caravan insurance

standardized.X=scale(Caravan[, -86])
var(Caravan[, 1])
var(Caravan[, 2])
var(standardized.X[, 1])
var(standardized.X[, 2])
# Now, every column of standardized.X has a sd of 1 and a mean of 0

# Now, split the observations into a test set with the first 1,000 obs
# and a training set
test= 1:1000
train.X= standardized.X[-test, ] # Observations > 1000
train.Y= Purchase[-test]
test.X= standardized.X[test, ] # Observations 1-1000
test.Y= Purchase[test]
set.seed(1)
knn.pred= knn(train.X, test.X, train.Y, k=1)
mean(test.Y!= knn.pred)
mean(test.Y!= "No")

table(knn.pred, test.Y)
9 / (68 + 9)
# Among 77 such customers, 11.7% actually purchase insurance

knn.pred= knn(train.X, test.X, train.Y, k = 3)
table(knn.pred, test.Y)
5 /26
# Success rate increased to 19%

knn.pred= knn(train.X, test.X, train.Y, k= 5)
table(knn.pred, test.Y)
4 / 15
# Success rate increased to 27%

glm.fits= glm(Purchase ~., data= Caravan, family= binomial,
              subset= -test)
# Warning message
glm.probs= predict(glm.fits, Caravan[test, ], type= "response")
glm.pred= rep("No", 1000)
glm.pred[glm.probs > .5] = "Yes"
table(glm.pred, test.Y)
11 / (22 + 11)
# Success rate increase to 33%







