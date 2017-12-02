# Polynomial Regression
# A regression equation (or function) is linear when its parameters (coefficients) are linear.
# dependent variable = constant + (parameter)(independent variable x) + (parameter)(independent variable x)
# While the equation must remain linear in the parameters, you can transform the predictor variables x by raising it by a power to produce a curvature that better fits the data
# The model will still be linear even though the predictor variable x is squared or cubed etc (polynomial)
# Problem is, while linear regression can model curves, it might not be able to model the specific curve that exists in your data
# At that point will have to use non-linear regression (which is anything that doesn't follow linear equation structure)


# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# # Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ ., data = dataset)
summary(lin_reg)

# Fitting Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,dataset)
summary(poly_reg)

# Visualizing the Linear Regression results
library(ggplot2)
ggplot() +geom_point(aes(x = dataset$Level, y = dataset$Salary),
                     color = 'red')+ 
        geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
                  color = 'blue')+
        ggtitle('Truth or Bluff (Linear Regression)') +
        xlab('Level') +
        ylab('Salary')
        
        
# Visualizing the Polynomial Regression results
ggplot() +geom_point(aes(x = dataset$Level, y = dataset$Salary),
                     color = 'red')+ 
        geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
                  color = 'blue')+
        ggtitle('Truth or Bluff (Polynomial Regression)') +
        xlab('Level') +
        ylab('Salary')

# Predicting a new result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level =6.5)) #Add 6.5 level to dataset

# Predicting a new result with Polynomial Regression
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))
                 