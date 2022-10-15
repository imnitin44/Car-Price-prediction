# Car Price prediction

![275109233_1412429589176761_7257274794558322188_n](https://user-images.githubusercontent.com/111623861/195993279-467440e4-79a1-49e9-94fd-a9c7b15b0626.png)

### Primary Goal :- To predict the selling price of the used cars.

## Steps performed

### 1. Collecting Data:
As we know that machines initially learns from the dataset that you give them. It is of the utmost importance to collect reliable data so that your machine learning model can find the correct patterns. The quality of the data that you feed to the machine will determine how accurate your model is. If you have incorrect or outdated data, you will have wrong outcomes or predictions which are not relevant. 
Make sure we use data from a reliable source, as it will directly affect the outcome of our model. Good data is relevant, contains very few missing and repeated values, and has a good representation of the various subcategories/classes present. 

For this project Data was collected from the internally and externally (third party API's).


### 2. Preprocessing of Data:
After we have our data,we have to prepare & clean the data as per our requirements. We can do this by :

a) Putting together all the data you have and randomizing it. This helps make sure that data is evenly distributed, and the ordering does not affect the learning process.

b) Cleaning the data to remove unwanted data, missing values, rows, and columns, duplicate values, data type conversion, etc. We might even have to restructure the dataset and change the rows and columns or index of rows and columns.
Data had unwanted string values that were removed and the data type was changed from object to float as per our required data type.

c) Visualizing the data to understand how it is structured and understand the relationship between various variables and classes present.Data had many Null values,the data visualization was perfomed to also to get idea about the data and replace the null values with mean/median.

d) Train_Test_Split - a training set and a testing set. The training set is the set your model learns from. A testing set is used to check the accuracy of your model after training.

e) Standarding Data - We have to sandardize the data set to get the high accuracy.Here by using StandardScaler() function data was Standardised.X_train was fit_transformed to avoid data leakage and X_test was just transformed.

### 3. Model Selection : 
A machine learning model determines the output you get after running a machine learning algorithm on the collected data. It is important to choose a model which is relevant to the task at hand. Over the years, scientists and engineers developed various models suited for different tasks like speech recognition, image recognition, prediction, etc. Apart from this, you also have to see if your model is suited for numerical or categorical data and choose accordingly.
Here data has the relationship between input and output so its a regression problem.Linear Regression was taken into consideration.

### 4. Training the Model:
Training is the most important step in machine learning. In training, we have to pass the prepared data to our machine learning model to find patterns and make predictions. It results in the model learning from the data so that it can accomplish the task set. Over time, with training, the model gets better at predicting. 

### 5. Evaluating the Model:
After training our model,we have to predict to see how it’s performing. We check our model perfomance with the training and test dataset.
Linear Regression model was trained with the X_train,y_train and the model scoring was checked with both training and test data.Scoring was found to be R-squared Train =  0.6772561122214695 R-squared Test =  0.6847476846833296.


### 6. Making Predictions
We can use your model on unseen data to make predictions accurately.X_test dataset was passed to the model to predict.Metrics (scoring) module from sklearn library was used to check scoring for regression model.
RMSE was found 463908.5629934025. Model had not that much higher scoring as expected so the dataset was tried with the Ridge, Lasso & ElasticNet with cross validation (Regularisation techniques).

New Features were added to the model by using polynomial features(degree=2).

### Ridge model scoring 
R-squared Train =  0.9062843865402949
R-squared Test =  0.9037818464378018
RMSE =  256290.0558786351

### Lasso model scoring
R-squared Train =  0.9045992566665596
R-Squered Test =  0.9042400267477413
RMSE =  255679.11511321831

### ElasticNet model scoring
R-squared Train =  0.8772503708304519
R-squared Test =  0.8737063868986282
RMSE =  293625.61576684366

Here Lasso method had the low bias and low variance.
So,Lasso model was taken as the best reqularization method that fit data well and had the highest accuracy to predict selling price.




