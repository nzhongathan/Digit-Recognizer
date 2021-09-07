# Beginner Classification Models
For beginners to computer vision, building full scaled Convolutional Neural Networks can be daunting and challenging. Here are simple and effective classification methods that work very well in terms of accuracy. Throughout this notebook I go through:
- Importing Libraries
- Preparing the Data
- Normalizing the Data
- Training the K Nearest Neighbors Classifier
- Training the XGBoost Classifier
- Training the Multinomial Naive Bayes Classifier
- Choosing the Model and Creating the Submission

## Importing Libraries
Here are some important libraries we will be using throughout this notebook.
- **NumPy** and **Pandas** help with loading in and working with the data
- **Matplotlib** and **Seaborn** helps with plotting and visualizing the images
- **Sklearn** is a great resource for beginners to use premade algorithms for machine learning tasks
  - KNeighbors Classifier will be one of the models we are training
  - Train_test_split is how we will split our data to obtain validatoin scores
  - Accuarcy_score will be how we evalute our model
  - XGBClassifier will be another one of the models we are training
  - Naive_Bayes and the three classifiers will be the final three models we are training

## Preparing the Data
First, we have to read the train and test data. We can see that unlike more complex computre vision tasks, we are given the pixel values in a table, and have no actual "images" to load in.

