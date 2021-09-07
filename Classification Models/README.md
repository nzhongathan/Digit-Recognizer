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

![Capture(18)](https://user-images.githubusercontent.com/69808907/132283424-78dab8e2-ae78-4419-b536-a2879fb211da.PNG)

But the table of data is not enough as we have to split the label, or what we are predicting, from the training data, or the pixels. We will label the pixels as our "x" values and our labels as our "y" values. Finally, we will split them into train and validation datasets, which will be used to train and validate our model respectively.

## Normalizing the Data
I will be doing some simple preprocessing to start out, to improve the model quality and speed up training. First I will normalize the data to the values of 0 and 1, to speed up training and help the model converge faster.

## Training the K-Nearest Neighbors Classifier
Now that all of our data is ready, we can go on to training the classifier. K-Nearest Neighbors Classifier is an algorithm where each data point is grouped based on their neighbor's labels. For example, if data point x is close to five other data points which are labelled 2, the classifeir will label data point x as 2 as well.

![a](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/KNN_final1_ibdm8a.png)

Especially for this dataset, where each number is fairly distinct and different, KNN classifier should work well, as it can create clear distinctions between the different groups and points. There are still a couple parameters we have to specify, though.

First, we have the n_neighbors parameter, which denotes how many neighbors we should compare each point with. Remember to keep in mind how large your training data is and how distinct your labels are, as too many neighbors many result in large influences from wrong labels and groups.

![b](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/03faee64-e85e-4ea0-a2b4-e5964949e2d1/d99b9a4d-618c-45f0-86d1-388bdf852c1d/images/screenshot.gif)

I also defined the weight parameter, which dictates how much of an impact each neighbor has on the selection of the final label. By default, the weight is set to "uniform", which means every point is weighted the same, but I will be using the "distance" weight, as closer points will have a larger impact on the point than farther away points.

![Capture(19)](https://user-images.githubusercontent.com/69808907/132283570-77e642ad-7cdb-49b4-b1e5-8a4e2429d699.PNG)

Seems like this model trained really well! Let's try out some different models before we decide on this one.

