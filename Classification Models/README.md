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

## Training the XGBoost Classifier
XGBoost, otherwise known as eXtreme Gradient Boosting, is a great resource to train gradient boosted decision trees fast and accurately. It has an option to use either trees or linear models, and there are several other parameters for you to choose from. For this model, I will be using a decision tree because I am doing a classification task.

Decision trees are "flowchart-like" structures, where each node represents a "test" on an attribute, and each branch representing a different outcome. Each attribute will help differentiate each input from the next, until ultimately they reach the final layer, which is their class.

![c](https://www.cs.cmu.edu/~bhiksha/courses/10-601/decisiontrees/DT.png)

For this data, the tree method should work out pretty well, especially because each number will cover different pixels, so different combinations of pixels should be a pretty strong way to differentiate between the labels. Most of the hyperparameters needed to train a XGBoost model will be discovered through extensive hyperparameter testing and optimization, so I will just train the base model for now. The main parameter to keep in mind is the booster parameter, where you decide between a tree or linear model, but the default is set to tree so I will not have to specify anything.

![d](https://images.slideplayer.com/34/8334889/slides/slide_2.jpg)

![Capture(20)](https://user-images.githubusercontent.com/69808907/132283702-718c3f42-dc66-4a87-9238-a05be826ac2a.PNG)

Great! This model seemed to outperform the KNN Classifier by a little bit. Time to try out one more choice.

## Training the Naive Bayes Classifier
Finally, our last classifier will be a Naive Bayes Classifier. Naive Bayes is a probabilistic model, which uses probability to make its classification choices. It applies the Bayes Theorm, which describes the probability of an event based on prior infromation, with a strong assumption of independence between the features. This makes Naive Bayes probably not the best classifier for this kind of problem, since each number relies on combinations of other features, but it is worth a try nonetheless. Naive Bayes Classifier can be coupled with Kernel Density Estimation, creating a much more complex model, but I will be training their base models for now.

I will be training a Multinomial, Gaussian, and Bernoulli Naive Bayes Classifier. First, the Multinomial classifier is a Naive Bayes model based off a Multinomial distribution. Here, the model is able to handle discrete values better and are used more when there are integer values rather than boolean.

![e](https://blogs.sas.com/content/iml/files/2013/08/multinomial.png)

![Capture(21)](https://user-images.githubusercontent.com/69808907/132283821-40dff099-752c-432c-971f-1ba869997c27.PNG)

Gaussian distribution is similar except it uses the Gaussian distrubtion to predict. Therefore, the distribution is preset, regardless of the data, and using the normal Gaussian distribution, the Naive Bayes model is able to make its predictions.

![f](https://miro.medium.com/max/24000/1*IdGgdrY_n_9_YfkaCh-dag.png)

![Capture(22)](https://user-images.githubusercontent.com/69808907/132283881-947252a4-8c8a-4b16-aea9-08fad138368a.PNG)

Finally, I will be training a Bernoulli Naive Bayes Classifer. This basically implements a binomial distribution, which is similar to the multinomial distribution, except it uses 0 and 1s in a boolean fashion. Therefore, values are either present or not, whereas the count of the value is obsolete. This model should technically be the best one to use on our model.

![g](https://www.mathworks.com/help/examples/stats/win64/CompareBinomialAndNormalDistributionPdfsExample_01.png)

![Capture(23)](https://user-images.githubusercontent.com/69808907/132283954-7406e709-9218-48a3-b84e-f985cd33a67a.PNG)

Yikes, this classifier performed way worse than the other two. Although the Bernoulli Classifier was supposed to be the best, it only barely outperformed the multinomial classifier, and still way off the XGBoost Classifier, which seems to be our best and our submission model.

