# Simple Image Preprocessing Techniques Using MINST Data
Image preprocessing is an important step in any computer vision task, especially as the data becomes more complicated, noisy, and diverse. Good preprocessing could make the difference between a good model and an excellent model, impacting the final score greatly. Below, I will go over several very simple preprocessing techniques using the MINST digits dataset. (Disclaimer: By no means are these the ONLY preprocessing techniques available.)
- Importing Libraries and Setting Up the Dataframe
- Plotting Functions and Resizing Images
- Image Normalization
- Image Blurring
- Morphological Operations
- Edge Sharpening

For the preprocessed images, I will display the original four first and then the preprocessed four.

## Importing Libraries and Establishing the Dataframe
First, I imported the necessary libraries that I will be using. These include:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- cv2
- Tensorflow.keras

I also created a new Dataframe which is specifically a sample of 4 of each number, creating a dataset with 40 numbers with 783 pixels representing each one. Each of the pixels has a value of 1 or 0 indicating if its white or not. 

![Capture](https://user-images.githubusercontent.com/69808907/132280742-37180a4d-6e4b-4acf-b5df-49b9dea400e4.PNG)

## Plotting Functions and Resizing Images
Next I created a few functions that will help me with my project. They are as listed below:
- display_one: Used to display one sample of one number
- display: Used to display all the images in the dataset
- display_change: Prints out 8 images per number, first four are the original four samples, last four are the images after preprocessing
- size: Resizes pixels into arrays for matplotlib to read and plot

![Capture(1)](https://user-images.githubusercontent.com/69808907/132280904-87bedbb3-a711-402b-a1d4-0a7b2bd2fe61.PNG)

## Image Normalization
Time to get started! Normalizing image arrays between 0 and 1 is very beneficial when training deep learning models, because it helps the models converge and train faster.

![Capture(20](https://user-images.githubusercontent.com/69808907/132280948-67b971a1-05e8-49ef-a117-68b240dd02e4.PNG)

