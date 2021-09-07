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

## Image Blurring
Next, we will take a look at image blurring. Image blurring is a way of to reduce the detail and noise in an image, making it more blurry, but helping reduce overfitting and improve generalization in training deep learning models. Small, minute details in certain images may cause the models to depend on those details, making them ineffective against variation in other images. We will use OpenCV in order to apply averaging, Gaussian filtering, median filtering, and bilateral filtering.

### Averaging
Averaging is done by convoling an image with a normalized box filter, by taking the mean of the pixels in the kernel area and replacing the middle/central element. For example, a 3x3 normalized box filter would look like this.

![Capture(2)](https://user-images.githubusercontent.com/69808907/132281110-0e494a88-cc16-4da9-8e9b-51dd2a43c6a7.PNG)

The box filter's width and height can be changed in the blur function, where a bigger box filter would lead to higher generationlization and a greater loss in higher level details.

![Capture(3)](https://user-images.githubusercontent.com/69808907/132281150-dbd9f0aa-5649-4900-9aca-2bf1d90ca677.PNG)

### Gaussian Filtering
In Gaussian Filtering, instead of using a normalized box filter, a Gaussian kernel is used instead. This method is especially effective in removing Gaussian noise, which is noise that has a probability density function equal to the normal distribution.

Here, the width and height are specificed again (But this time they have to be odd), and the standard deviation must be specified.

![Capture(4)](https://user-images.githubusercontent.com/69808907/132281212-a0747add-13c3-4c61-bc2f-7c5bfa24fb4e.PNG)

![Capture(5)](https://user-images.githubusercontent.com/69808907/132281241-b5ad6f24-e54b-49f1-a615-d4a27554b477.PNG)

### Median Filtering
Median filtering, which is very similar to averaging, changes the central element of the kernel area to the median of the values in the kernel space. This is very effective against salt-and-pepper noise and the kernel size should always be a positive odd number.

![Capture(6)](https://user-images.githubusercontent.com/69808907/132281289-77bdf765-54a9-4499-a549-8261a7d3bfee.PNG)

Note: The image passing through the medianBlur function must be of dtype float32.

![Capture(7)](https://user-images.githubusercontent.com/69808907/132281334-74d1a083-c24d-470d-a764-31f1599d43a0.PNG)

### Bilateral Filtering
And finally, bilateral filtering utilizes Gaussian filtering twice in order to preserve edge detail while also effectively removing noise. First, a Gaussian filter is taken in space, but a second one is taken as a function of the pixel difference. The first Gaussian function ensures only nearby pixels are blurred, while the second Gaussian function ensures that only pixels whose values are close to the central element are blurred, rather than elements with greater differences, which could indicate an edge.

![Capture(8)](https://user-images.githubusercontent.com/69808907/132281368-a5409c65-efa9-4463-b4cf-74460438b596.PNG)

![Capture(9)](https://user-images.githubusercontent.com/69808907/132281393-56c3c64a-4b52-4426-a1ea-c83074cfabbe.PNG)

## Morphological Operations
Next, we will discuss morphological operations, which are a collection of nonlinear operations that deal with the shape (or morphology) of the image. These techniques are less concerned with the pixel values, such as the smoothing techniques presented above, rather the relative ordering of the pixel values (According to [Wikipedia](https://en.wikipedia.org/wiki/Mathematical_morphology)). These techniques utilize structuring elements, which are positioned throughout the image at different locations, where the operation figures out the correlation with the structuring elements with its surrounding elements. Some operations test whether they "fit" while others test contrast and "hits".

