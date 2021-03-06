# Simple Image Preprocessing Techniques Using MNIST Data
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

![b](https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/morph-probing.gif)

With each morphological operation, I will provide examples of a change in kernel size and a change in iterations, in order to encompass those hyperparameters. In general, though, a larger kernel size works in larger steps while larger iterations tend increase the effect of the operation.

### Dilation
First off, dilation monitors "hits" or contrasts to the pixels, and adds an extra layer of pixels to the inner and outer boundaries of the shapes.

![c](https://homepages.inf.ed.ac.uk/rbf/HIPR2/figs/diltbin.gif)

This is dilations with one iteration and a 5x5 one matrix.

![Capture(10)](https://user-images.githubusercontent.com/69808907/132281616-fdae59cb-f2a1-48d8-9f26-529242bc6248.PNG)

This changes the parameters to five iterations.

![Capture(110](https://user-images.githubusercontent.com/69808907/132281654-f0ff859d-4663-4715-9513-bf07cb69133b.PNG)

And finally this changes it back to one iteration but with a 10x10 matrix instead.

![Capture(11)](https://user-images.githubusercontent.com/69808907/132281686-67f331ac-fca5-4d65-8347-8235ab5e80ac.PNG)

### Erosion
Erosion is the opposite of dilation, where it scans for "fits" among the boundaries, and strips a layer from the inner and outer boundaries of the shape. This can be used to sharpen edges or increase constrast between two very similar images.

![d](https://homepages.inf.ed.ac.uk/rbf/HIPR2/figs/erodbin.gif)

This is erosion with one iteration and a 5x5 one matrix.

![Capture(12)](https://user-images.githubusercontent.com/69808907/132281978-d0e48425-8549-45ea-88af-b0ddbae8073c.PNG)

This is erosion with two iterations.

![Capture(13)](https://user-images.githubusercontent.com/69808907/132282008-1d0ff146-3356-4145-bbaa-ccfd1c0c6cea.PNG)

And finally, this is one iteration with a 10x10 matrix.

![Capture(14)](https://user-images.githubusercontent.com/69808907/132282053-9f29d823-4c0c-4232-88a9-b86879c515e7.PNG)

### Compound Operations
There are also examples of compound opeartions in morphology. The two main ones are opening and closing and image, which are a combination of a dilation and an erosion.
- Closing is a way of filling in holes and solidifying images, increasing generalization and decreasing the importance of smaller marks or details. An erosion is performed first and then a dilation is performed.
- Opening is a way of decreasing small details in an image to "open" up larger details and forgo smaller, unimportant details. A dilation is performed first and then an erosion is performed. Any pixels that "survive" after the erosion are restroed after the dilation.

Opening and closing can also be applied, which ultimately closes larger objects and forgoes smaller details that are not connected to the main content of the image.

![e](https://i.ytimg.com/vi/1owu136z1zI/maxresdefault.jpg)

#### Closing

![Capture(15)](https://user-images.githubusercontent.com/69808907/132282182-ce9aa3fe-a5b2-47c6-8c0d-7f3e5d39ed2a.PNG)

#### Opening

![Capture(16)](https://user-images.githubusercontent.com/69808907/132282217-80f28d42-a91e-463b-88b7-8074ffe9ee77.PNG)

## Edge Sharpening
And finally, edge sharpening is a useful tool, especially if your dataset has many labelled images with very similar features and edge sharpening can help make them more defined. Here, edge sharpening uses the function filter2D, by passing the kernel through, which increases the difference between the central element and its surrounding elements, making the distinction. This distinction can be more helpful for image EDAs and understanding the data better or possibly for feature engineering.

![f](https://static.packt-cdn.com/products/9781785283932/graphics/B04554_02_11.jpg)

![Capture(17)](https://user-images.githubusercontent.com/69808907/132282272-e7ff1ddb-5d5e-4afe-96c8-3f11317a55e5.PNG)
