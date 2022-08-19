# Reverse-Image-Search-Fashion-Apparel-Recommendation-System
A simple web app where the user uploads an image of a fashion product they are looking for and the web app returns similar products that exists in our database.

 Reverse Image search Fashion Apparel Recommendation System
Reverse Image Search for the Fashion Industry Using Convolutional Neural Networks


![image](https://user-images.githubusercontent.com/70193389/185605884-2e36761b-f6ea-4e6a-891b-21161a6c3773.png)

 
 
## Brief Description & History

With the dawn of online shopping, it is very easy these days to find anything you wish to buy, even from the comfort of your home. You can easily complete a purchase, and have the goods delivered to you. However, even though all these products are readily available at the stores, sometimes, it can be difficult locating them on the websites to initiate the purchases. Part of the problem inherently lies in the manner through which our search mechanisms have conventionally been carried out. Up until now, the most common means we have used to search for goods is one in which we input text into a search box and then the appropriate listings accompanied by pictures of the products are returned to us as output. Notwithstanding, we sometimes inevitably find ourselves in situations where it is difficult to precisely describe the particular item we wish to find in few enough words to fit into a text field search box . This is especially true in the field of fashion.
Consider an example of when trying to procure a blazer, or a dress of a particular design. There are so many nuances and subtleties that differentiate that blazer from the plethora of others already in existence. Consider trying to describe these all in a text field search box. You may, though, happen to have a similar image to one that you intend to buy, already saved on your phone or laptop. It would be of great convenience, to be able to simply upload this image to the online platform where you are doing your shopping, then have all of the most similar images (in this case, blazers) available on the database of that system,returned to you for you to choose from, with the simple clicks of a few buttons.
 Another practical example of the need for a feature like this is if one were looking through a fashion magazine or if they saw someone throughout their day wearing something they fancied, but did not know the particular name of the clothing item. Quickly snapping and saving a picture which could then be used to search for similar products on online stores would then help this person procure this product for themselves eventually as well. This is the field of computer vision, and it has been extensively explored since the early 1990s . It still attracts a great amount of attention today, and recent efforts in research and integration of using image search technology has played a big role in improving the efficiency of online shopping in the present day.
Early implementations of this particular reverse image search functionality were implemented in websites such as TinEye. which is now one of the largest and most popular reverse image search websites presently, with also one of the largest databases of indexed images. Other examples of websites which utilize this capability are Ditto, Snap Fashion, ViSenze, Cortica, Google Image search, Bing Image Search , Flickr, Yahoo Images, and Getty Images to name a few. Although not all of the aforementioned websites are particularly online stores, their successful implementation of this feature gives it practical validation. A great example of a website that does in effect integrate reverse image searching for online shopping is Taobao, the largest Chinese online shopping company, grossing over billions of dollars in profits each year. Furthermore, with cutting edge technology increasing at an exponential rate and showing no signs of slowing down, high definition photos are becoming commonplace, and companies are in no delay to leverage them. Online stores heavily rely on using images for advertising and showcasing their products to their customers in as much detail as possible. With this rapid increase in the number of existing images on databases and servers, it becomes increasingly difficult and expensive as well to physically annotate every new image that comes to existence. Annotation and tagging are the manual addition of descriptive text and tags to images for retrieval through textual searches .
Recent popular social websites, such as Facebook, Instagram etc., make it possible for users to submit tags with the photos that they upload.. This provides a means of indexing, and allows for the photos to be found later in a much easier fashion. Unfortunately, one of the largest shortcomings of this is that plenty of the user submitted tags are biased to the individuals’ sentiments, as in an example of a person uploading a picture of a cat, and adding a tag that says ‘cute’. 
Although this tag is relevant to the user submitting it, it is not necessarily helpful to another person who may be browsing the system looking for pictures of cats. Hence from all this we can see, it is with ever increasing importance that effective systems and technologies capable of automatically sifting through and organizing images, using more sophisticated technologies, become readily available as well .







 
 
 
 
 
 
Reverse Image search Fashion Apparel Recommendation System
Reverse Image Search for the Fashion Industry Using Convolutional Neural Networks
Approach to the problem:
Defining the problem statement
Requirement Analysis/ Understanding the working of project
Collecting the data
Import model
Feature extraction
Export Model
Making of web app
Generate recommendations
 
## 1. Defining the problem statement ##
Create a web app which shows similar fashion products to the product whose image the user uploaded. 
 
## 2. Requirement Analysis/Understanding the working of the project ##
 
In the technical world of computing, reverse image search has been referred to as Content Based Image Retrieval(CBIR). 
 
This simply means that images are searched for and retrieved, using only the contents of the image, without the assistance of any external labels or metatags. A picture is usually broken down into its defining features, sometimes referred to as feature vectors. The measure of level of feature extraction determines the image content level. The information inherently obtained from an image can be categorized into three levels.
 
**Low level features** - Shape, texture, color, spatial information and motion are considered as low level. 
**Middle level features** - Organization of different types of objects, scenes and roles represent this category.
**High level features** - Include objects or scenes with emotional or religious significance impressions, meaning associated with the combination of perceptual visual contents. 
 
Such features are obtained by applying mathematical algorithms to the image, and as such there are a number of choices as to which one may use. Generally speaking, any image search system can be broken down into four important and distinctive components, which are;
 
 Defining which feature you want to extract from the images, e.g. shape, color, texture etc. This also entails defining the algorithm you will use to do so.
 
 Saving and indexing your features – after you have chosen which algorithm to use, you must apply it to all of the images in your dataset, and then index and save them in the appropriate database.
 
 Application of a distance metric – Now that the list of features is obtained, we must further apply  another mathematical function to determine which features are closest in similarity, and hence this ultimately reveals which images are most similar.
 
Some examples of distance metrics are hamming distance, cosine distance , Euclidean distance, k means clustering  etc. I used Euclidean distance in this project.
 
 Returning and displaying the best matches to the user – after the most similar features are determined, the images that they belong to are then retrieved once again and displayed back to the user in an orderly
 
 
 
 

## Feature extraction ##
 
  ![image](https://user-images.githubusercontent.com/70193389/185606030-8a7a2a4e-6758-4fe6-8e19-352ba6666041.png)
  
  

                                           
                                          

 
 
 
 
 
 
 
 
 
 
 
 
### Theory behind CNN ###
 
A CNN learns how to break down an image and automatically learns which features are important to extract. All CNNs can be divided into a number of important building blocks. Here below they are listed and explained in more detail along with the configurations that we used for
our network.

![image](https://user-images.githubusercontent.com/70193389/185606098-f66f848e-84ae-4ae9-807b-f2f6fba243a9.png)


### Core components a CNN network ###

![image](https://user-images.githubusercontent.com/70193389/185606172-0075fd8e-96eb-4773-b44d-a356f6a55612.png)

**CNN feature detector**


A number of feature detectors are passed over the image, and a set of feature maps obtained as a result. These are the actual image pixels of an image, hence in effect the size of the image is reduced allowing for less computational expense, and while yet still preserving the important
information and spatial information of an image. This also allows for the defining features of specific classes to still be detected even if those objects were to appear in different positions within an image, e.g. bottom left corner or top right corner. 

**Pooling Layer – this layer further reduces the size of the feature maps, further discarding irrelevant information from the image where features are not detected and selecting
only the regions with the highest feature occurrences. This also assists in accounting for slight distortions or variances amongst different objects of the same class, i.e. different
objects of the same class will have approximately similar pooled feature maps. An illustration of the pooling process is shown

![image](https://user-images.githubusercontent.com/70193389/185606289-5a967aeb-4732-4b97-bd52-6ba5717f1367.png)

Max Pooling operation


After getting the output from the model ,we will flatten it and normalize it and save and export the model using the pickle module in python for later use (for web app).

For the web application , we will use the Streamlit module in python.
Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. 

To generate recommendations, we look for the feature vectors which have the least euclidean distance from the feature vector of the target image.

Finally, we display the 5 vectors/images with the least Euclidean distance.
 
 
 
 
## 3. Collecting the Data ##
The data used in this was obtained from kaggle. The dataset includes 44,000 high resolution images of fashion products. This dataset was used to build a reverse image search engine to create an image based recommendation system.



 A small portion of the dataset
        
   ![image](https://user-images.githubusercontent.com/70193389/185606338-fc365862-2182-4c88-b516-a51870a1430e.png)


Dataset link: https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset
 
4. ## Import Model/ Feature Selection/ Generation of recommendations
The model I used was ResNet-50 is a convolutional neural network that is 50 layers deep. It is a very popular pre-trained neural network . It has been trained on more than a million images from the ImageNet database.
 
The network can take the input image having height, width as multiples of 32 and 3 as channel width. Before using ResNet 50 , we convert our images to dimensions of 224 X 224 (normally the standard value but we can modify according to our needs). The input size is 224 ,224 ,3.
 
So, the shape of the numpy array we use is (224 ,224, 3). But in order for Keras to process this array, we need to convert this array into a 4D matrix (1,224,224,3) as Keras works on a batch of images instead of just a single image.
 
Then we pass this array to the preprocessed input of ResNet which converts the input to the required input form, the form which was used for training the images in the ImageNet dataset originally.
 
After taking the input , the Resnet model  then predicts and returns a vector/matrix with 2048 dimensions/features. 
 
One image represents a vector having 2048 features/dimensions. So ,our complete dataset has 44,000 vectors with each vector having 2048 features.
 
After getting the output from the model ,we flatten it and normalize it and save and export the model using the pickle module in python for later use (for web app).


## 4. Making of web app/Generating  Recommendations ##
Finally, for the web application we write some new code and  reuse some code.

I used the streamlit module in python. Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.

First, the user uploads the image of the product or similar products they are looking for.

The pillow module is used to open/access the image. The image is then stored in the uploads folder where all the images that the user has uploaded are stored.

Then we extract features from the image uploaded and get a feature vector and compare it to the other feature vector we had in our dataset using the pickle file we had saved earlier.

To generate recommendations, we look for the feature vectors which have the least euclidean distance from the feature vector of the target image.

We use the Nearest Neighbours algorithm to calculate the most similar products for this. 

Finally, we display the 5 vectors/images with the least Euclidean distance.

