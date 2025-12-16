# Fashion_MNIST Data Set
This project demonstrates the Keras deep learning library that helps create an artificial neural network built on Tensorflow.
# Table Of Contents
- [Origin + Background](#origin+background)
- [Composition](#composition)
- [Issues](#issues)
- [Statistics](#statistics)
- [References](#references)
## Origin + Background 
- The fashion MNIST was introducted by Zalando Reserch in 2017 [5]. The authors were Han Xiao, Kashif Rasul, and Roland Vollgraf. 
- It was made to have the same concept as the MNIST handwritten dataset with same image sizing of 28 times 28, same train + testing splits w/ 60,000/10,0000, and 10 classes. The difference is the content in which the FashionMNIST is classifying fashion items instead of digits [5]. 
- This was also made to help idenitfy and test the classification of complex images [5].
- The images of the clothing comes from clothing at Zalando which is a European retailer. The original image sizings were processed and resized + converted to fit a 28 times 28 greyscale image formatting with labeling. Formatting matched MNIST for easy use of reuse of code written for MNIST [2]. 
## Composition 
- 70,000 Images --> 60,000 for training and 10,000 for testing [1]. 
- Property of Images --> Greyscale images with 28 times 28 pixels. Total of 784 pixels. Each pixel has a single pixel value, indicating the lightness or darkness of that pixel. The higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column is class labels and the rest of columns are the pixel values for the images [1]. 
- Number of classes --> 10 classes.
#### The classes are as follows:
- 0 --> T-shirt/Top
- 2 --> Trouser
- 3 --> Pullover
- 4 --> Dress
- 5 --> Sandal
- 6 --> Shirt
- 7 --> Sneaker
- 8 --> Bag
- 9 --> Ankle Boot
#### Each class has aprox. the same amount --> 7,000 per class. The exact distrubtion for training and testing may vary [1].
## Issues 
- There can be some label errors or miscategorized items in the dataset [6].
- Cleanlab Studio suggested the t-shirt/top label to more than 100 images labeled as shirt, and suggested the dress label to more than 40 images labeled as shirt. This may have lead to inconsistent model training [6].
- They also mentioned that some footwear images annotated with the incorrect category [6].
## Statistics
- Training and Testing is a 60,000/10,000 split [1].
- It spans across 10 classes.
- The images show items have distigushing features hoever it may overlap if identifying though pixelations. There are mentods like PCA that show clusters for the classes, howvere it is not perfectly linearly separable and alot of the images are near the boundary of the clusters which represents overlap [3].
- Some classes are more easliey and linearly separable by features, but others may need non-linear feature extraction because some classes may be confused for other classes [3]. 
- In terms of center of mass, taking the mean of all the pixels across the images of a class, the average would should the outline of the silhouette of the class [3].
## References
- [1]“Fashion MNIST,” www.kaggle.com. https://www.kaggle.com/datasets/zalando-research/fashionmnist
- [2]zalandoresearch, “zalandoresearch/fashion-mnist,” GitHub, Nov. 15, 2020. https://github.com/zalandoresearch/fashion-mnist
- [3]Ksaraswat, “Classification of Fashion MNIST Dataset Using DNN and CNN,” Medium, Nov. 23, 2023. https://medium.com/%40ksaraswat_97923/fashion-mnist-image-classification-deep-learning-using-dnn-and-cnn-675f786d1b0d (accessed Oct. 21, 2025).
- [4]Ultralytics, “Fashion-MNIST,” Ultralytics.com, 2023. https://docs.ultralytics.com/datasets/classify/fashion-mnist/#applications (accessed Oct. 21, 2025).
- [5]H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms,” arXiv:1708.07747 [cs, stat], Sep. 2017, Available: https://arxiv.org/abs/1708.07747
- [6]“The Fashion MNIST Dataset (cited in 2,200+ papers) contains Hundreds of Miscategorized Items,” Cleanlab, 2023. https://cleanlab.ai/blog/csa/csa-4/ (accessed Oct. 21, 2025).
‌

