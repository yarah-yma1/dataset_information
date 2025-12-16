# CIFAR-10 Dataset

## Background

The CIFAR-10 dataset was introduced by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton in their 2009 paper "_Learning Multiple Layers of Features from Tiny Images_" at the University of Toronto [1]. It is actually labeled as a subset from the much larger "_80 Million Tiny Images_" dataset from 2008 as released by Torralba, Fergus, and Freeman. The point of creating this subset was to select a manageable, well-labeled subset of small color images (32 x 32) that covered a limited number of object categories which is later used to facilitate benchmarks in image classification.

The creators manualled curated labels rather than utilizing the labels that already exist from the _Tiny Images_ dataset. The results include a clean, 10 fixed object classes dataset that covers a variety of categories in small image resolutions. 

Over time, CIFAR-10 became one of the most known benchmarks used for computer vision and deep learning in research, teaching, and evaluation. The small resolution size and accessibility helps it be the middle man for toy datasets and larger high resolution datasets [6].

## Dataset Structure

CIFAR-10 consists of 60,000 color images of size 32 x 32 pixels with three color channels (RGB). There are 10 classes with each class containing 6,000 images. The dataset is typically split into a training set of 50,000 images and a test set of 10,000 images.

![alt_text](/media/picCIFAR10/CIFAR1032by32.png)

The test set is organized so that there are exactly 1,000 images per class. The training set is typically divided into 5 batches of 10,000 images each for convenience in provided formats. Of the 5 training batches, each class is represented by 5,000 images so the training set is balanced across all classes. The training batches are randomized, so the individual batches may not be perfectly balanced.

In most software frameworks such as Keras or Pytorch, the images are stored as 8-bit unsigned integers ranging from 0-255 per channel with the labels as integers ranging 0-9 indicating the class. The names of these class in order are:
```
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

Because of the fixed size and data format, many libraries provide built-in loaders [6].
```
tf.keras.datasets.cifar10

or

torchvision.datasets.CIFAR10
```

Due to how small the CIFAR-10 images are and their relative low detail, there is a limit to how fine distinctions can be made visually. Classes like airplane, truck, and ship may be harder to distinguish.

![alt_text](/media/picCIFAR10/cifar10examples.png)

## Data Trends, Seperability, Overlap, Center of Mass

Image datasets like CIFAR-10 live in high dimensional pixel space (32 x 32 x 3 = 3,072 dimensions). In that space, linear separability between arbitrary classes is unlikely unless features, or learned projections, collapse the dimension [3]. In raw pixel space, the different classes overlap heavily. For example, the pixel distributions for cat and dog may share a lot of statistical structure such as backgrounds, textures, and colors. The classes are not linearly separable in raw input space.

Instead, we can look at centroids (mean image per class) in pixel space. The centroid for each class is the mean over images in that class and typically reveal a blurry class prototype. For example, one may see the wings of an airplane under the airplane class or a long neck for the horse class. The intercentroid distances (distance in pixel space) reflect how far apart the classes are on average. However, the variation around a centroid is large in this case. The signal to noise ratio, or distance between centroids relative to the class scatter, is also modest.

In many visualizations of this dataset, some classes cluster more tightly than others such as frog, ship, or airplane while other classes are more spread like dog and cat because of intraclass diversity derived from poses or backgrounds. When projecting into 2D space, there are overlapping clouds for visually similar classes such as cats and dogs where dissimilar classes might be in more distinct clusters like airplane and ship [4]. 

A study in 2020 claims about 3.3% of test images in the CIFAR-10 dataset have duplicates in the training set that are easily recognizable by memorization. This will add bias when comparing image recognition techniques regarding their generalization capability. Barz and Denzler introduces a new dataset called the "fair CIFAR," or (ciFAIR) dataset, in order to eliminate the bias. The duplicates in the test sets are replaces with new images sampled from the same domain. After evaluating the new dataset for classification performance with variouse CNN architectures, Barz and Denzler were able to investigate whether recent research has overfitted into memorizing data instead of learning abstract concepts. They discovered a 9% drop in classification accuracy relative to the original performance of the CIFAR-10 dataset [2, p. 1].

Due to its wide usage, research shows that many methods of evaluating this dataset overfits to the anomalies of CIFAR-10 such as exploited duplicate images rather than learning generalizable features [2]. 

## Known Issues, Limiations, and Caveats
### Low Resolution and Limited Information 
- 32 x 32 is very small, so many fine visual distinctions such as textures and fine shape are lost. This limits the upper limits of testing a model in comparison to a higher resolution dataset [3].

### Label Noise
- About 0.5% of images in the test set are mislabeled [3]

### Duplicate and Near Duplicate Images
- The overlap between test and train sets mean models might "memorize" instead of generalize [2].

### Overfitting to Benchmark
- Due to CIFAR-10 being used as a benchmark heavily, many models are implicitly tuned for it by gearing the model towards the dataset's specific quirks rather than general methodologies [2].

### Unrepresentative of Real Data
- Real image datasets have variable image size, aspect ratios, types of backgrounds, noise, imbalanced classes, and unlabeled data. CIFAR-10 is relatively clean, well labeled, and balanced, so it will be easier to train than real data [3].

## Summary
CIFAR-10 remains a fundamental benchmark moderate in size and simple enough to work with quickly. It comes from the "_80 Million Tiny Images_" dataset, but it is hand-labeled and filtered to be cleaner to read [8]. The point of creating this subset was to select a manageable, well-labeled subset of small color images (32 x 32) that covered a limited number of object categories which is later used to facilitate benchmarks in image classification. The results include a clean, 10 fixed object classes dataset that covers a variety of categories in small image resolutions. Over time, CIFAR-10 became one of the most known benchmarks used for computer vision and deep learning in research, teaching, and evaluation [6]. The small resolution size and accessibility helps it be the middle man for toy datasets and larger high resolution datasets.In raw pixel space, classes are not linearly separable. These separations must rely on learned nonlinear transformations or embeddings. The dataset's balance and formatting makes it convenient, but it has known duplicates, label noise, and low resolution that limits how representative is is of real world tasks. When using CIFAR-10, it is good as a benchmark but the user should be mindful that the performance may easily saturate. When working with this dataset, one must distinguish whether the code is evaluating generalization or overfitting to the dataset. 

## Citations

1. [1] A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” Apr. 2009. Available: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
2. [2] B. Barz and J. Denzler, “Do We Train on Test Data? Purging CIFAR of Near-Duplicates,” Journal of Imaging, vol. 6, no. 6, p. 41, Jun. 2020, doi: https://doi.org/10.3390/jimaging6060041.
3. [3] Vignya Durvasula, “CIFAR 10 Dataset: Everything You Need To Know - AskPython,” AskPython, Jan. 22, 2024. https://www.askpython.com/python/examples/cifar-10-dataset (accessed Oct. 21, 2025).
4. [4] GeeksforGeeks, “CIFAR10 Image Classification in TensorFlow,” GeeksforGeeks, Apr. 29, 2021. https://www.geeksforgeeks.org/deep-learning/cifar-10-image-classification-in-tensorflow/
5. [5] franky, “Once Upon a Time in CIFAR-10,” Medium, Sep. 02, 2022. https://franky07724-57962.medium.com/once-upon-a-time-in-cifar-10-c26bb056b4ce
6. [6] K. Team, “Keras documentation: CIFAR10 small images classification dataset,” keras.io. https://keras.io/api/datasets/cifar10/
7. [7] Toronto, “cifar10,” Huggingface.co, Mar. 30, 2022. https://huggingface.co/datasets/uoft-cs/cifar10 (accessed Oct. 21, 2025).
8. [8] A. Krizhevsky, “CIFAR-10 and CIFAR-100 datasets,” Toronto.edu, 2009. https://www.cs.toronto.edu/~kriz/cifar.html
