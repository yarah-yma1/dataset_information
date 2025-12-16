# Iris Dataset

## Background

The Iris dataset was originally collected by botanist Edgar Anderson, who measured morphological variations in three species of iris flowers (_Iris setosa, Iris versicolor, and Iris virginica_) in the Gaspe Peninsula. The dataset is composed of measurements of sepal length, sepal width, petal length, and petal width, all recorded in centimeters. In 1936, statistician Ronald A. Fisher used Anderson's data in his paper "_The Use of Multiple Measurements in Taxonomic Problems_" to illustrate multivariate discriminant analysis methods, which is another way of saying that he was finding linear combinations of the measurements to best separate the different species. It is appropriate to therefore credit Anderson for the dataset and Fisher for the statistical analysis [2]. 

![alt_text](/media/picIRIS/IrisDatasetPic.png)

## Dataset Structure

The dataset contains 150 samples, 5 attributes, 4 numerical features, and 1 categorical species label. There are 3 classes, which are the iris species, with 50 samples each, so the data is balanced among the species [1]. The features include:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)
- Species (_Iris setosa, Iris versicolor, and Iris virginica_)

The dataset does not have missing values and is stored in simple tabular form. It can be loaded from Python's scikit-learn or loaded in via dataset.

```
from sklearn import datasets

iris = datasets.load_iris()
```

## Summary Statistics
The full statistics per class will vary based on which dataset version is used, but a widely reported summary of all 150 samples gives the means as follows:
- Sepal length (~5.843 cm)
- Sepal width (~3.057 cm)
- Petal length (~3.758 cm)
- Petal width (~1.199 cm)

By the classes, the approximate average values and qualitative notes are as follows:
- _Iris setosa_: smaller petals and sepals overall, quite distinct in petal measurements
- _Iris versicolor_: intermediate sizes between setosa and virginica
- _Iris virginica_: larger sepal and petal lengths and widths compared to the other two

| Species  | Sepal Length | Sepal Width  | Petal Length | Petal Width |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| _Iris setosa_  | ~ 5 cm  | ~ 3.4 cm | ~ 1.46 cm | ~ 0.24 cm |
| _Iris versicolor_  | ~ 5.9 cm  | ~ 2.77 cm | ~ 4.26 cm | ~ 1.33 cm |
| _Iris virginica_  | ~ 6.59 cm  | ~ 2.97 cm | ~ 5.55 cm | ~ 2.03 cm |

These centroids show that setosa is clearly separated in petal size from the other two. Versicolor and virginica are closer together in feature space hence the overlaps [3]. 

![alt_text](/media/picIRIS/IRISStatistics.png)

## Data Geometry, Separability, and Trends

In the original 4D numeric feature space of sepals and petals, one key observation is that the setosa class is almost linearly separable from the other two species while versicolor and virginica are not linearly separable from each other without error [3].

When plotting sepal length vs sepal width, the clustering is weaker and hence overlaps exist mostly between versicolor and virginica. When plotting petal length vs petal width, setosa forms a tight and distinct cluster (of small petal sizes) separated from the other species while versicolor and virginica overlaps more in this space. The class centroids show that setosa's centroid lies somewhere far from the other two. The distance between the centroids of versicolor and virginica is smaller in comparison, and their in-class scatter overlaps with each other. Due to this overlap, a simple linear boundary can almost perfectly separate setosa vs versicolor and virginica, but it cannot split versicolor and virginica well. This is why Fisher's linear discriminant method yielded good results for the setosa-veriscolor case but had more difficulty with virginica [3]. 

Petal length and width tend to discriminate species more strongly than sepal dimensions. The in-class scatter of versicolor and virginica is larger relative to setosa's separation. In many tutorials, the scatterplot or PCA of the first two principle components shows setosa is well separated while the other two overlap substantially [3].

## Known Issues, Limitations, and Caveats
### Multiple versions and Transcription errors
- The version typically used in machine learning libraries differs slightly from Fisher's original paper as the UCI Machine Learning Repository has documented errors [4].

### Limited size and simplicity
- The dataset only has 150 samples and 4 numeric features, so the dataset is quite small by modern standards. It is often too easy for many classification methods and may not exercise more complex modelling challenges like noise, missing data, or heterogeneity [1].

### Representative Limitations
- The dataset is a clean, balanced, and measured under controlled conditions, so it does not fully reflect more complex considerations of the real world such as class imbalances or missing features [4].

## Summary
The Iris dataset is a fundamental dataset that is widely used in teaching and benchmarking classification and discriminant analysis methods because of its simple yet instructive structure. With 150 samples at 50 samples per class, 4 numeric features, and 3 Iris specieis, it offers a clear demonstration of how feature measurements can distinguish biological categories [2]. The _Iris setosa_ specieis is fairly well separated while _Iris virginica_ and _Iris versicolor_ overlap. These observations illustrate limitations of linear separability and the need for careful analysis of the class's geometry. While this dataset is highly convenient and popular, users should be aware that its small size, simplicity, and version inconsistencies can impact results if it is used as a benchmark for advanced tasks [1]. 

## Citations
1. [1] “The Iris Dataset,” scikit-learn, 2024. https://scikit-learn.org/1.4/auto_examples/datasets/plot_iris_dataset.html
2. [2] R. A. FISHER, “THE USE OF MULTIPLE MEASUREMENTS IN TAXONOMIC PROBLEMS,” Annals of Eugenics, vol. 7, no. 2, pp. 179–188, Sep. 1936, doi: https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
3. [3] R. A. Fisher, “UCI Machine Learning Repository,” archive.ics.uci.edu, 1936. https://archive.ics.uci.edu/dataset/53/iris
4. [4] E. Anderson, The irises of the gaspe peninsula, Bulletin of the American Iris Society, vol.59, pp.2-5, 1935, https://wiki.irises.org/pub/Hist/Info1986SIGNA37/SIGNA_37.pdf
