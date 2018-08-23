import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
# first: instances(rows), second: attributes(columns)
print(dataset.shape)
print(dataset.head(20))
# contains mean, std, min, count, percentiles
# conclusion: same scale and similar ranges
print(dataset.describe())
# method to count instances
# conclusion: each class contains one third of the instances
print(dataset.groupby('class').size())

# univariate plots: to better understand each attribute
# multivariate plots: to better understand the relationships between attributes

# UNIVARIATE:
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# shows gaussian distribution (= normalverteilung)
dataset.hist()

# MULTIVARIATE
# scatter plot matrix
# if there is diagonal grouping between some pairs of attributes, it suggests
# high correlation and a predictable relationship
scatter_matrix(dataset)
# plt.show()

# split dataset with 80% for training and 20% for validation
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
# (ration of correctly predicted instances divided by the total number of instances) multiplied by 100
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# 10-fold cross validation: split datset into 10 parts, train on 9 and test on 1
# do this for all possible combinations

# excursion: randomness in machine learning
# randomness in data: the data may have errors and may contain noise that makes the relation between inputs and outputs less clear
# randomness in evaluation: we work with only a small subset of the data --> use k-fold cross validation --> to see how the model works on average
# randomness in algorithms: algorithms use randomness to generalize the borader problem; they are often called stochastic algorithms
# pseudorandom number generators are not really random. They use a seed and determine random-looking numbers in a deterministic process
# if the generator is not explicitly seeded, it may use the current system time in seconds or milliseconds
# the value of the seed does not matter, but the same seeding will result in the same sequence of random numbers
