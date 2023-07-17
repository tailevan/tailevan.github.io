---
layout: mathjax
title:  High Dimensional Data Analysis with Geomics Data
date:   2023-06-01
---
# Background 
- We will analyze a RNA dataset to figure out the hierarchical structure and discovering important genes. 
- The datasets provided are high dimensional data with each row corresponds to a cell, each column is a level of expression of the i-th cell

# Visualization - Dimension Reduction

## Principle Component Analysis (PCA)
- The visualization of the PCA which  project the high dimensional data into two principal components which preserve its highest variances.
```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
X_data = np.load("p2_unsupervised/X.npy")
X_log_tranformed = np.log2(X_data + 1)
pca = PCA()
z = pca.fit_transform(X_log_tranformed)
# Plot 
plt.scatter(z[:,0], z[:,1])
plt.title("PCA of the log trasformation data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

- ![PCA](/images/PCA-figure-of-log-tranformed-data.png)
	- The data is most dispersed along two main directions, which are captured by the principal components. The plot displays the data in terms of these components.
	
	- The plot shows three distinct clusters of data points, corresponding to three different cell types. The data points within each cluster are close together, while the data points between clusters are far apart. This indicates that the cell types are very different from each other in terms of the variance captured by the principal components.



- The Explained Variance by component plot expresses that the majority of the variance concentrated on the around first 20 components
	- ![elbow_method](/images/Explained_variance_by_component.png)
```python
plt.plot(np.arange(1,101),pca.explained_variance_ratio_[0:100])
plt.title("% Explained variance by component",size=18)
plt.xlabel("Component #",size=14)
plt.ylabel("% Variance Explained",size=14)
plt.show()
```


- The visualization of the TSNE on the first 50 PCs the data with perplexity = 40
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=50, verbose=1)
ztsne = tsne.fit_transform(z[:,:50])
plt.scatter(ztsne[:,0], ztsne[:,1])
plt.title("TSNE Plot, perplexity = 50")
plt.axis("equal")
```
![TSNE](/images/TSNE-Plot.png)
<mark> visualization shows numerous clusters</mark>
- The TSNE plot shows how data points with similar features are grouped together. TSNE uses a probability-based method to reduce the dimensionality of the data and preserve its structure. It tries to make the distances between points in the plot match their probabilities of being neighbors in the original data <mark> use TSNE </mark> 

- It should be chosen based on the size and complexity of the data. A common range for perplexity is 5 to 50, so I chose 40 as a reasonable value.
- There are many clusters in the TSNE plot this shows that there are numerous sub-types for a cell. <mark> weak explain </mark>

- I used KMeans with 3 clusters to assign labels to the sub-type cells based on their PCA plot, which shows the 3 main cell types. Then I plotted the TSNE with the label information to visualize how the sub-type cells are distributed. <mark> show different clusters fall into 3 groups </mark>
```python
from sklearn.cluster import KMeans
# Cluster the 3 main type and label it in y variable
kmeans = KMeans(n_clusters=3, n_init=10)
y = kmeans.fit_predict(z)
# Plot the TSNE plot with the label of type of cells
plt.scatter(ztsne[:,0], ztsne[:,1], c=y)
plt.title("TSNE Plot, perplexity = 50")
plt.axis("equal")
```
- ![TNSE](/images/TSNE-plot-with-label-color-code-for-cell-types.png)
- The PCA plot with the label information from KMeans 3 clusters
	- ![PCA](/images/PCA-plot-with-label.png)

# Unsupervised Feature Selection

- The TSNE plot shows about **30 clusters** that can be visually identified by how close the data points are within each cluster and how far they are from other clusters.
	- ![TSNE](/images/TSNE-Plot.png)


- Perform logistic regression with cross-validation and L1 regularization on a dataset of features and labels. It creates a model, fits it to the training data, and evaluates its accuracy score.
- There are several parameter in the logistic regression
	- As our data is in high dimension, large, and sparse, there `solver` should be `saga`, or `sag`
	- And the penalty is `l1, l2, elastic`
	- The regularization `C`, to be selected by the Cross-Validation in the `LogistricRegressionCV` function
- So to choose the best solver and the penalty, we should use the grid search with logistic regression

- After that, we could obtain the best `LogisticRegressionCV` function with the following parameter:
```python
log_reg = LogisticRegressionCV(cv=5,Cs=[0.01,0.1,1,10],max_iter=1000,penalty="l1",solver="saga",multi_class="ovr")
```
- The performance of the regularization parameter and validation is the score of the logistic regression function on the test set as: **0.8967**

- Grid search
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search over
param_grid = {
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "penalty": ["l1", "l2", "elasticnet"],
    "C": [0.01, 0.1, 1, 10]
}

# Create a logistic regression model
log_reg = LogisticRegression()

# Create a grid search object
grid_search = GridSearchCV(log_reg, param_grid, cv=5)

# Fit the grid search on the training data
grid_search.fit(X_train,y_train)

# Print the best parameters and score
print(grid_search.best_params_)
print(grid_search.best_score_)

```


- To fit the logistic regression model into data of traning set.
```python
# Find the label for the data points
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=30, n_init=10)
y = kmeans.fit_predict(X_log_tranformed)

# Split training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_log_tranformed, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
log_reg = LogisticRegressionCV(cv=5,Cs=[0.01,0.1,1,10],max_iter=1000,penalty="l1",solver="saga",multi_class="ovr")
log_reg.fit(X_train,y_train)
log_reg.score(X_train,y_train)
```


Take the evaluation training data in `p2_evaluation` and use a subset of the genes consisting of the features you selected. Train a logistic regression classifier on this training data, and evaluate its performance on the evaluation test data. Report your score. (Don't forget to take the log transform before training and testing.)

Compare the obtained score with two baselines: random features (take a random selection of 100 genes), and high-variance features (take the 100 genes with highest variance). Finally, compare the variances of the features you selected with the highest variance features by plotting a histogram of the variances of features selected by both methods.

-  Firstly, calculate the sum of the absolute values of the coefficients for each feature in the logistic regression model. This gives a measure of how important each feature is for the model prediction. <mark> select feature using sum of absolute values of coefficients</mark>
-  Then, finds the indices of the 100 features with the highest coefficient values. These are the most relevant features for the model.
-  After that, create and fit a logistic regression model with cross-validation <mark> logistic regression with cross validation</mark> using L1 regularization and saga solver on a subset of the training data that contains only the 100 most relevant features. This can help reduce overfitting and improve generalization by selecting only the most informative features.
-  Then, evaluates the performance of the model on the training data by calculating its accuracy score <mark> evaluate performance </mark>. This gives an estimate of how well the model fits the data.
- Finally, select the features with the top 100 corresponding coefficient values by looking at the top highest 100 sum of absolute value per column to fit and report the score of: 
```python
coefficient_values = np.sum(np.abs(log_reg.coef_), axis=0)
# The highest 100 corresponding coefficient values indices
best_indices = np.argsort(coefficient_values)[-100:]
log_reg = LogisticRegressionCV(cv=5,Cs=[0.01,0.1,1,10],max_iter=1000,penalty="l1",solver="saga",multi_class="ovr")
log_reg.fit(X_train[:, best_indices], y_train)
```


- To train the logistic regression on this training data
```python
# Load the data
X_train = np.log2(np.load('p2_evaluation/X_train.npy')+1)
X_test = np.log2(np.load('p2_evaluation/X_test.npy')+1)
y_train = np.load('p2_evaluation/y_train.npy')
y_test = np.load('p2_evaluation/y_test.npy')
# Fit the model into data
log_reg = LogisticRegressionCV(cv=5,Cs=[0.01,0.1,1,10],max_iter=1000,penalty="l1",solver="saga",multi_class="ovr")
log_reg.fit(X_train[:, best_indices], y_train)
# Evaluate the performance
log_reg.score(X_train,y_train)
```
- The score is **0.8978**

- To calculate baseline score for 100 random features <mark> random </mark>
```python
log_reg = LogisticRegressionCV(cv=5,Cs=[0.01,0.1,1,10],max_iter=1000,penalty="l1",solver="saga",multi_class="ovr")
random_indices = np.random.choice(0,45767)
log_reg.fit(X_train[:, random_indices], y_train)
log_reg.score(X_train,y_train)
```
- The score is **0.6462** 

- To calculate the baseline score for 100 highest variance features <mark> Highest variance</mark>
```python
# To find the column with 
variances = np.var(X_train, axis=0)
high_variance_indeces = np.argsort(variances)[-100:]
# To fit the Logistic Regression CV with 100 highest variance features
log_reg = LogisticRegressionCV(cv=5,Cs=[0.01,0.1,1,10],max_iter=1000,penalty="l1",solver="saga",multi_class="ovr")
log_reg.fit(X_train[:, high_variance_indices], y_train)
log_reg.score(X_train,y_train)
```
- The score is **0.8624**

# Influence of Hyper-parameters

As we increase the number of PCs used, we may observe different patterns in the visualization. 
-   With a small number of PCs (e.g., 10), there are some clusters forming but also some overlap between different classes. This may indicate that some information is lost by using too few PCs.
-   With a moderate number of PCs (e.g., 50 or 100), there are clearer separation between different classes and more distinct clusters. This may indicate that most of the relevant information is preserved by using enough PCs.
-   With a large number of PCs (e.g., 250, 500 or 1000), there are worse results than with a moderate number of PCs. This may indicate that using too many PCs introduces noise or redundancy that affects T-SNE’s performance.

- ![TSNE](/images/TSNE-PCs-plots.png)

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define the number of PCs to use
n_pcs = [10, 50, 100, 250, 500]

# Loop over each number of PCs
for n_pc in n_pcs:
    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=n_pc)
    X_pca = pca.fit_transform(X)

    # Apply T-SNE to visualize the data
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_pca)

    # Plot the resulting visualization
    plt.figure()
    plt.scatter(X_tsne[:,0], X_tsne[:,1])
    plt.title(f"T-SNE with {n_pc} PCs")
```


- **T-SNE perplexity (Category A - Visualization)**
	- As a common range for perplexity is 5 to 50, take the list of [5, 10, 20, 30, 40, 50] to examine the visualization
	- ![TSNE](/images/TNSE-perplexities-examine.png)
- The true label shows how the data points form distinct clusters in the TSNE plot. The clusters become more visible as the perplexity increases. The best perplexity value is 40 because it keeps the local structure of clusters separate and coherent, and does not split them apart as far as perplexity 50.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

perplexities = [5, 10, 20, 30, 40, 50]
fig, axes = plt.subplots(2, 3, figsize=(10, 10))

for i, perplexity in enumerate(perplexities):
    # Create a TSNE instance with the given perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity)
    X_embedded = tsne.fit_transform(X)

	ax = axes[i//3][i%3]
    # Scatter plot the embedded points with different colors for each label
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
    # Set the title of the subplot as "Perplexity=perplexity"
    ax.set_title(f"Perplexity={perplexity}")
# Show the figure
plt.show()
```

 - **T-SNE learning rate (Category A - Visualization)**
- The learning rate for t-SNE is usually in the range **[10.0, 1000.0]**. If the learning rate is too high or too low, it may affect the quality of the visualization.
- ![TSNE](/images/TSNE-learning-rates.png)

- Increasing the learning rate from 10 to 1000 the TSNE plot shows that the clearer separation of clusters or groups that reflect the underlying structure of as colored by true label. 
 
- If the learning rate is too high, the data points may be too spread out and lose their local similarities. If the learning rate is too low, the data points may be too clustered and obscure their global differences. The optimum learning rate from the plots is 30 as the clusters are separated clearly and reflect the underlying structure of true labels.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

learning_rates = [10, 20, 30, 50, 100, 1000]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, lr in enumerate(learning_rates):
# Fit the TSNE with assigned learning rate
    tsne = TSNE(n_components=2, learning_rate=lr)
    X_embedded = tsne.fit_transform(X)
# Plot the results
    ax = axes[i//3][i%3]
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
    ax.set_title(f" TNSE plot with Learning rate: {lr}")    
plt.show()
```


**Type of regularization (l1, l1, elastic net) in the logistic regression step and how the resulting features selected differ**

- Use train_test_split to split the data into training and testing sets with a ratio of 70:30
- The code is then fitting the model to the training data (X_train and y_train) using fit method. Then, calculate the accuracy score of the model on the training data using score method on the test data (X_test and y_test).

|                                | l1 penalty | l2 penalty | elasticnet penalty with Cross-Validation method between several l1 ratio |
| ------------------------------ | ---------- | ---------- | ------------------------------------------------------------------------ |
| Accuracy score on the test set | 0.9677     | 0.9677     | 0.9611                                                                  |


```python
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Fit the logistic regression into training with
    # l1 penalty
log_reg_l1 = LogisticRegression(max_iter=1000,penalty="l1",solver="saga",multi_class="ovr")
log_reg_l1.fit(X_train,y_train)
   # l2 penalty
log_reg_l2 = LogisticRegression(max_iter=1000,penalty="l2",solver="saga",multi_class="ovr")
log_reg_l2.fit(X_train,y_train)
    # elasticnet penalty with Cross-Validation of several l1 ratios
log_reg_elastic = LogisticRegressionCV(penalty='elasticnet', solver='saga', l1_ratios=[0.1, 0.5, 0.9],multi_class="ovr")
log_reg_elastic.fit(X_train,y_train)

print(log_reg_l1.score(X_test, y_test), log_reg_l2.score(X_test, y_test), log_reg_elastic.score(X_test, y_test))
```
