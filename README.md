# Prism: an unsupervised clustering and nearest neighbor search algorithm
[Clustering](https://en.wikipedia.org/wiki/Cluster_analysis) places similar vectors into the same cluster.
[Nearest neighbor search](https://en.wikipedia.org/wiki/Nearest_neighbor_search) finds similar vectors to a search query vector.
Prism does both of these things using [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) and [decision trees](https://en.wikipedia.org/wiki/Decision_tree)
The [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) learning algorithm is used to find an artificial neural network [autoencoder](https://en.wikipedia.org/wiki/Autoencoder).
The autoencoder is trained to maximize the [variance](https://en.wikipedia.org/wiki/Variance) of the output of the middle layer for the given training data.
The top half of the autoencoder is removed and a decision is learned using [variance reduction](https://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction).
The learned decision is used to split the training data into two sets, and the algorithm is recursively applied to create a binary tree.

# Results
The below results are from applying the algorithm to the [iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set).
The label entropy is computed for the learned clusters. The entropy is normalized between 0 and 1. Lower entropy is better.
The best entropy of [kmeans](https://en.wikipedia.org/wiki/K-means_clustering) is 0.153569, and the best entropy of the prism algorithm is 0.086526.
Prism performs ~1.8 times better than kmeans on the iris data set.

## Results for kmeans with different distance metrics
| distance func            | entropy  |
| ------------------------ | -------- |
| CanberraDistance         | 0.153569 |
| EuclideanDistance        | 0.248515 |
| ManhattanDistance        | 0.257049 |
| SquaredEuclideanDistance | 0.263581 |
| BrayCurtisDistance       | 0.271857 |
| ChebyshevDistance        | 0.441861 |
| HammingDistance          | 1.000000 |

## Results for run of algorithm with seed of 1
| mode          | consistency | entropy  |
| ------------- | ----------- | -------- |
| variance      | 11          | 0.086526 |
| raw           | 6           | 0.238475 |
| none          | 6           | 0.343876 |
| parallel      | 7           | 0.343876 |
| orthogonality | 7           | 0.351338 |
| mixed         | 7           | 0.351338 |
| entropy       | 11          | 0.364361 |

## Averaged results for 16 runs of algorithm with different seeds
| mode          | entropy mean | entropy variance |
| ------------- | ------------ | ---------------- |
| variance      | 0.086526     | 0.000000         |
| raw           | 0.238475     | 0.000000         |
| none          | 0.278432     | 0.009951         |
| parallel      | 0.301946     | 0.018403         |
| orthogonality | 0.360151     | 0.002570         |
| mixed         | 0.363574     | 0.002479         |
| entropy       | 0.382759     | 0.010021         |
