
<!-- README.md is generated from README.Rmd. Please edit that file -->

# Machine Learning Methods in R and Python with Quarto

This repository presents a collection of examples of **Machine
Learning** methods classified by category, implemented in both
**Python** and **R**, using the [**Quarto**](https://quarto.org/)
publishing platform to integrate code, results and documentation in the
same reproducible flow.

Each section includes `.qmd` and `.Rmd` notebooks that combine theory,
code, and visualizations.

<!-- badges: start -->

<!-- badges: end -->

<p align="center">

<img src="img/ML Categories_Page_2.png" alt="Machine Learning" width="80%"/>
</p>

## Table of Contents

1.  [About This Repository](#about-this-repository)
2.  [How to Use](#how-to-use)
3.  [Machine Learning Categories and
    Methods](#machine-learning-categories-and-methods)
    - [Bayesian](#bayesian)
    - [Clustering](#clustering)
    - [Decision Tree](#decision-tree)
    - [Deep Learning](#deep-learning)
    - [Dimensionality Reduction](#dimensionality-reduction)
    - [Ensemble](#ensemble)
    - [Instance-Based](#instance-based)
    - [Neural Network](#neural-network)
    - [Regression](#regression)
    - [Regularization](#regularization)
    - [Decision Rules](#decision-rules)
4.  [Requirements](#requirements)
5.  [References](#references)
6.  [Code of Conduct](#codeofconduct)
7.  [License](#license)

## About This Repository

The goal of this repository is to serve as a practical resource for
understanding and applying various machine learning algorithms. We aim
to:

- **Provide clear explanations:** Briefly describe each method’s core
  concepts.
- **Offer practical examples:** Illustrate implementation using real or
  simulated datasets.
- **Support multiple languages:** Showcase code in both R and Python,
  the two most popular languages for data science.
- **Ensure reproducibility:** Utilize Quarto documents (`.qmd`) to
  combine code, explanations, and outputs seamlessly.

**Please note:** The scripts and examples within this repository are
primarily **personal notebooks**. They are designed for educational
purposes, exploration, and my own learning journey. While effort is made
to ensure accuracy and best practices, they might not always represent
production-ready code or exhaustive analyses.

## How to Use

To explore the examples in this repository, you’ll need to:

1.  **Clone the repository:**

``` bash
git clone https://github.com/dvillasanao/ML_Examples.git
cd ML_Examples
```

2.  **Install Quarto:** Follow the instructions on the [Quarto
    website](https://quarto.org/docs/getting-started/).
3.  **Install R and Python environments:**
    - **For R:** Ensure you have R installed and necessary packages
      (e.g., `tidyverse`, `caret`, `e1071`, `tensorflow`, `keras`,
      etc.).
    - **For Python:** Ensure you have Python installed and necessary
      packages (e.g., `scikit-learn`, `pandas`, `numpy`, `matplotlib`,
      `seaborn`, `tensorflow`, `keras`, `pytorch`, etc.).
    - It’s highly recommended to use virtual environments (e.g., `conda`
      or `venv` for Python, `renv` for R) to manage dependencies.
4.  **Render Quarto documents:** Navigate to the specific method’s
    directory and render the `.qmd` file. For example: This will
    generate HTML, PDF, or Word documents depending on the output
    formats specified in the `.qmd` file.

``` bash
quarto render bayesian/naive_bayes.qmd
```

## Repository Structure.

    machine-learning-methods/
      │
      ├── Bayesian/
      ├── Clustering/
      ├── DecisionTree/
      ├── DeepLearning/
      ├── DimensionalityReduction/
      ├── Ensemble/
      ├── InstanceBased/
      ├── NeuralNetwork/
      ├── Regression/
      ├── Regularization/
      ├── DecisionRules/
      └── README.md

## Machine Learning Categories and Methods

Each category below contains a directory with Quarto documents (`.qmd`
files) for specific methods, demonstrating both R and Python
implementations.

| Category | Description |
|----|----|
| **Bayesian** | Models based on Bayesian inference such as Naive Bayes. |
| **Clustering** | Unsupervised clustering (k-means, DBSCAN, etc.). |
| **Decision Tree** | Decision trees for classification and regression. |
| **Deep Learning** | Deep networks with TensorFlow, PyTorch or Keras. |
| **Dimensionality Reduction** | Techniques such as PCA, t-SNE, UMAP. |
| **Ensemble** | Methods such as Random Forest, Gradient Boosting. |
| **Instance Based** | Methods such as k-NN, based on similarity between instances. |
| **Neural Network** | Simple artificial neural networks (MLP, perceptron). |
| **Regression** | Linear, polynomial, logistic regression. |
| **Regularization** | L1, L2 penalization (Ridge, Lasso). |
| **Decision Rules** | Rule-based algorithms such as RIPPER or OneR. |

------------------------------------------------------------------------

### Bayesian

Bayesian methods leverage probability theory to model uncertainty and
update beliefs based on data. They provide a principled way to
incorporate prior knowledge.

- **Methods Covered:**
  - **Naive Bayes:** Simple probabilistic classifier based on applying
    Bayes’ theorem with strong (naive) independence assumptions between
    the features.
    - `bayesian/naive_bayes.qmd` (R & Python examples)
  - **(Add more Bayesian methods as you implement them, e.g., Bayesian
    Networks, Gaussian Processes)**

------------------------------------------------------------------------

### Clustering

Clustering algorithms group similar data points together based on their
inherent characteristics, without prior knowledge of labels.

- **Methods Covered:**
  - **K-Means Clustering:** An iterative algorithm that partitions `n`
    observations into `k` clusters where each observation belongs to the
    cluster with the nearest mean.
    - `clustering/k_means.qmd` (R & Python examples)
  - **Hierarchical Clustering:** Builds a hierarchy of clusters, either
    by starting with individual data points and merging (agglomerative)
    or starting with one large cluster and splitting (divisive).
    - `clustering/hierarchical_clustering.qmd` (R & Python examples)
  - **DBSCAN:** Density-Based Spatial Clustering of Applications with
    Noise, a non-parametric clustering algorithm that identifies
    clusters based on the density of data points.
    - `clustering/dbscan.qmd` (R & Python examples)
  - **(Add more Clustering methods, e.g., Mean Shift, Affinity
    Propagation)**

------------------------------------------------------------------------

### Decision Tree

Decision trees are non-parametric supervised learning methods used for
classification and regression. They partition the data into subsets
based on feature values, forming a tree-like structure.

- **Methods Covered:**
  - **Classification and Regression Trees (CART):** A foundational
    algorithm for building decision trees.
    - `decision_tree/cart.qmd` (R & Python examples)
  - **(Add more Decision Tree methods, e.g., C4.5, ID3)**

------------------------------------------------------------------------

### Deep Learning

Deep learning is a subset of machine learning that uses artificial
neural networks with multiple layers (deep neural networks) to learn
complex patterns from data.

- **Methods Covered:**
  - **Feedforward Neural Networks (MLPs):** The simplest form of deep
    neural network, consisting of an input layer, one or more hidden
    layers, and an output layer.
    - `deep_learning/mlp.qmd` (R & Python examples)
  - **Convolutional Neural Networks (CNNs):** Primarily used for image
    processing and computer vision tasks, leveraging convolutional
    layers.
    - `deep_learning/cnn.qmd` (R & Python examples)
  - **Recurrent Neural Networks (RNNs):** Designed for sequential data,
    such as time series or natural language, with connections that form
    directed cycles.
    - `deep_learning/rnn.qmd` (R & Python examples)
  - **(Add more Deep Learning architectures, e.g., LSTMs, GRUs,
    Transformers, GANs)**

------------------------------------------------------------------------

### Dimensionality Reduction

Dimensionality reduction techniques reduce the number of random
variables under consideration by obtaining a set of principal variables.
This can help with visualization, noise reduction, and improving
algorithm performance.

- **Methods Covered:**
  - **Principal Component Analysis (PCA):** A linear dimensionality
    reduction technique that transforms data into a new coordinate
    system such that the greatest variance by any projection lies on the
    first coordinate (called the first principal component).
    - `dimensionality_reduction/pca.qmd` (R & Python examples)
  - **t-Distributed Stochastic Neighbor Embedding (t-SNE):** A
    non-linear dimensionality reduction technique well-suited for
    visualizing high-dimensional datasets.
    - `dimensionality_reduction/tsne.qmd` (R & Python examples)
  - **(Add more Dimensionality Reduction methods, e.g., LDA, UMAP,
    Autoencoders)**

------------------------------------------------------------------------

### Ensemble

Ensemble methods combine multiple machine learning models to obtain
better predictive performance than could be obtained from any of the
constituent models alone.

- **Methods Covered:**
  - **Bagging (Bootstrap Aggregating):** Combines predictions from
    multiple models trained on different bootstrap samples of the
    original dataset.
    - **Random Forest:** An ensemble learning method for classification
      and regression that operates by constructing a multitude of
      decision trees at training time.
      - `ensemble/random_forest.qmd` (R & Python examples)
  - **Boosting:** Sequentially builds an ensemble by training weak
    learners and adding them to the ensemble, with each new learner
    focusing on correcting errors made by previous ones.
    - **Gradient Boosting Machines (GBM):** A powerful boosting
      technique that builds models additively by fitting new models to
      the residuals of previous models.
      - `ensemble/gbm.qmd` (R & Python examples)
    - **XGBoost, LightGBM, CatBoost:** Highly optimized gradient
      boosting implementations.
      - `ensemble/xgboost.qmd` (R & Python examples)
  - **Stacking:** Combines multiple models using a meta-learner, where
    the predictions of base models are used as input features for the
    meta-learner.
    - `ensemble/stacking.qmd` (R & Python examples)
  - **(Add more Ensemble methods, e.g., AdaBoost)**

------------------------------------------------------------------------

### Instance-Based

Instance-based learning (or memory-based learning) algorithms simply
store the training instances and generalize from them directly when a
new instance is encountered.

- **Methods Covered:**
  - **K-Nearest Neighbors (KNN):** A non-parametric method used for
    classification and regression. The output is based on the k-nearest
    training examples in the feature space.
    - `instance_based/knn.qmd` (R & Python examples)
  - **(Add more Instance-Based methods, e.g., Self-Organizing Maps)**

------------------------------------------------------------------------

### Neural Network

*Note: While Deep Learning is a subset of Neural Networks, this category
specifically refers to more foundational or classical neural network
architectures not typically classified under “deep learning” due to
fewer layers or specific historical contexts.*

- **Methods Covered:**
  - **Perceptron:** The simplest type of artificial neural network, a
    linear binary classifier.
    - `neural_network/perceptron.qmd` (R & Python examples)
  - **Multi-Layer Perceptron (MLP):** (Could be a simpler MLP example
    here, or redirect to `deep_learning/mlp.qmd` if the distinction is
    primarily in complexity/number of layers).
    - `neural_network/mlp_simple.qmd` (R & Python examples - for a
      basic, non-deep example)
  - **(Add more Neural Network concepts, e.g., Backpropagation explained
    conceptually)**

------------------------------------------------------------------------

### Regression

Regression algorithms are supervised learning methods used to predict a
continuous target variable.

- **Methods Covered:**
  - **Linear Regression:** Models the relationship between a dependent
    variable and one or more independent variables by fitting a linear
    equation to the observed data.
    - `regression/linear_regression.qmd` (R & Python examples)
  - **Polynomial Regression:** Models the relationship between the
    independent variable `x` and the dependent variable `y` as an nth
    degree polynomial.
    - `regression/polynomial_regression.qmd` (R & Python examples)
  - **Logistic Regression:** A statistical model that in its basic form
    uses a logistic function to model a binary dependent variable,
    though it can be extended to multi-class problems (Despite its name,
    it’s primarily a classification algorithm).
    - `regression/logistic_regression.qmd` (R & Python examples)
  - **Support Vector Regression (SVR):** An extension of Support Vector
    Machines (SVMs) for regression problems.
    - `regression/svr.qmd` (R & Python examples)
  - **(Add more Regression methods, e.g., Ridge, Lasso (covered in
    Regularization), Elastic Net)**

------------------------------------------------------------------------

### Regularization

Regularization techniques are used to prevent overfitting in machine
learning models, especially in linear models, by adding a penalty term
to the loss function.

- **Methods Covered:**
  - **L1 Regularization (Lasso Regression):** Adds a penalty equal to
    the absolute value of the magnitude of coefficients. Can lead to
    sparse models by setting some coefficients to zero.
    - `regularization/lasso_regression.qmd` (R & Python examples)
  - **L2 Regularization (Ridge Regression):** Adds a penalty equal to
    the square of the magnitude of coefficients. Shrinks coefficients
    towards zero but does not set them to exactly zero.
    - `regularization/ridge_regression.qmd` (R & Python examples)
  - **Elastic Net Regularization:** Combines L1 and L2 penalties,
    offering the benefits of both.
    - `regularization/elastic_net_regression.qmd` (R & Python examples)
  - **Dropout (for Neural Networks):** A regularization technique for
    neural networks where randomly selected neurons are ignored during
    training.
    - `regularization/dropout.qmd` (R & Python examples, typically
      within a deep learning model)

------------------------------------------------------------------------

### Decision Rules

Decision rule-based systems explicitly define rules to classify or
predict outcomes.

- **Methods Covered:**
  - **OneR:** (One Rule) A simple and interpretable classification
    algorithm that generates a one-rule classifier.
    - `decision_rules/oner.qmd` (R & Python examples)
  - **RIPPER (Repeated Incremental Pruning to Produce Error
    Reduction):** A rule-based classification algorithm that learns
    decision rules from data.
    - `decision_rules/ripper.qmd` (R & Python examples)
  - **(Add more Decision Rule methods)**

------------------------------------------------------------------------

## Requirements

- [Quarto](https://quarto.org/)
- Python 3.x with libraries: `scikit-learn`, `pandas`, `matplotlib`,
  `seaborn`, etc.
- R with packages: `caret`, `mlr3`, `tidymodels`, among others.

## References

- [Quarto Documentation](https://quarto.org/docs/)
- [Scikit-learn (Python)](https://scikit-learn.org/)
- [Tidymodels (R)](https://www.tidymodels.org/)
- [The caret Package](https://topepo.github.io/caret/)
- [CRAN Machine Learning Task
  View](https://cran.r-project.org/web/views/MachineLearning.html)

## Code of Conduct

Please review the [Code of Conduct](CODE_OF_CONDUCT.md) before
contributing.

## License

This work by Diana Villasana Ocampo is licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
License Creative Commons Atribución 4.0 Internacional.</a>.
