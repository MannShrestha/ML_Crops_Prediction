# ML_Crops_Prediction
# 16 Machine Learning Algorithms

DataSet = (https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

## Data Description:
**Data fields**
- N - ratio of Nitrogen content in soil
- P - ratio of Phosphorous content in soil
- K - ratio of Potassium content in soil
- temperature - temperature in degree Celsius
- humidity - relative humidity in %
- ph - ph value of the soil
- rainfall - rainfall in mm

## for libraries
- You can use requirements.txt by
- Go to api directory download -> ``requirements.txt`` then in your command prompt use command ```pip install -r requirements.txt``` to install packages

### Data Inspection
- Data Shape
- Data Description
- Data Information
- Checking unique label
  
### Exploratory Data Analysis ( EDA )

### Feature Selection

### Modeling Machine Learning Algorithms
1. Logistic Regression:
   
   Logistic regression is a statistical technique used for binary classification, which aims to predict the probability of an event occurring based on a set of independent variables. It utilizes the logistic function to model the relationship between the predictor variables and the dependent variable, transforming the linear regression output into a range bounded between 0 and 1. By estimating the parameters through maximum likelihood estimation, logistic regression enables the classification of new observations into one of the two predefined categories, making it a widely employed and interpretable method in various fields such as medicine, finance, and social sciences.

2. K-Nearest Neighbour (KNN):

   K Nearest Neighbors (KNN) is a simple yet powerful supervised machine learning algorithm used for classification and regression tasks. It operates on the principle of proximity, where the classification of a new data point is determined by the majority vote or averaging of its K nearest neighbors in the feature space. KNN does not assume any underlying data distribution, making it a non-parametric method. The choice of K, the number of neighbors to consider, and the distance metric used are important factors that impact the algorithm's performance. KNN is intuitive, easy to implement, and can handle complex decision boundaries, making it a versatile algorithm used in various domains such as pattern recognition, recommendation systems, and anomaly detection.

  - ## Hyperparmeter Tuning
    Hyperparameter tuning refers to the process of finding the optimal values for the hyperparameters of a machine learning model that cannot be learned from the training data. Hyperparameters control the behavior and performance of the model, such as the learning rate, regularization strength, or the number of hidden units in a neural network. The process typically involves systematically exploring different combinations of hyperparameter values, training and evaluating the model on a validation set, and selecting the hyperparameter values that yield the best performance. Hyperparameter tuning is crucial for optimizing model performance, generalization, and avoiding overfitting, as it helps fine-tune the model to the specific characteristics of the dataset and problem at hand.

3. Decision Tree:

   A decision tree is a popular supervised machine learning algorithm used for both classification and regression tasks. It represents a flowchart-like structure where each internal node denotes a feature test, each branch represents an outcome of the test, and each leaf node corresponds to a class label or a predicted value. Decision trees are built based on a top-down greedy approach, recursively partitioning the training data by selecting the best feature to split on at each node, aiming to maximize information gain or minimize impurity. The resulting tree can be easily interpreted and visualized, allowing for transparent decision-making processes. Decision trees are versatile, capable of handling both numerical and categorical data, and can capture complex relationships between variables, although they are prone to overfitting and can be sensitive to small changes in the data. Nevertheless, various ensemble techniques such as random forests and boosting can be employed to enhance their performance and robustness.

4. Random Forest:

   Random forest is an ensemble machine learning algorithm that combines multiple decision trees to make accurate predictions for classification and regression tasks. It works by creating a multitude of decision trees using a subset of the training data and a random subset of features for each tree. During prediction, the output of the random forest is determined by aggregating the predictions of all individual trees, either through majority voting (classification) or averaging (regression). Random forests excel in handling high-dimensional data, capturing complex interactions between variables, and mitigating the overfitting issues commonly associated with decision trees. By utilizing randomization and averaging, they reduce the variance and improve the generalization capability of the model. Random forests are widely used in various domains, offering robust and reliable predictions, and are also capable of providing feature importance rankings, enabling valuable insights into the underlying data.

5. Naive Bayes Classifier:

   Naive Bayes classifier is a simple yet effective probabilistic machine learning algorithm used for classification tasks. It is based on Bayes' theorem and the assumption of feature independence, hence the term "naive." The algorithm calculates the posterior probability of a class given the observed features by multiplying the prior probability of the class and the likelihood of the features given the class. Despite the assumption of feature independence, Naive Bayes performs well in practice and is computationally efficient. It is particularly suitable for text classification tasks, such as spam detection or sentiment analysis, where each feature (e.g., word) can be considered independently. Naive Bayes can handle high-dimensional data and requires a relatively small amount of training data to estimate the probabilities accurately. It is a versatile algorithm that can be extended to handle continuous and categorical features using different distribution assumptions.

6. Extra Trees:

   Extra Trees, short for Extremely Randomized Trees, is an ensemble machine learning algorithm that builds a collection of decision trees and combines their predictions to make accurate classifications or regression predictions. Similar to random forests, Extra Trees uses a random subset of features for each tree and applies random splits at each internal node. However, what sets Extra Trees apart is its additional level of randomness, where it selects the splitting point randomly instead of finding the optimal split based on impurity measures. This randomization makes Extra Trees faster to train than random forests while still providing robustness against overfitting. By aggregating the predictions of multiple trees, Extra Trees can handle complex relationships in the data, handle high-dimensional feature spaces, and provide feature importance rankings. It is a useful algorithm in various domains, including pattern recognition, anomaly detection, and feature selection.

7. Support Vector Machines (SVM):

    Support Vector Machines (SVM) is a powerful supervised machine learning algorithm used for both classification and regression tasks. SVM works by finding an optimal hyperplane in a high-dimensional feature space that maximally separates different classes or approximates a regression function with a maximum margin. The algorithm aims to find the best decision boundary by selecting support vectors, which are the data points closest to the decision boundary. SVM can handle linearly separable as well as nonlinearly separable data by using kernel functions to transform the feature space. It is known for its ability to handle high-dimensional data, resistance to overfitting, and generalization capability. SVM has been widely applied in various domains, including image recognition, text classification, and bioinformatics, and offers flexibility through different kernel choices such as linear, polynomial, and radial basis functions.

8. Neural Networks (Multi-layer Perceptron):

    Neural Networks, specifically the Multi-layer Perceptron (MLP), are a class of powerful and flexible machine learning models inspired by the structure and functioning of the human brain. MLPs consist of multiple layers of interconnected artificial neurons, where each neuron receives inputs, applies an activation function, and passes the result to the next layer. The hidden layers enable the network to learn complex nonlinear relationships between features, making MLPs capable of solving a wide range of tasks such as classification, regression, and pattern recognition. The weights connecting the neurons are learned through an iterative process called backpropagation, which adjusts the weights to minimize the error between predicted and actual outputs. MLPs are known for their ability to capture intricate patterns in data, but they also require a substantial amount of labeled training data and longer training times compared to some other algorithms. However, advancements in hardware and optimization techniques have made MLPs increasingly popular and effective in areas like computer vision, natural language processing, and speech recognition.

9. AdaBoost:

    AdaBoost, short for Adaptive Boosting, is an ensemble machine learning algorithm that combines weak classifiers to create a strong classifier. It works by iteratively training a series of classifiers on different weighted subsets of the training data. In each iteration, the algorithm assigns higher weights to misclassified instances, allowing subsequent classifiers to focus on the difficult examples. During the prediction phase, AdaBoost combines the predictions of all individual classifiers using weighted majority voting. This boosting technique effectively adjusts the emphasis on misclassified instances, improving the overall accuracy of the final classifier. AdaBoost is particularly effective when used with simple base classifiers, known as weak learners, such as decision trees or stumps. It has been successfully applied to various domains and problems, including face detection, object recognition, and text classification, demonstrating its ability to handle complex tasks and improve predictive performance.

10. Light Gradient Boosting Machine (LGBM):

    Light Gradient Boosting Machine (LGBM) is a high-performance gradient boosting framework that uses tree-based models to solve machine learning problems efficiently. LGBM is known for its speed and scalability, making it well-suited for large datasets. It employs a gradient-based approach to train an ensemble of decision trees in a sequential manner, where each tree corrects the mistakes made by the previous trees. LGBM introduces several optimizations, such as leaf-wise tree growth, histogram-based feature discretization, and gradient-based binning, to achieve faster training times and lower memory usage while maintaining high accuracy. It also provides features like handling missing values and categorical features, early stopping criteria, and built-in mechanisms for handling class imbalance. LGBM has gained popularity in various domains and competitions due to its ability to handle diverse data types, handle large-scale datasets, and deliver competitive performance with efficient resource utilization.

11. CatBoost:

    CatBoost is a high-performance gradient boosting framework that excels in handling categorical features and producing accurate predictions. Developed by Yandex, CatBoost implements innovative techniques such as ordered boosting, which leverages the natural order of categorical variables, and gradient-based learning on preprocessed categorical features. It automatically handles categorical variables by converting them into numerical representations, reducing the need for manual preprocessing. CatBoost also introduces novel strategies to address the common challenges of overfitting and noisy data, including symmetric trees, random permutations, and Bayesian priors. With its robust handling of categorical features, strong predictive power, and efficient memory usage, CatBoost has gained popularity in various domains, including recommendation systems, fraud detection, and image classification.

12. Stochastic Gradient Descent (SGD):

    Stochastic Gradient Descent (SGD) is an optimization algorithm commonly used for training machine learning models. It is particularly suitable for large-scale datasets as it computes the gradients of the model parameters using a random subset of the training data in each iteration, making it computationally efficient. SGD iteratively updates the model weights by taking small steps in the direction of the negative gradient of the loss function. This iterative process enables the model to gradually converge towards the optimal set of parameters. Although SGD can be more noisy compared to other optimization algorithms, it often reaches a good solution and allows for faster training. Moreover, variants such as mini-batch SGD and adaptive learning rate methods like Adam or RMSprop enhance its performance by introducing batch-wise computations or adaptive step size adjustments. SGD has found applications in various machine learning tasks, including deep learning, where it plays a vital role in training large neural networks.

13. Linear Discriminant Analysis (LDA):

    Linear Discriminant Analysis (LDA) is a dimensionality reduction and classification technique widely used in machine learning and pattern recognition. LDA aims to find a linear combination of features that maximally separates different classes while minimizing the within-class scatter. It achieves this by projecting the data onto a lower-dimensional subspace while preserving the class-discriminatory information. LDA assumes that the data follows a Gaussian distribution and the classes have equal covariance matrices. By maximizing the ratio of between-class scatter to within-class scatter, LDA finds discriminant directions that optimize class separation. LDA can be used for both binary and multiclass classification problems and provides interpretable results. It is often employed as a preprocessing step before applying other classification algorithms or as a standalone classifier. LDA has proven effective in various applications, such as face recognition, text categorization, and bioinformatics.

14. Quadratic Discriminant Analysis (QDA):

    Quadratic Discriminant Analysis (QDA) is a classification technique that extends the concept of Linear Discriminant Analysis (LDA) by relaxing the assumption of equal covariance matrices across classes. Unlike LDA, which assumes a common covariance matrix for all classes, QDA allows for each class to have its own covariance matrix. QDA models the class distributions as multivariate Gaussian and estimates the mean and covariance matrix for each class. During prediction, QDA calculates the probability of an instance belonging to each class using the class-specific mean and covariance matrices and assigns the instance to the class with the highest probability. By allowing different covariance matrices, QDA can capture more complex relationships and decision boundaries between classes. However, it also requires a larger number of parameters to estimate, making it more prone to overfitting in cases of limited training data. QDA is particularly useful when there is evidence of distinct class-specific covariance structures in the data and has been applied in various domains, including medical diagnosis, pattern recognition, and finance.

15. GradientBoostingClassifier(GBC):

    GradientBoostingClassifier (GBC) is a popular machine learning algorithm that belongs to the ensemble methods family and utilizes gradient boosting to improve predictive performance. It constructs a strong predictive model by combining multiple weak prediction models, typically decision trees. GBC works by iteratively fitting new models to the residuals of the previous model, effectively minimizing the loss function by updating the model's parameters in the direction of steepest descent. This iterative process allows GBC to gradually improve the model's accuracy by focusing on the difficult-to-predict instances. It also incorporates regularization techniques to control overfitting, such as shrinkage and subsampling. GBC is known for its ability to handle complex nonlinear relationships in the data and provides high predictive accuracy. It has been successfully applied in various domains, including click-through rate prediction, financial forecasting, and medical diagnosis. However, GBC can be computationally expensive and sensitive to hyperparameter tuning, requiring careful optimization to achieve optimal performance.

16. XGBoost:

    XGBoost, short for Extreme Gradient Boosting, is an advanced implementation of the gradient boosting algorithm that excels in both speed and performance. Developed by Tianqi Chen, XGBoost combines the strengths of gradient boosting with a set of additional features and optimizations to enhance its effectiveness. It employs a similar iterative process, where weak learners (decision trees) are sequentially added and trained to minimize the loss function through gradient descent. XGBoost introduces regularization techniques such as shrinkage, column subsampling, and leaf-wise tree growth, which reduce overfitting and improve generalization. It also includes advanced features like handling missing values, built-in cross-validation, and support for parallel processing. XGBoost is renowned for its scalability, handling large-scale datasets efficiently, and delivering superior predictive accuracy. It has become a popular choice in various domains and machine learning competitions, showcasing its effectiveness in tasks such as click-through rate prediction, anomaly detection, and recommendation systems.

### Making Predictions

### Model Evaluation

### Accuracy Comparision
![Accuracy_Comparision](https://github.com/MannShrestha/ML_Crops_Prediction/assets/45268653/dcdfd1b0-4ba4-40a9-ad3f-db42efea3213)
![Train Vs test accuracy](https://github.com/MannShrestha/ML_Crops_Prediction/assets/45268653/1ba524c2-875d-4ecc-9cca-c6e77b15b5e0)

### Conclusion:
After applying ML-Algorithms, making model predictions and evalualtion, also compare accuracy of each algorithms. The SVM and XBoost gave me a better accuracy than other algorithms
