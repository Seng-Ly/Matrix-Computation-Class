# Logistic regression for a binary classification with a regularization

#### 1. Training Data

- load the training data file ('data-nonlinear.txt')
- each row $ \{ (x^{(i)}, y^{(i)}, l^{(i)}) \} $ of the data consists of a 2-dimensional point $ (x, y) $ with its label $ l $
- $ x, y \in \mathbb{R} $ and $ l \in \{0, 1\} $

#### 2. Logistic regression with a high dimensional feature function

- $ \hat{h} = \sigma(z) $ 
- $ z = g(x, y; \theta) $, where $ g $ is a high dimensional function and $ \theta \in \mathbb{R}^{100} $
- $ \theta = (\theta_{0,0}, \theta_{0,1}, \cdots, \theta_{9,9}) $
- $ g(x, y ; \theta) = \sum_{i=0}^{9} \sum_{j=0}^{9} \theta_{i,j} x^{i} y^{j} $
- $ \sigma(z) = \frac{1}{1 + \exp(-z)} $
- $ \sigma^{\prime}(z) = \sigma(z) (1 - \sigma(z)) $

#### 3. Objective Function with a regularization term

- $ J(\theta) = \frac{1}{m} \sum_{i=1}^m \left[ - l^{(i)} \log(\sigma(g(x^{(i)}, y^{(i)}; \theta))) - (1 - l^{(i)}) \log(1 - \sigma(g(x^{(i)}, y^{(i)}; \theta))) \right] + \frac{\lambda}{2} \sum_{i=0}^{9} \sum_{j=0}^{9} \theta_{i,j}^2 $
- the degree of regularization is determined by the control parameter $ \lambda $
- the larger value of $ \lambda $ yields smoother classification boundary

#### 4. Gradient Descent

- $ \theta_{i, j}^{(t+1)} \coloneqq \theta_{i, j}^{(t)} - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (\sigma(g(x^{(i)}, y^{(i)}; \theta^{(t)})) - l^{(i)}) \frac{\partial g(x^{(i)}, y^{(i)}; \theta^{(t)})}{\partial \theta_{i, j}} + \lambda \theta_{i, j}^{(t)} \right] $, for all $ i, j $
- you should choose a learning rate $ \alpha $ in such a way that the convergence is achieved
- you can use initial conditions $ \theta_{i, j}^{(0)} $ for all $ i, j $ to achieve the best accuracy
 
#### 5. Training

- find optimal parameters $ \theta $ using the training data with varying values of the regularization parameter $ \lambda $

#### 6. Compute the training accuracy

- the accuracy is computed by $ \frac{\textrm{number of correct predictions}}{\textrm{total number of predictions}} $
- compute the accuracy with varying values of the regularization parameter $ \lambda $

## Code

- load the training data from the file and plot the points

    python
import numpy as np
import matplotlib.pyplot as plt

data    = np.genfromtxt("data-nonlinear.txt", delimiter=',')

pointX  = data[:, 0]
pointY  = data[:, 1]
label   = data[:, 2]

pointX0 = pointX[label == 0]
pointY0 = pointY[label == 0]

pointX1 = pointX[label == 1]
pointY1 = pointY[label == 1]

plt.figure()
plt.scatter(pointX0, pointY0, c='b')
plt.scatter(pointX1, pointY1, c='r')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
   

<img src="data-nonlinear.png"  width="500">

## Submission

### 1. Codes, Comments and Results

_PDF file that is exported from Notebook including codes, comments, and results for the above problem using Jupyter Notebook or Colab_

#### [Plotting the results]

##### 1. Plot the training data [1pt]
- plot the training data points $ (x, y) $ with their labels $ l $ in colors (blue for label 0 and red for label 1)

##### 2. Plot the training error with varying regularization parameters
- choose a value for $ \lambda_1 $ in such a way that  over-fitting  is demonstrated and plot the training error $ J(\theta) $ at every iteration of gradient descent until convergence (in red color) [3pt]
- choose a value for $ \lambda_2 $ in such a way that  just-right  is demonstrated and plot the training error $ J(\theta) $ at every iteration of gradient descent until convergence (in green color) [3pt]
- choose a value for $ \lambda_3 $ in such a way that  under-fitting  is demonstrated and plot the training error $ J(\theta) $ at every iteration of gradient descent until convergence (in blue color) [3pt]
- the above three curves should be presented all together in a single figure

##### 3. Display the values of the chosen regularization parameters
- display the value of the chosen $ \lambda_1 $ for the demonstration of  over-fitting  (in red color) [1pt]
- display the value of the chosen $ \lambda_2 $ for the demonstration of  just-right  (in green color) [1pt]
- display the value of the chosen $ \lambda_3 $ for the demonstration of  under-fitting  (in blue color) [1pt]

##### 4. Plot the training accuracy with varying regularization parameters
- plot the training accuracy with the chosen $ \lambda_1 $ for  over-fitting  at every iteration of gradient descent until convergence (in red color) [3pt]
- plot the training accuracy with the chosen $ \lambda_2 $ for  just-right  at every iteration of gradient descent until convergence (in green color) [3pt]
- plot the training accuracy with the chosen $ \lambda_3 $ for  under-fitting  at every iteration of gradient descent until convergence (in blue color) [3pt]
- the above three curves should be presented all together in a single figure

##### 5. Display the final training accuracy with varying regularization parameters
- display the final training accuracy obtained with the chosen $ \lambda_1 $ for  over-fitting  in number (%) at convergence (in red color) [1pt]
- display the final training accuracy obtained with the chosen $ \lambda_2 $ for  just-right  in number (%) at convergence (in green color) [1pt]
- display the final training accuracy obtained with the chosen $ \lambda_3 $ for  under-fitting  in number (%) at convergence (in blue color) [1pt]

##### 6. Plot the optimal classifier with varying regularization parameters superimposed on the training data
- plot the boundary of the optimal classifier with the chosen $ \lambda_1 $ for  over-fitting  at convergence (in red color) [3pt]
- plot the boundary of the optimal classifier with the chosen $ \lambda_2 $ for  just-right  at convergence (in green color) [3pt]
- plot the boundary of the optimal classifier with the chosen $ \lambda_3 $ for  under-fitting  at convergence (in blue color) [3pt]
- the boundary of the classifier is defined by $ \{ (x, y) \mid \sigma(g(x, y ; \theta)) = 0.5 \} = \{ (x, y) \mid g(x, y ; \theta) = 0 \} $
- the boundaries of the classifiers with different regularization parameters should be presented with the training data points $ (x, y) $ with their labels $ l $ in colors (blue for label 0 and red for label 1)
- you can use  contour  function in python3

### 2. Commit History

_PDF file that is exported from the commit history at github_

#### [Apply  commit  at the end of the implementations at least for the following steps]

You should  git commit  at least 4 times with meaningful and proper commit messages in such way that you can demonstrate the progress of your programming in an effective way [1pt]