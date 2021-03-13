## Question 1: Regularization is defined as:

**The minimization of the sum of squared residuals subject to a constraint on the weights.** This is correct because this is the goal of regularization.


## Question 2: The regularization with the square of an L2 distance may improve the results compared to OLS when the number of features is higher than the number of observations. **True/False**

**True** because OLS fails when the number of features is greater than observations, so L2 would do a better job.


## Question 3: The L1 norm always yields shorter distances compared to the Euclidean norm. **True/False**

**False** because it is not always the case.


## Question 4: Typically, the regularization is achieved by

**Minimizing the average of the squared residuals plus a penalty function whose input is the vector of coefficients** because the main idea of regularization is to minimize the sum of square residuals plus a constraint on the vector of weights.


## Question 5: A regularization method that facilitates variable selection (estimating some coefficients as zero) is

**Lasso** because it is able to shrink coefficients to 0.


## Question 6: Write your own Python code to import the Boston housing data set (from the sklearn library) and scale the data (not the target) by z-scores. If we use all the features with the Linear Regression to predict the target variable then the root mean squared error (RMSE) is: (your answer should include only the first 4 decimals that you get from the code)
```markdown
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

scale = StandardScaler()
xscaled = scale.fit_transform(df)

model = linear_model.LinearRegression()

model.fit(xscaled,y)
y_pred = model.predict(xscaled[:,])
rmse = np.sqrt(mean_squared_error(y,y_pred))
rmse
```
This code allows us to understand that the RMSE is 4.6791. 


## Question 7: On the Boston housing data set if we consider the Lasso model with 'alpha=0.03' then the 10-fold cross-validated prediction error is: (for the 10-fold cross-validation shuffle you should use random_state=1234, your final answer should include only the first 4 decimals that you get from the code)
```markdown
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error as MAE

def DoKFold(X,y,model):
  PE = []
  for idxtrain,idxtest in kf.split(X):
  	Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    yhat = pipe.predict(Xtest)
    PE.append(MAE(ytest,yhat))
  return 1000 * np.mean(PE)
  
data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

X = df.values
kf = KFold(n_splits=10, shuffle=True, random_state=1234)

model = Lasso(alpha=0.03)
scale = StandardScaler()

pipe = Pipeline([('scale', scale), ('Regressor', model)])

DoKFold(X,y,model)
```
Based on this code, the prediction error is 3380.6870.


## Question 8: On the Boston housing data set if we consider the Elastic Net model with 'alpha=0.05' and 'l1_ratio=0.9' then the 10-fold cross-validated prediction error is: (for the 10-fold cross-validation shuffle you should use random_state=1234, your final answer should include only the first 4 decimals that you get from the code)
```markdown
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.linear_model import ElasticNet

def DoKFold(X,y,model):
  PE = []
  for idxtrain,idxtest in kf.split(X):
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    yhat = pipe.predict(Xtest)
    PE.append(MAE(ytest,yhat))
  return 1000 * np.mean(PE)
  
data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

X = df.values
kf = KFold(n_splits=10, shuffle=True, random_state=1234)

model = ElasticNet(alpha=0.05,l1_ratio=0.9)
scale = StandardScaler()

pipe = Pipeline([('scale', scale), ('Regressor', model)])

DoKFold(X,y,model)
```
Based on this code, the prediction error is 3364.4761.


## Question 9: If we create all quadratic polynomial (degree=2) features based on the z-scores of the original features and then apply OLS, the root mean squared error is: (provide your answer with 4 decimals)
```markdown
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

polynomial_features= PolynomialFeatures(degree=2)

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

scale = StandardScaler()
xscaled = scale.fit_transform(df)

x_poly = polynomial_features.fit_transform(xscaled)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
rmse
```
Based on the code, the RMSE is 2.4483.


## Question 10: If we create all quadratic polynomial (degree=2) features based on the z-scores of the original features and then apply the Ridge regression with alpha=0.1 and we create a Quantile-Quantile plot for the residuals then the result shows that the obtained residuals pretty much follow a normal distribution. **True/False**
```markdown
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

polynomial_features= PolynomialFeatures(degree=2)

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

scale = StandardScaler()
xscaled = scale.fit_transform(df)

x_poly = polynomial_features.fit_transform(xscaled)

model = Ridge(alpha=0.1)
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

residuals = y - y_poly_pred

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import pylab
import statsmodels.api as sm
pylab.figure(figsize=(10,10))
sm.qqplot(residuals, loc = 0, scale = 1, line='s')
pylab.show()
```
Based on the QQ plot shown, the residuals follow pretty much a normal distribution.
