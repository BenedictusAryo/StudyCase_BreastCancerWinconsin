
# Study Case : Winscosin Breast Cancer

#### Benedict Aryo
As part of Study Case Assignment on Make Ai Bootcamp
##### June 2018

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter) <br>b) texture (standard deviation of gray-scale values) <br>c) perimeter <br>d) area <br>e) smoothness (local variation in radius lengths) <br>f) compactness (perimeter^2 / area - 1.0) <br>g) concavity (severity of concave portions of the contour) <br>h) concave points (number of concave portions of the contour) <br>i) symmetry <br>j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

### Import Library and Dataset


```python
import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns
sns.set()
plt.style.use('bmh')
```


```python
df=pd.read_csv('Dataset/data.csv')
print("Dataset size : ",df.shape)
df=df.drop(columns=['id','Unnamed: 32'])
df.head()
```

    Dataset size :  (569, 33)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
pandas_profile=pandas_profiling.ProfileReport(df)
pandas_profile.to_file(outputfile='Pandas_ProfilingOutput.html')
#pandas_profile
```

## [Detail HTML Pandas Profiling](Pandas_ProfilingOutput.html)

### Explore the Values

Explore Distribution values from the dataset using describe statistic and histogram


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>...</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>0.062798</td>
      <td>...</td>
      <td>16.269190</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>0.007060</td>
      <td>...</td>
      <td>4.833242</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>0.049960</td>
      <td>...</td>
      <td>7.930000</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>0.057700</td>
      <td>...</td>
      <td>13.010000</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>0.061540</td>
      <td>...</td>
      <td>14.970000</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>0.066120</td>
      <td>...</td>
      <td>18.790000</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>0.097440</td>
      <td>...</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>




```python
df.hist(figsize=(16,25),bins=50,xlabelsize=8,ylabelsize=8);
```


![png](output_8_0.png)



## Training Dataset Preparation

Since most of the Algorithm machine learning only accept array like as input, so we need to create an array from dataframe set to X and y array before running machine learning algorithm


```python
X=np.array(df.drop(columns=['diagnosis']))
y=df['diagnosis'].values
```


```python
print ("X dataset shape : ",X.shape)
print ("y dataset shape : ",y.shape)
```

    X dataset shape :  (569, 30)
    y dataset shape :  (569,)
    

The dataset is splitted by X the parameter and y for classification labels

# Machine Learning Model

### Import Machine Learning Library from Scikit-Learn


```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
```

### Using 5 Machine Learning Model

Machine learning model used is Classification model, since the purpose of this Study case is to classify diagnosis between "Malignant" (M) Breast Cancer and "Benign" (B) Breast Cancer

* Model 1 : Using Simple Logistic Regression
* Model 2 : Using Support Vector Classifier
* Model 3 : Using Decision Tree Classifier
* Model 4 : Using Random Forest Classifier
* Model 5 : Using Gradient Boosting Classifier


```python
model_1 = LogisticRegression()
model_2 = SVC()
model_3 = DecisionTreeClassifier()
model_4 = RandomForestClassifier()
model_5 = GradientBoostingClassifier()
```

# Model Fitting

since we need to fit the dataset into algorithm, so proper spliting dataset into training set and test set are required

## Method 1. Train test split

Using Scikit learn built in tools to split data into training set and test set to check the result score of the model <br>
train_test_split configuration using 20% data to test and 80& data to train the model, random_state generator is 45.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

print ("Train size : ",X_train.shape)
print ("Test size : ",X_test.shape)
```

    Train size :  (455, 30)
    Test size :  (114, 30)
    

### Fitting train dataset into model


```python
model_1.fit(X_train,y_train)
model_2.fit(X_train,y_train)
model_3.fit(X_train,y_train)
model_4.fit(X_train,y_train)
model_5.fit(X_train,y_train)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=None, subsample=1.0, verbose=0,
                  warm_start=False)



### Predict and show Score and F1 Score prediction using test data


```python
# Predict data
y_pred1=model_1.predict(X_test)
y_pred2=model_2.predict(X_test)
y_pred3=model_3.predict(X_test)
y_pred4=model_4.predict(X_test)
y_pred5=model_5.predict(X_test)

#Show F1 Score
from sklearn.metrics import f1_score
f1_model1=f1_score(y_test,y_pred1,average='weighted',labels=np.unique(y_pred1))
f1_model2=f1_score(y_test,y_pred2,average='weighted',labels=np.unique(y_pred2))
f1_model3=f1_score(y_test,y_pred3,average='weighted',labels=np.unique(y_pred3))
f1_model4=f1_score(y_test,y_pred4,average='weighted',labels=np.unique(y_pred4))
f1_model5=f1_score(y_test,y_pred5,average='weighted',labels=np.unique(y_pred5))

print("F1 score Model 1 : ",f1_model1)
print("F1 score Model 2 : ",f1_model2)
print("F1 score Model 3 : ",f1_model3)
print("F1 score Model 4 : ",f1_model4)
print("F1 score Model 5 : ",f1_model5)
```

    F1 score Model 1 :  0.9557756825927252
    F1 score Model 2 :  0.7741935483870968
    F1 score Model 3 :  0.9112731152204836
    F1 score Model 4 :  0.9557756825927252
    F1 score Model 5 :  0.9734654095556352
    

## Method 2. Cross validation method

Using Cross validation will resulted in more reliability of the model <br>
in this case using StratifiedKFold from Scikit Learn, with n_split = 10 times and Shuffle = True


```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True)
skf.get_n_splits(X,y)
```




    10




```python
# Set Container to gather the cross validation result of the model
score_list_model1,score_list_model2,score_list_model3,score_list_model4,score_list_model5 = [],[],[],[],[]
```


```python
for train_index, test_index in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)
    model_3.fit(X_train, y_train)
    model_4.fit(X_train, y_train)
    model_5.fit(X_train, y_train)
    y_pred1=model_1.predict(X_test)
    y_pred2=model_2.predict(X_test)
    y_pred3=model_3.predict(X_test)
    y_pred4=model_4.predict(X_test)
    y_pred5=model_5.predict(X_test)
    score_list_model1.append(f1_score(y_test,y_pred1,average='weighted',labels=np.unique(y_pred1)))
    score_list_model2.append(f1_score(y_test,y_pred2,average='weighted',labels=np.unique(y_pred2)))
    score_list_model3.append(f1_score(y_test,y_pred3,average='weighted',labels=np.unique(y_pred3)))
    score_list_model4.append(f1_score(y_test,y_pred4,average='weighted',labels=np.unique(y_pred4)))
    score_list_model5.append(f1_score(y_test,y_pred5,average='weighted',labels=np.unique(y_pred5)))
 
```


```python
score_table = pd.DataFrame({"F1 Score model 1" :score_list_model1,
                           "F1 Score model 2" :score_list_model2,
                           "F1 Score model 3" :score_list_model3,
                           "F1 Score model 4" :score_list_model4,
                           "F1 Score model 5" :score_list_model5})
score_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1 Score model 1</th>
      <th>F1 Score model 2</th>
      <th>F1 Score model 3</th>
      <th>F1 Score model 4</th>
      <th>F1 Score model 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.929401</td>
      <td>0.765957</td>
      <td>0.931034</td>
      <td>0.948029</td>
      <td>0.948029</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.948029</td>
      <td>0.765957</td>
      <td>0.982829</td>
      <td>0.947418</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.929018</td>
      <td>0.774194</td>
      <td>0.895625</td>
      <td>0.894737</td>
      <td>0.930417</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.947610</td>
      <td>0.774194</td>
      <td>0.930417</td>
      <td>0.964912</td>
      <td>0.982537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.000000</td>
      <td>0.774194</td>
      <td>0.930417</td>
      <td>1.000000</td>
      <td>0.982537</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.929018</td>
      <td>0.774194</td>
      <td>0.910661</td>
      <td>0.929018</td>
      <td>0.964509</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.982537</td>
      <td>0.774194</td>
      <td>0.912683</td>
      <td>0.964912</td>
      <td>0.964912</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.982051</td>
      <td>0.769231</td>
      <td>0.946153</td>
      <td>0.963889</td>
      <td>0.963889</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.911692</td>
      <td>0.769231</td>
      <td>0.911105</td>
      <td>0.945469</td>
      <td>0.982221</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.964572</td>
      <td>0.769231</td>
      <td>0.982221</td>
      <td>0.982051</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_1=np.mean(score_list_model1)
final_2=np.mean(score_list_model2)
final_3=np.mean(score_list_model3)
final_4=np.mean(score_list_model4)
final_5=np.mean(score_list_model5)

print("F1 Score Average Model_1",final_1)
print("F1 Score Average Model_2",final_2)
print("F1 Score Average Model_3",final_3)
print("F1 Score Average Model_4",final_4)
print("F1 Score Average Model_5",final_5)
```

    F1 Score Average Model_1 0.9523927704887092
    F1 Score Average Model_2 0.7710574943244813
    F1 Score Average Model_3 0.9333145924345627
    F1 Score Average Model_4 0.9540435232696123
    F1 Score Average Model_5 0.971905034024636
    

## Hyperparameter Search
### Purpose is to Optimize Model 5 (Gradient Boosting model)
#### 1. Get Current Params


```python
model_5.get_params()
```




    {'criterion': 'friedman_mse',
     'init': None,
     'learning_rate': 0.1,
     'loss': 'deviance',
     'max_depth': 3,
     'max_features': None,
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 100,
     'presort': 'auto',
     'random_state': None,
     'subsample': 1.0,
     'verbose': 0,
     'warm_start': False}



#### 2. Optimization in _'max depth'_ , _'min samples leaf'_
Using GridSearch CV


```python
from sklearn.model_selection import GridSearchCV
gb_tuned_params = {'max_depth' : [1, 2, 3, 4], 
                   'min_samples_leaf': [1, 3, 5],
                  'min_samples_split' : [2, 3, 5]}
GridGB = GridSearchCV(GradientBoostingClassifier(),gb_tuned_params, cv=5)
GridGB.fit(X,y)

print("Best Params : ",GridGB.best_params_)
print()
means = GridGB.cv_results_['mean_test_score']
stds = GridGB.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, GridGB.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
```

    Best Params :  {'max_depth': 1, 'min_samples_leaf': 5, 'min_samples_split': 2}
    
    0.963 (+/-0.037) for {'max_depth': 1, 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.963 (+/-0.037) for {'max_depth': 1, 'min_samples_leaf': 1, 'min_samples_split': 3}
    0.963 (+/-0.037) for {'max_depth': 1, 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.961 (+/-0.036) for {'max_depth': 1, 'min_samples_leaf': 3, 'min_samples_split': 2}
    0.961 (+/-0.036) for {'max_depth': 1, 'min_samples_leaf': 3, 'min_samples_split': 3}
    0.961 (+/-0.036) for {'max_depth': 1, 'min_samples_leaf': 3, 'min_samples_split': 5}
    0.968 (+/-0.030) for {'max_depth': 1, 'min_samples_leaf': 5, 'min_samples_split': 2}
    0.968 (+/-0.030) for {'max_depth': 1, 'min_samples_leaf': 5, 'min_samples_split': 3}
    0.968 (+/-0.030) for {'max_depth': 1, 'min_samples_leaf': 5, 'min_samples_split': 5}
    0.953 (+/-0.042) for {'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.954 (+/-0.043) for {'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 3}
    0.954 (+/-0.037) for {'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.951 (+/-0.051) for {'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 2}
    0.951 (+/-0.051) for {'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 3}
    0.951 (+/-0.051) for {'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 5}
    0.963 (+/-0.046) for {'max_depth': 2, 'min_samples_leaf': 5, 'min_samples_split': 2}
    0.963 (+/-0.046) for {'max_depth': 2, 'min_samples_leaf': 5, 'min_samples_split': 3}
    0.963 (+/-0.046) for {'max_depth': 2, 'min_samples_leaf': 5, 'min_samples_split': 5}
    0.961 (+/-0.045) for {'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.961 (+/-0.039) for {'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 3}
    0.961 (+/-0.052) for {'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.963 (+/-0.048) for {'max_depth': 3, 'min_samples_leaf': 3, 'min_samples_split': 2}
    0.960 (+/-0.045) for {'max_depth': 3, 'min_samples_leaf': 3, 'min_samples_split': 3}
    0.961 (+/-0.045) for {'max_depth': 3, 'min_samples_leaf': 3, 'min_samples_split': 5}
    0.968 (+/-0.037) for {'max_depth': 3, 'min_samples_leaf': 5, 'min_samples_split': 2}
    0.968 (+/-0.037) for {'max_depth': 3, 'min_samples_leaf': 5, 'min_samples_split': 3}
    0.968 (+/-0.037) for {'max_depth': 3, 'min_samples_leaf': 5, 'min_samples_split': 5}
    0.958 (+/-0.053) for {'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.963 (+/-0.055) for {'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 3}
    0.958 (+/-0.055) for {'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.960 (+/-0.037) for {'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 2}
    0.961 (+/-0.047) for {'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 3}
    0.956 (+/-0.036) for {'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 5}
    0.963 (+/-0.060) for {'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 2}
    0.963 (+/-0.067) for {'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 3}
    0.965 (+/-0.058) for {'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 5}
    

#### 3. Fit to find F1 Score using hyperparameter best params in model_5 GradientBoostingClassifier


```python
Optimized_model=GradientBoostingClassifier(max_depth=3,min_samples_leaf=5,min_samples_split=5) 

score_list_optimized=[]

for train_index, test_index in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    Optimized_model.fit(X_train, y_train)
    y_pred=Optimized_model.predict(X_test)
    score_list_optimized.append(
        f1_score(y_test,y_pred,average='weighted',labels=np.unique(y_pred)))
print()
print("F1 Score Optimized model : ",np.mean(score_list_optimized))
```

    
    F1 Score Optimized model :  0.969909500760872
    


# Conclusion

After Testing 5 Model of machine learning classifier and testing both using train test split and cross validation method, conclude that __Model 5__ which is __Gradient Boosting__ winth with crossvalidation F1 Score  __0.969__ , and Optimized parameter : 'max_depth': 1, 'min_samples_leaf': 5, 'min_samples_split': 2
