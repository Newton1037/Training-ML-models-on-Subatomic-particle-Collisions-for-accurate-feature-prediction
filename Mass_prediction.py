#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score , confusion_matrix


Data_frame = pd.read_csv(r"C:\Users\Ritesh\OneDrive\Documents\dielectron.csv")
Data_frame.head(5)



# In[2]:


Data_frame.info()


# In[3]:


Data_frame.describe()


# In[4]:


Data_frame.shape


# In[3]:


Data_frame.hist(bins = 100 , figsize = (20,15))
plt.show()


# In[2]:


Data_frame['E_total'] = Data_frame['E1'] + Data_frame['E2']
Data_frame.head(5)


# In[3]:


Data_frame.drop('Event' , axis=1 , inplace=True)
Data_frame.drop('Run' , axis=1 , inplace=True)


# In[4]:


corr_matrix = Data_frame.corr()
print(corr_matrix['M'].sort_values(ascending=True))


# In[10]:


Data_frame.plot(kind='scatter' , x='M' , y='pt1' , alpha=0.15)
plt.show()


# In[11]:


plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix , annot=True , fmt='.3f' , cmap='YlGnBu')
plt.title('Correlation matrix')
plt.show()


# In[19]:


from pandas.plotting import scatter_matrix

attributes = ['pt1' , 'pt2' , 'E1' , 'E2' , 'E_total' , 'M']
scatter_matrix(train_set[attributes] , figsize = (20,15))
plt.show()


# In[4]:


Data_frame['pt1_cat'] = pd.cut(Data_frame['pt1'], bins=[0, 10, 20, 30, 40, 50, np.inf], labels=[1, 2, 3, 4, 5, 6])                                        
plt.hist(Data_frame['pt1_cat'])
plt.show()


# In[5]:


from sklearn.model_selection import StratifiedShuffleSplit

str_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in str_split.split(Data_frame, Data_frame['pt1_cat']):
    train_set = Data_frame.loc[train_index]
    test_set = Data_frame.loc[test_index]
    
train_set


# In[6]:


test_set


# In[9]:


Data_frame


# In[7]:


for set in (train_set , test_set):
    set.drop('pt1_cat' , axis=1 , inplace=True)
    
train_set.info()


# In[8]:


train_set2 = train_set.dropna(subset=['M'] , inplace=False)
train_set2.info()


# In[9]:


train_set.info()


# In[10]:


test_set2 = test_set.dropna(subset=['M'] , inplace=False)
test_set2.info()


# In[11]:


# Separate label data (output or the target variable that the ML model aims to predict)
# and predictors (input variables that provide information to the machine learning model)
# The goal of training a ML model is to learn a mapping from the predictors to the label data 
# so that the model can make accurate predictions on new, unseen data.

train_predictors = train_set2.drop('M' , axis=1 , inplace=False)
test_predictors = test_set2.drop('M' , axis=1 , inplace=False)
train_label = train_set2['M'].copy()
test_label = test_set2['M'].copy()


# # Data preprocessing is done below

# In[12]:


# Scaling the training dataframe(predictor) to have more precide training of the ML model
# StandardScaler() is used to perform standardization on the features,
# which means transforming the data so that it has a mean of 0 and a standard deviation of 1.
# cern_prepared will contain the scaled predictor data, which is now ready for use in training machine learning models
from sklearn.preprocessing import OneHotEncoder, StandardScaler

scale = StandardScaler()
cern_prepared = scale.fit_transform(train_predictors)
cern_test_prepared = scale.transform(test_predictors)


# In[13]:


from sklearn.preprocessing import PolynomialFeatures

cern_poly = PolynomialFeatures(2)
data_prepared = cern_poly.fit_transform(train_predictors)
data_test_prepared = cern_poly.transform(test_predictors)


# # We will start with training and evaluation of Regression ML models
# ### 1st is Decision Tree Regressor on training set with predict()

# In[17]:


from sklearn.tree import DecisionTreeRegressor

tree_regress = DecisionTreeRegressor(max_depth = 20)
tree_regress.fit(cern_prepared , train_label)


# In[18]:


tree_train_pred = tree_regress.predict(cern_prepared)
tree_train_mse = mean_squared_error(tree_train_pred , train_label)
tree_train_rmse = np.sqrt(tree_train_mse)

print("Training set Decision Tree RMSE value : ",tree_train_rmse)
 


# ### evaluating using predict() on testing set

# In[19]:


tree_reg = DecisionTreeRegressor(max_depth = 20)
tree_reg.fit(cern_test_prepared , test_label)


# In[20]:


tree_test_pred = tree_reg.predict(cern_test_prepared)
tree_test_mse = mean_squared_error(tree_test_pred , test_label)

tree_test_rmse = np.sqrt(tree_test_mse)

print("Testing set Decision Tree RMSE value : ",tree_test_rmse)


# ## To check overfitting and underfitting we compare Training and Test Performance

# In[21]:


difference = tree_train_rmse - tree_test_rmse
print("Difference between rmse values of training and testing sets is ",difference)


# ### Evaluating by Decision Tree Regressor on the training and testing set by Cross Validation

# In[22]:


tree_train_mse = cross_val_score(tree_regress , cern_prepared , train_label , cv=10 , scoring='neg_mean_squared_error')
tree_test_mse = cross_val_score(tree_reg , cern_test_prepared , test_label , cv=10 , scoring='neg_mean_squared_error')

tree_train_rmse = np.sqrt(-tree_train_mse)
tree_test_rmse = np.sqrt(-tree_test_mse)

print("Training set Decision Tree RMSE values : " , tree_train_rmse)
print("Testing set Decision Tree RMSE values : ",tree_test_rmse)


# ## To check overfitting and underfitting we compare Training and Test Performance

# In[23]:


diff = tree_test_rmse - tree_train_rmse
print("Difference between rmse values of training and testing sets is ",diff)


# ### Now we check our model performance by giving polynomial features as input in the dataset i.e. dataset is quadratic

# In[24]:


tree_r = DecisionTreeRegressor(max_depth=20)
tree_r.fit(data_prepared , train_label)


# In[25]:


tree_r_test = DecisionTreeRegressor(max_depth=20)
tree_r_test.fit(data_test_prepared , test_label)


# In[26]:


tr_train_mse = cross_val_score(tree_r , data_prepared , train_label , cv=10 , scoring='neg_mean_squared_error')
tr_test_mse = cross_val_score(tree_r_test , data_test_prepared , test_label , cv=10 , scoring='neg_mean_squared_error')

tr_train_rmse = np.sqrt(-tr_train_mse)
tr_test_rmse = np.sqrt(-tr_test_mse)

print("Quadratic Training set Decision Tree RMSE values : " , tr_train_rmse)
print("Quadratic Testing set Decision Tree RMSE values : ",tr_test_rmse)


# In[27]:


diffe = tr_test_rmse - tr_train_rmse
print("Difference between rmse values of quadratic training and testing sets is ",diffe)


# ## This is Feature Importance Analysis with Decision Tree Regressor

# In[17]:


def train_and_evaluate_tree(X, y, max_depth):
    tree_regressor_feat = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree_regressor_feat.fit(X, y)

    feature_importances = tree_regressor_feat.feature_importances_

    tree_mse_feat = cross_val_score(tree_regressor_feat, X, y, cv=5, scoring='neg_mean_squared_error')
    tree_rmse_feat = np.sqrt(-tree_mse_feat.mean())

    return tree_rmse_feat, feature_importances

max_depth_values = [None, 10, 15, 30, 50]

results = {'Max Depth': [], 'Tree RMSE': []}

for max_depth in max_depth_values:
    tree_rmse_feat, feature_importance = train_and_evaluate_tree(cern_prepared, train_label, max_depth)
    results['Max Depth'].append(max_depth)
    results['Tree RMSE'].append(tree_rmse_feat)

results_decisionTree = pd.DataFrame(results)

best_max_depth = 50

best_tree_rmse, best_feature_importances = train_and_evaluate_tree(cern_prepared, train_label, best_max_depth)

print(f"Best Max Depth: {best_max_depth}")
print(f"Best Tree RMSE: {best_tree_rmse}")

feature_importance_df = pd.DataFrame({'Feature': range(cern_prepared.shape[1]), 'Importance': best_feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance of Decision Tree')
plt.xticks(rotation=45, ha='right')
plt.show()


# # 2nd Random Forest Regressor on training set using predict()

# In[40]:


from sklearn.ensemble import RandomForestRegressor

forest_regress = RandomForestRegressor(max_depth=20)
forest_regress.fit(cern_prepared , train_label)


# In[31]:


forest_train_pred = forest_regress.predict(cern_prepared)
forest_train_mse = mean_squared_error(forest_train_pred , train_label)
forest_train_rmse = np.sqrt(forest_train_mse)

print("Training set Random Forest RMSE value : ",forest_train_rmse)


# ##  Random Forest Regressor on testing set using predict()
# 

# In[41]:


forest_reg = RandomForestRegressor(max_depth=20)
forest_reg.fit(cern_test_prepared , test_label)


# In[33]:


forest_test_pred = forest_reg.predict(cern_test_prepared)
forest_test_mse = mean_squared_error(forest_test_pred , test_label)
forest_test_rmse = np.sqrt(forest_test_mse)

print("Testing set Random Forest RMSE value : ",forest_train_rmse)


# ### To check overfitting and underfitting we compare Training and Test Performance

# In[34]:


comparison = forest_test_rmse - forest_train_rmse
print("Difference between rmse values of training and testing sets is " , comparison)


# ### Evaluating by Random Forest Regressor on the training and testing set by Cross Validation

# In[35]:


forest_train_mse = cross_val_score(forest_regress , cern_prepared , train_label , cv=10 , scoring='neg_mean_squared_error')
forest_test_mse = cross_val_score(forest_reg , cern_test_prepared , test_label , cv=10 , scoring='neg_mean_squared_error')

forest_train_rmse = np.sqrt(-forest_train_mse)
forest_test_rmse = np.sqrt(-forest_test_mse)

forest_train_rmse
forest_test_rmse


# In[36]:


compare = forest_test_rmse - forest_train_rmse
print("Difference between rmse values of training and testing sets via cross validation is \n" , compare)


# ## Now we will do feature importance analysis with Random Forest Regressor

# In[34]:


def train_and_evaluate_forest(X, y, max_depth):
    for_regressor_feat = RandomForestRegressor(max_depth=max_depth, random_state=42)
    for_regressor_feat.fit(X, y)

  
    feature_importances = for_regressor_feat.feature_importances_

    for_mse_feat = cross_val_score(for_regressor_feat, X, y, cv=5, scoring='neg_mean_squared_error')
    for_rmse_feat = np.sqrt(-for_mse_feat.mean())

    return for_rmse_feat, feature_importances


max_depth_values = [None, 10, 15, 30, 50]


results = {'Max Depth': [], 'Forest RMSE': []}


for max_depth in max_depth_values:
    for_rmse_feat, feature_importance = train_and_evaluate_forest(cern_prepared, train_label, max_depth)
    results['Max Depth'].append(max_depth)
    results['Forest RMSE'].append(for_rmse_feat)


results_decisionTree = pd.DataFrame(results)
best_max_depth = 50

best_tree_rmse, best_feature_importances = train_and_evaluate_forest(cern_prepared, train_label, best_max_depth)

print(f"Best Max Depth: {best_max_depth}")
print(f"Best Forest RMSE: {best_tree_rmse}")

feature_importance_df = pd.DataFrame({'Feature': range(cern_prepared.shape[1]), 'Importance': best_feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance of Random Forest')
plt.xticks(rotation=45, ha='right')
plt.show()


# # Now we will start Classification ML models
# ###  Use numpy's digitize to convert continuous values to discrete classes
# 

# In[14]:


bin_edges = [0, 1, 2, 3, 4, 5]  
bin_labels = [0, 1, 2, 3, 4]  

train_label_indices = np.digitize(train_label, bin_edges, right=True)
test_label_indices = np.digitize(test_label, bin_edges, right=True)

train_label_discrete = [bin_labels[i - 1] if i > 0 and i <= len(bin_labels) else 'Invalid' for i in train_label_indices]
test_label_discrete = [bin_labels[i - 1] if i > 0 and i <= len(bin_labels) else 'Invalid' for i in test_label_indices]



# ## Training and evaluation of the Classification ML models 
# ### 1st is KNN Classifier on training set by predict()
# 

# In[15]:


from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(cern_prepared , train_label_discrete)


# In[18]:


knn_train_predictions = knn.predict(cern_prepared)
knn_train_accuracy = accuracy_score(train_label_discrete , knn_train_predictions)
knn_train_confusion_matrix = confusion_matrix(train_label_discrete , knn_train_predictions)


print("KNN Accuracy for training set : ", knn_train_accuracy)
print("KNN confusion matrix for training set : \n" , knn_train_confusion_matrix)


# ### Now we check for testing set by predict()

# In[26]:


knn_ag = KNeighborsClassifier(n_neighbors = 3)
knn_ag.fit(cern_test_prepared , test_label_discrete)


# In[20]:


knn_test_predictions = knn_ag.predict(cern_test_prepared)
knn_test_accuracy = accuracy_score(test_label_discrete , knn_test_predictions)
knn_test_confusion_matrix = confusion_matrix(test_label_discrete , knn_test_predictions)


print("KNN Accuracy for testing set : ", knn_test_accuracy)
print("KNN confusion matrix for testing set : \n" , knn_test_confusion_matrix)


# #### To check overfitting and underfitting we compare Training and Test Performance¶

# In[21]:


accuracy_difference = knn_train_accuracy - knn_test_accuracy
print("Difference between accuracy score of training and testing sets is " , accuracy_difference)


# ### We evaluate the KNN model by using Cross Validation in both Training and Testing Dataset

# In[27]:


knn_train_scores = cross_val_score(knn , cern_prepared , train_label_discrete , cv=10 , scoring='accuracy')
knn_test_scores = cross_val_score(knn_ag , cern_test_prepared , test_label_discrete , cv=10 , scoring='accuracy')

print("KNN Accuracy for training set : " , knn_train_scores)
print("KNN Accuracy for testing set : " , knn_test_scores)


# ### To check overfitting and underfitting we compare Training and Test Performance¶

# In[28]:


valid_difference = knn_test_scores - knn_train_scores
print("Difference between knn scores of training and testing sets is " , valid_difference)


# ## Feature Importance Analysis of KNN Classifier 

# In[35]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

def train_and_evaluate_knn(X, y, n_neighbors):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X, y)
    
    knn_predictions = knn.predict(X)
    knn_rmse = np.sqrt(mean_squared_error(y, knn_predictions))

    feature_importances = [np.abs(np.corrcoef(X[:, i], knn_predictions)[0, 1]) for i in range(X.shape[1])]
    
    return knn_rmse, feature_importances


n_neighbors_values = [3, 5, 7, 10]

knn_results = {'Neighbors': [], 'KNN RMSE': []}

for n_neighbors in n_neighbors_values:
    knn_rmse, feature_importances = train_and_evaluate_knn(cern_prepared, train_label, n_neighbors)
    knn_results['Neighbors'].append(n_neighbors)
    knn_results['KNN RMSE'].append(knn_rmse)

knn_results_df = pd.DataFrame(knn_results)


best_neighbors_index = knn_results_df['KNN RMSE'].idxmin()
best_neighbors_value = knn_results_df.loc[best_neighbors_index, 'Neighbors']
best_knn_rmse, best_feature_importances = train_and_evaluate_knn(cern_prepared, train_label, best_neighbors_value)

feature_importance_df_knn = pd.DataFrame({'Feature': range(cern_prepared.shape[1]), 'Importance': best_feature_importances})
feature_importance_df_knn = feature_importance_df_knn.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df_knn['Feature'], feature_importance_df_knn['Importance'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance of K Nearest Neighbors (KNN)')
plt.xticks(rotation=45, ha='right')
plt.show()


# # 2nd is SVC Classifier
# ### First training and evaluation on training set by predict()

# In[35]:


from sklearn.svm import SVC

svc = SVC(kernel='linear' , C=1.0) 
svc.fit(cern_prepared , train_label_discrete)


# In[36]:


svc_train_predictions = svc.predict(cern_prepared)
svc_train_accuracy = accuracy_score(train_label_discrete , svc_train_predictions)
svc_train_confusion_matrix = confusion_matrix(train_label_discrete , svc_train_predictions)

print("SVC Accuracy for training set :", svc_train_accuracy)
print("SVC Confusion matrix for training set : \n", svc_train_confusion_matrix)


# ### Now we check test set performance by predict()

# In[37]:


svc_cg = SVC(kernel='linear' , C=1.0)
svc_cg.fit(cern_test_prepared , test_label_discrete)


# In[38]:


svc_test_predictions = svc_cg.predict(cern_test_prepared)
svc_test_accuracy = accuracy_score(test_label_discrete , svc_test_predictions)
svc_test_confusion_matrix = confusion_matrix(test_label_discrete , svc_test_predictions)

print("SVC Accuracy for testing set :", svc_test_accuracy)
print("SVC Confusion matrix for testing set : \n", svc_test_confusion_matrix)


# #### Now we check overfitting or underfitting by comparing training and test performance

# In[34]:


svc_accuracy_difference = svc_train_accuracy - svc_test_accuracy
print("Difference between accuracy score of training and testing sets is " , svc_accuracy_difference)


# ## We evaluate the KNN model by using Cross Validation in both Training and Testing Dataset

# In[22]:


svc_train_scores = cross_val_score(svc , cern_prepared , train_label_discrete , cv=10 , scoring='accuracy')
svc_test_scores = cross_val_score(svc_cg , cern_test_prepared , test_label_discrete , cv=10 , scoring='accuracy')

print("KNN Accuracy for training set : " , svc_train_scores)
print("KNN Accuracy for testing set : " , svc_test_scores)


# In[24]:


svc_valid_difference = svc_test_scores - svc_train_scores
print("Difference between knn scores of training and testing sets is " , svc_valid_difference)


# ## Feature Importance Analysis of SVC Classifier

# In[37]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_label_discrete_encoded = label_encoder.fit_transform(train_label_discrete)

def train_and_evaluate_svc(X, y, C):
    svc_feat = SVC(kernel='linear', C=C)
    svc_feat.fit(X, y)

    svc_mse_feat = cross_val_score(svc_feat, X, y, cv=5, scoring='neg_mean_squared_error')
    svc_rmse_feat = np.sqrt(-svc_mse_feat.mean())

    coefficients = np.abs(svc_feat.coef_.flatten())

    return svc_rmse_feat, coefficients

C_values = [0.1, 1.0, 10.0]

svc_results = {'C': [], 'SVC RMSE': [], 'Feature Importances': []}

for C_value in C_values:
    svc_rmse_feat, feature_importances = train_and_evaluate_svc(cern_prepared, train_label_discrete_encoded, C_value)
    svc_results['C'].append(C_value)
    svc_results['SVC RMSE'].append(svc_rmse_feat)
    svc_results['Feature Importances'].append(feature_importances)

svc_results_df = pd.DataFrame(svc_results)

svc_results_df = svc_results_df.dropna()

print("\nSupport Vector Regressor (SVR) Results:")
print(svc_results_df)

# ...

if not svc_results_df.empty:
    best_C_index = svc_results_df['SVC RMSE'].idxmin()
    best_C_value = svc_results_df.loc[best_C_index, 'C']
    best_svc_rmse, best_feature_importances = train_and_evaluate_svc(cern_prepared, train_label_discrete_encoded, best_C_value)

    if len(range(cern_prepared.shape[1])) == len(best_feature_importances):
        feature_importance_df_svc = pd.DataFrame({'Feature': range(cern_prepared.shape[1]), 'Importance': best_feature_importances})
        feature_importance_df_svc = feature_importance_df_svc.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance_df_svc['Feature'], feature_importance_df_svc['Importance'])
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance of Support Vector Regressor (SVR)')
        plt.xticks(rotation=45, ha='right')
        plt.show()
    else:
        print("Error: 'Feature' and 'Importance' arrays have different lengths.")
else:
    print("DataFrame is empty. No results to display.")

