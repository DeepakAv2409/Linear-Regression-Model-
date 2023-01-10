#!/usr/bin/env python
# coding: utf-8

# ### NAME : DEEPAK AV
#     
# ### Linear regression
# 
# ### Problem 1:compactiv.xlsx

# ## We will construct a linear model that can predict a car's mileage (usr) by using its other attributes.

# 
# ### Import Libraries

# In[1]:


# Importing all the libraries for linear regression.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ### Load and explore the data

# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


# Loading the data to the python project. 
df=pd.read_excel('compactiv.xlsx')


# In[5]:


df


# # 1.1 Read the data and do exploratory data analysis. Describe the data briefly. (Check the Data types, shape, EDA, 5 point summary). Perform Univariate, Bivariate Analysis, Multivariate Analysis.
# 

# In[6]:


# let's check the first 5 rows of the data.
df.head()


# In[7]:


# let's check the last 5 rows of the data.
df.tail()


# In[8]:


# let's check column types and number of values
df.info()


# In[9]:


# let's check the shape of the data
df.shape


# In[10]:


# Analysing the 5 point summary of the data.
df.describe()


# In[11]:


# Analysing the null values. 
df.isnull()


# In[12]:


# Calculating the null values present in each column of the data.
df.isnull().sum()


# In[13]:


df.columns


# ### UNIVARIATE ANALYSIS

# In[14]:


# Analysing the data with the method of univariate analysis with the boxplot.
plt.figure(figsize=(20,5))
sns.boxplot(data=df)


# ### BIVARIATE ANALYSIS

# In[15]:


# Analysing the data with the method of bivariate analysis with the scatterplot.
plt.figure(figsize=(20,10))
sns.scatterplot(data=df)


# ### MULTIVARIATE ANALYSIS

# In[16]:


# Analysing the data with the method of multivariate analysis with the pairplot.
plt.figure(figsize=(10,5))
sns.pairplot(data=df,size=2) 


# # 1.2 Impute null values if present, also check for the values which are equal to zero. Do they have any meaning or do we need to change them or drop them? Check for the possibility of creating new features if required. Also check for outliers and duplicates if there.

# In[17]:


# Calculating the null values present in each column of the data.
df.isnull().sum()


# #### Inference : There are 104 null values present in 'rchar' and 15 null values present in 'wchar' features in the dataset.

# In[18]:


# Imputing the null values present in the data with median.
df['rchar'].fillna(df['rchar'].median(), inplace = True)


# #### Inference : Imputing the null values present in 'rchar' with median.

# In[19]:


# Imputing the null values present in the data with median.
df['wchar'].fillna(df['wchar'].median(), inplace = True)


# #### Inference : Imputing the null values present in 'wchar' with median.

# In[20]:


# After imputing checking the null values in the data.
df.isnull().sum()


# In[21]:


# Analysing the data wheather it contains zero values or not ?
(df == 0).sum()


# In[22]:


(df == 0).sum().sum()


# #### Inference : There are 31,775 zero values present in the datset.

# In[23]:


#Printing tha zero values present in the data.
df.loc[~(df==0).all(axis=1)]


# In[24]:


# Dropping the categorical feature present in the data.
df=df.drop(["runqsz"], axis=1)


# #### Inference : Dropping the categorical value to treat the outliers.

# In[25]:


# Checking the outliers through boxplot.
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.boxplot(data=df)


# In[26]:


# Removing the outliers using interquartile range.
def remove_outlier(col):
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range 


# In[27]:


# Treating the outliers
for i in df.columns:
    LL, UL = remove_outlier(df[i])
    df[i] = np.where(df[i] > UL, UL, df[i])
    df[i] = np.where(df[i] < LL, LL, df[i])


# In[28]:


# Plotting a boxplot to chech the outliers is removed or not?
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.boxplot(data=df)


# In[29]:


#### Inference :


# In[30]:


# Checking for the duplicate values present in the dataset.
df.duplicated().sum()


# In[31]:


#### Inference :


# In[32]:


# Dropping the 'pgscan' feature in the datset.
df=df.drop(["pgscan"], axis=1)


# # 1.3 Encode the data (having string values) for Modelling. Split the data into train and test (70:30). Apply Linear regression using scikit learn. Perform checks for significant variables using appropriate method from statsmodel. Create multiple models and check the performance of Predictions on Train and Test sets using Rsquare, RMSE & Adj Rsquare. Compare these models and select the best one with appropriate reasoning.

# In[33]:


df_corr = df.corr()
df_corr


# In[ ]:





# In[34]:


plt.figure(figsize=(20,20))
sns.heatmap(df_corr, annot=True)


# In[35]:


df.columns


# ### Split Data

# In[36]:


# independent variables
X=df[['lread', 'lwrite', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar',
       'wchar', 'pgout', 'ppgout', 'pgfree', 'atch', 'pgin', 'ppgin', 'pflt',
       'vflt', 'freemem', 'freeswap']]
# dependent variable
y=df['usr']


# In[ ]:





# In[37]:


# let's add the intercept to data
X = sm.add_constant(X)


# #### We will now split X and y into train and test sets in a 70:30 ratio.

# In[38]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[39]:


print(X_train.head())


# In[40]:


print(X_test.head())


# #### Fit Linear Model

# In[41]:


#fitting the linear model
fitmod= sm.OLS(y_train,X_train)
fitres=fitmod.fit()


# In[42]:


# let's print the regression summary
print(fitres.summary())


# ### Interpretation of R-squared
# ####      -The R-squared value tells us that our model can explain 79.0% of the variance in the training set.

# In[43]:


# checking the varience inflation factor of the predictors\
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_series1 = pd.Series(
  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
  index=X_train.columns,
)
print('VIF values: \n\n{}\n'.format(vif_series1))


# #### Let's remove/drop multicollinear columns one by one and observe the effect on our predictive model.¶

# In[44]:


X_train2 =X_train.drop(["ppgout"],axis=1)
fitmod_1=sm.OLS(y_train, X_train2)
fitres_1=fitmod_1.fit()
print(
    "R-squared:",
    np.round(fitres_1.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_1.rsquared,5),
)


# #### Inference : On dropping 'ppgout', adj. R-squared almost remains the same.

# In[45]:


X_train3 = X_train.drop(["pgfree"], axis=1)
fitmod_2 = sm.OLS(y_train, X_train3)
fitres_2 = fitmod_2.fit()
print(
    "R-squared:",
    np.round(fitres_2.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_2.rsquared_adj, 3),
)


# #### Inference : On dropping 'pgfree', adj. R-squared almost remains the same.

# In[46]:


X_train4 = X_train.drop(["vflt"], axis=1)
fitmod_3 = sm.OLS(y_train, X_train4)
fitres_3 = fitmod_3.fit()
print(
    "R-squared:", 
    np.round(fitres_3.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_3.rsquared_adj, 3),
)


# #### Inference : On dropping 'vflt', adj. R-squared decresed by 0.001

# In[47]:


X_train5 = X_train.drop(["ppgin"], axis=1)
fitmod_4 = sm.OLS(y_train, X_train5)
fitres_4 = fitmod_4.fit()
print(
    "R-squared:", 
    np.round(fitres_4.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_4.rsquared_adj, 3),
)


# #### Inference : On dropping 'ppgin', adj. R-squared decresed by 0.001

# In[48]:


X_train6 = X_train.drop(["pgin"], axis=1)
fitmod_5 = sm.OLS(y_train, X_train6)
fitres_5 = fitmod_5.fit()
print(
    "R-squared:", 
    np.round(fitres_5.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_5.rsquared_adj, 3),
)


# #### Inference : On dropping 'pgin', adj. R-squared almost remains the same.

# In[49]:


X_train7 = X_train.drop(["fork"], axis=1)
fitmod_6 = sm.OLS(y_train, X_train7)
fitres_6 = fitmod_6.fit()
print(
    "R-squared:", 
    np.round(fitres_6.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_6.rsquared_adj, 3),
)


# #### Inference : On dropping 'fork', adj. R-squared almost remains the same.

# In[50]:


X_train8 = X_train.drop(["pflt"], axis=1)
fitmod_7 = sm.OLS(y_train, X_train8)
fitres_7 = fitmod_7.fit()
print(
    "R-squared:", 
    np.round(fitres_7.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_7.rsquared_adj, 3),
)


# #### Inference : On dropping 'pflt', adj. R-squared decreased by 0.011
# ####             This sharp decline indicates that 'pflt' is an important predictor and shouldn't be removed.

# In[51]:


X_train9 = X_train.drop(["pgout"], axis=1)
fitmod_8 = sm.OLS(y_train, X_train9)
fitres_8 = fitmod_8.fit()
print(
    "R-squared:", 
    np.round(fitres_8.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_8.rsquared_adj, 3),
)


# #### Inference : On dropping 'pgout', adj. R-squared decresed by 0.001

# In[52]:


X_train10 = X_train.drop(["sread"], axis=1)
fitmod_9 = sm.OLS(y_train, X_train10)
fitres_9 = fitmod_9.fit()
print(
    "R-squared:", 
    np.round(fitres_8.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_8.rsquared_adj, 3),
)


# #### Inference : On dropping 'sread', adj. R-squared decresed by 0.001

# In[53]:


X_train = X_train.drop(["ppgout"], axis=1)


# #### Inference : Since there is no effect on adj. R-squared after dropping the 'ppgout' column, and it has highest number in value of varience influence factor, so we remove it from the training set.

# In[54]:


fitmod_10 = sm.OLS(y_train, X_train)
fitres_10 = fitmod_10.fit()
print(fitres_10.summary())


# #### Inference : -The R-squared value tells us that our model can explain 79.0% of the variance in the training set.

# ## Let's check if multicollinearity is still present in the data.

# In[55]:


vif_series2 = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
)
print("VIF values: \n\n{}\n".format(vif_series2))


# In[56]:


X_train11 = X_train.drop(["vflt"], axis=1)
fitmod_11 = sm.OLS(y_train, X_train11)
fitres_11 = fitmod_11.fit()
print(
    "R-squared:", 
    np.round(fitres_11.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_11.rsquared_adj, 3),
)


# #### Inference : On dropping 'vflt', adj. R-squared decresed by 0.001

# In[57]:


X_train12 = X_train.drop(["ppgin"], axis=1)
fitmod_12 = sm.OLS(y_train, X_train12)
fitres_12 = fitmod_12.fit()
print(
    "R-squared:", 
    np.round(fitres_12.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_12.rsquared_adj, 3),
)


# In[58]:


X_train13 = X_train.drop(["pgin"], axis=1)
fitmod_13 = sm.OLS(y_train, X_train13)
fitres_13 = fitmod_13.fit()
print(
    "R-squared:", 
    np.round(fitres_13.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_13.rsquared_adj, 3),
)


# #### Inference : On dropping 'pgin', adj. R-squared almost remains the same.

# In[59]:


X_train14 = X_train.drop(["fork"], axis=1)
fitmod_14 = sm.OLS(y_train, X_train14)
fitres_14 = fitmod_14.fit()
print(
    "R-squared:", 
    np.round(fitres_14.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_14.rsquared_adj, 3),
)


# #### Inference : On dropping 'fork', adj. R-squared almost remains the same.

# In[60]:


X_train15 = X_train.drop(["pflt"], axis=1)
fitmod_15 = sm.OLS(y_train, X_train15)
fitres_15 = fitmod_15.fit()
print(
    "R-squared:", 
    np.round(fitres_15.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_15.rsquared_adj, 3),
)


# #### Inference : On dropping 'pflt', adj. R-squared decreased by 0.011
# ####                    This sharp decline indicates that 'pflt' is an important predictor and shouldn't be removed.

# In[61]:


X_train16 = X_train.drop(["sread"], axis=1)
fitmod_16 = sm.OLS(y_train, X_train16)
fitres_16 = fitmod_16.fit()
print(
    "R-squared:", 
    np.round(fitres_16.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_16.rsquared_adj, 3),
)


# #### Inference : On dropping 'sread', adj. R-squared almost remains the same.

# In[62]:


X_train17 = X_train.drop(["sread"], axis=1)
fitmod_17 = sm.OLS(y_train, X_train17)
fitres_17 = fitmod_17.fit()
print(
    "R-squared:", 
    np.round(fitres_17.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_17.rsquared_adj, 3),
)


# In[63]:


X_train = X_train.drop(["pgin"], axis=1)


# #### Inference : Since there is no effect on adj. R-squared after dropping the 'pgin' column, and it has highest number in value of varience influence factor, so we remove it from the training set.

# In[64]:


fitmod_18 = sm.OLS(y_train, X_train)
fitres_18 = fitmod_18.fit()
print(fitres_18.summary())


# #### Inference : -The R-squared value tells us that our model can explain 79.0% of the variance in the training set.

# ## Let's check if multicollinearity is still present in the data.

# In[65]:


vif_series3 = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
)
print("VIF values: \n\n{}\n".format(vif_series3))


# ### Let's remove/drop multicollinear columns one by one and observe the effect on our predictive model.

# In[66]:


X_train18 = X_train.drop(["vflt"], axis=1)
fitmod_19 = sm.OLS(y_train, X_train18)
fitres_19 = fitmod_19.fit()
print(
    "R-squared:", 
    np.round(fitres_19.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_19.rsquared_adj, 3),
)


# #### Inference : On dropping 'vflt', adj. R-squared decresed by 0.001

# In[67]:


X_train19 = X_train.drop(["fork"], axis=1)
fitmod_20 = sm.OLS(y_train, X_train19)
fitres_20 = fitmod_20.fit()
print(
    "R-squared:", 
    np.round(fitres_20.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_20.rsquared_adj, 3),
)


# #### Inference : On dropping 'fork', adj. R-squared almost remains the same.

# In[68]:


X_train20 = X_train.drop(["pflt"], axis=1)
fitmod_21 = sm.OLS(y_train, X_train20)
fitres_21 = fitmod_21.fit()
print(
    "R-squared:", 
    np.round(fitres_21.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_21.rsquared_adj, 3),
)


# #### Inference : On dropping 'pflt', adj. R-squared decreased by 0.011

# In[69]:


X_train21 = X_train.drop(["sread"], axis=1)
fitmod_22 = sm.OLS(y_train, X_train21)
fitres_22 = fitmod_22.fit()
print(
    "R-squared:", 
    np.round(fitres_22.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_22.rsquared_adj, 3),
)


# #### Inference : On dropping 'sread', adj. R-squared almost remains the same.

# In[70]:


X_train = X_train.drop(["fork"], axis=1)


# In[71]:


fitmod_23 = sm.OLS(y_train, X_train)
fitres_23 = fitmod_23.fit()
print(fitres_23.summary())


# ## Let's check if multicollinearity is still present in the data.

# In[72]:


vif_series4 = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
)
print("VIF values: \n\n{}\n".format(vif_series4))


# In[73]:


X_train22 = X_train.drop(["vflt"], axis=1)
fitmod_24 = sm.OLS(y_train, X_train21)
fitres_24 = fitmod_24.fit()
print(
    "R-squared:", 
    np.round(fitres_24.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_24.rsquared_adj, 3),
)


# In[74]:


X_train23 = X_train.drop(["pflt"], axis=1)
fitmod_25 = sm.OLS(y_train, X_train23)
fitres_25 = fitmod_25.fit()
print(
    "R-squared:", 
    np.round(fitres_25.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_25.rsquared_adj, 3),
)


# In[75]:


X_train24 = X_train.drop(["pgout"], axis=1)
fitmod_26 = sm.OLS(y_train, X_train24)
fitres_26 = fitmod_26.fit()
print(
    "R-squared:", 
    np.round(fitres_26.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_26.rsquared_adj, 3),
)


# In[76]:


X_train25 = X_train.drop(["sread"], axis=1)
fitmod_27 = sm.OLS(y_train, X_train25)
fitres_27 = fitmod_27.fit()
print(
    "R-squared:", 
    np.round(fitres_27.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_27.rsquared_adj, 3),
)


# In[77]:


X_train26 = X_train.drop(["pgfree"], axis=1)
fitmod_28 = sm.OLS(y_train, X_train26)
fitres_28 = fitmod_28.fit()
print(
    "R-squared:", 
    np.round(fitres_28.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_28.rsquared_adj, 3),
)


# In[78]:


X_train27 = X_train.drop(["swrite"], axis=1)
fitmod_30 = sm.OLS(y_train, X_train27)
fitres_30 = fitmod_30.fit()
print(
    "R-squared:", 
    np.round(fitres_30.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_30.rsquared_adj, 3),
)


# In[79]:


X_train28 = X_train.drop(["lread"], axis=1)
fitmod_31 = sm.OLS(y_train, X_train28)
fitres_31 = fitmod_31.fit()
print(
    "R-squared:", 
    np.round(fitres_31.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_31.rsquared_adj, 3),
)


# In[80]:


X_train = X_train.drop(["vflt"], axis=1)


# In[81]:


fitmod_32 = sm.OLS(y_train, X_train)
fitres_32 = fitmod_32.fit()
print(fitres_32.summary())


# ## Let's check if multicollinearity is still present in the data.

# In[82]:


vif_series5 = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
)
print("VIF values: \n\n{}\n".format(vif_series5))


# In[83]:


X_train29 = X_train.drop(["pgout"], axis=1)
fitmod_33 = sm.OLS(y_train, X_train29)
fitres_33 = fitmod_33.fit()
print(
    "R-squared:", 
    np.round(fitres_33.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_33.rsquared_adj, 3),
)


# In[84]:


X_train30 = X_train.drop(["sread"], axis=1)
fitmod_34 = sm.OLS(y_train, X_train30)
fitres_34 = fitmod_34.fit()
print(
    "R-squared:", 
    np.round(fitres_34.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_34.rsquared_adj, 3),
)


# In[85]:


X_train31 = X_train.drop(["pgfree"], axis=1)
fitmod_35 = sm.OLS(y_train, X_train31)
fitres_35 = fitmod_35.fit()
print(
    "R-squared:", 
    np.round(fitres_35.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_35.rsquared_adj, 3),
)


# In[86]:


X_train32 = X_train.drop(["lread"], axis=1)
fitmod_36 = sm.OLS(y_train, X_train32)
fitres_36 = fitmod_36.fit()
print(
    "R-squared:", 
    np.round(fitres_36.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_36.rsquared_adj, 3),
)


# In[87]:


X_train = X_train.drop(["sread"], axis=1)


# In[88]:


fitmod_37 = sm.OLS(y_train, X_train)
fitres_37 = fitmod_37.fit()
print(fitres_37.summary())


# ## Let's check if multicollinearity is still present in the data.

# In[89]:


vif_series6 = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
)
print("VIF values: \n\n{}\n".format(vif_series6))


# In[90]:


X_train33 = X_train.drop(["pgout"], axis=1)
fitmod_38 = sm.OLS(y_train, X_train33)
fitres_38 = fitmod_38.fit()
print(
    "R-squared:", 
    np.round(fitres_38.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_38.rsquared_adj, 3),
)


# In[91]:


X_train34 = X_train.drop(["pgfree"], axis=1)
fitmod_39 = sm.OLS(y_train, X_train34)
fitres_39 = fitmod_39.fit()
print(
    "R-squared:", 
    np.round(fitres_39.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_39.rsquared_adj, 3),
)


# In[92]:


X_train35 = X_train.drop(["lread"], axis=1)
fitmod_40 = sm.OLS(y_train, X_train35)
fitres_40 = fitmod_40.fit()
print(
    "R-squared:", 
    np.round(fitres_40.rsquared, 3),
    "\nAdjusted R-squared:",
    np.round(fitres_40.rsquared_adj, 3),
)


# In[93]:


X_train = X_train.drop(["pgfree"], axis=1)


# In[94]:


fitmod_41 = sm.OLS(y_train, X_train)
fitres_41 = fitmod_41.fit()
print(fitres_41.summary())


# ## Let's check if multicollinearity is still present in the data.

# In[95]:


vif_series7 = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
)
print("VIF values: \n\n{}\n".format(vif_series7))


# ### Inference: In our data multicolinearity is still present. So we should drop the 'lread' column to get the accuracy.

# In[96]:


X_train = X_train.drop(["lread"], axis=1)


# In[97]:


vif_series8 = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
)
print("VIF values: \n\n{}\n".format(vif_series8))


# In[98]:


X_train = X_train.drop(["pflt"], axis=1)


# In[99]:


fitmod_42 = sm.OLS(y_train, X_train35)
fitres_42 = fitmod_42.fit()
print(fitres_42.summary())


# ### Inference : After dropping the features causing strong multicollinearity and the statistically insignificant ones, our model performance hasn't dropped sharply. This shows that these variables did not have much predictive power.

# In[100]:


vif_series9 = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
)
print("VIF values: \n\n{}\n".format(vif_series9))


# ### Inference : VIF for all features is <3

# In[ ]:





# In[101]:


df_pred = pd.DataFrame()

df_pred["Actual Values"] = y_train.values.flatten()  # actual values
df_pred["Fitted Values"] = fitres_42.fittedvalues.values  # predicted values
df_pred["Residuals"] = fitres_42.resid.values  # residuals

df_pred.head()


# In[ ]:





# ### TEST FOR LINEARITY AND INDEPENDENCE

# In[102]:


# let us plot the fitted values vs residuals
sns.set_style("whitegrid")
sns.residplot(
    data=df_pred, x="Fitted Values", y="Residuals", color="purple", lowess=True
)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Fitted vs Residual plot")
plt.show()


# #### Inference : We observe that the pattern has slightly decreased and the data points seems to be randomly distributed.

# In[103]:


# columns in training set
X_train35.columns


# In[104]:


# checking the distribution of variables in training set with dependent variable
plt.figure(figsize=(10,10))
sns.pairplot(df[['usr', 'lwrite', 'scall', 'swrite', 'exec', 'rchar', 'wchar', 'pgout',
       'atch', 'ppgin', 'pflt', 'freemem', 'freeswap']])
plt.show()


# In[105]:


sns.histplot(df_pred["Residuals"], kde=True)
plt.title("Normality of residuals")
plt.show()


# #### Inference : The residual terms are normally distributed.

# ####  The QQ plot of residuals can be used to visually check the normality assumption. The normal probability plot of residuals should approximately follow a straight line.

# In[106]:


import pylab
import scipy.stats as stats

stats.probplot(df_pred["Residuals"], dist="norm", plot=pylab)
plt.show()


# #### Inference : Partially the points are lying on the straight line in QQ plot

# In[107]:


stats.shapiro(df_pred["Residuals"])


# #### Inference : when p-value is  < 0.05, the residuals are rejected in shapiro test. but the tested value is greater than 0.05
# 

# ### TEST FOR HOMOSCEDASTICITY

# In[108]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip


# In[109]:


name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(df_pred["Residuals"], X_train8)
lzip(name, test)


# #### Inference : All the assumptions of linear regression are now satisfied. Let's check the summary of our final model (fitres_42).

# In[110]:


print(fitres_42.summary())


# ### Let's print the linear regression equation.

# In[111]:


# let's check the model parameters
fitres_42.params


# In[112]:


Equation = "usr ="
print(Equation, end=" ")
for i in range(len(X_train35.columns)):
    if i == 0:
        print(fitres_42.params[i], "+", end=" ")
    elif i != len(X_train35.columns) - 1:
        print(
            fitres_42.params[i],
            "* (",
            X_train35.columns[i],
            ")",
            "+",
            end="  ",
        )
    else:
        print(fitres_42.params[i], "* (", X_train35.columns[i], ")")


# ## We can now use the model for making predictions on the test data.

# In[113]:


X_train35.columns


# In[114]:


X_test.columns


# In[115]:


# dropping columns from the test data that are not there in the training data
X_test2 = X_test.drop(
    ["lread", "sread", "fork","ppgout","pgfree","pgin","vflt"], axis=1
)


# In[121]:


# let's make predictions on the test set
y_pred = fitres_42.predict(X_test2)


# In[118]:


# let's check the RMSE on the train data
rmse1 = np.sqrt(mean_squared_error(y_train, df_pred["Fitted Values"]))
rmse1


# In[119]:


# let's check the RMSE on the test data
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred))
rmse2


# In[120]:


# let's check the MAE on the train data
mae1 = mean_absolute_error(y_train, df_pred["Fitted Values"])
mae1


# In[ ]:


# let's check the MAE on the test data
mae2 = mean_absolute_error(y_test, y_pred)
mae2


# #### Inference : We can see that RMSE on the train and test sets are comparable. So, our model is not suffering from overfitting.
# ####                    MAE indicates that our current model is able to predict mpg within a mean error of 3.4 units on the test data.
# ####                    Hence, we can conclude the model "fitres_42" is good for prediction as well as inference purposes.
