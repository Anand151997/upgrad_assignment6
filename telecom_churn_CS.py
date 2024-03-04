#!/usr/bin/env python
# coding: utf-8

# In[1]:



#imporitng the required packages
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


# In[2]:


## REading the data
telecom_df= pd.read_csv('telecom_churn_data.csv')


# In[ ]:





# In[3]:


telecom_df.head()


# In[4]:


telecom_df.shape


# In[5]:


##Checking the column names
telecom_df.columns


# In[6]:


#Checking the datatypes
telecom_df.dtypes


# In[7]:


#Checking the Null values
telecom_df.isnull().sum()


# In[8]:


# Filter the DataFrame to show only columns with null values
null_columns = telecom_df.columns[telecom_df.isnull().any()]
telecom_df[null_columns][telecom_df[null_columns].isnull().any(axis=1)]


# In[9]:


null_values_df = telecom_df[null_columns][telecom_df[null_columns].isnull().any(axis=1)]
null_values_df.head(10)  # Display the first 10 rows with null values


# In[10]:


# To get the precentage of null values from all columns
round(100*(telecom_df.isnull().sum()/len(telecom_df.index)),2)


# In[11]:


telecom_df.info(verbose=1)
telecom_df.info()


# In[12]:



# Creating column name list by types of columns
id_cols = ['mobile_number', 'circle_id']
date_cols = ['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8', 'last_date_of_month_9', 'date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8', 'date_of_last_rech_9', 'date_of_last_rech_data_6', 'date_of_last_rech_data_7', 'date_of_last_rech_data_8', 'date_of_last_rech_data_9']
cat_cols =  ['night_pck_user_6', 'night_pck_user_7', 'night_pck_user_8', 'night_pck_user_9', 'fb_user_6', 'fb_user_7', 'fb_user_8', 'fb_user_9']
num_cols = [column for column in telecom_df.columns if column not in id_cols + date_cols + cat_cols]


# In[13]:


# The number of columns in each list
print("--> ID cols: %d\n--> Date cols:%d\n--> Numeric cols:%d\n--> Category cols:%d" % (len(id_cols), len(date_cols), len(num_cols), len(cat_cols)))


# # DATA CLEANING

# In[14]:




# Setting Display Options
pd.set_option('display.max_rows', None)

# Checking Missing Values Percentages
telecom_null_check = 100 * telecom_df.isnull().sum() / len(telecom_df)

# Creating a DataFrame to Display Results
telecom_null_df = pd.DataFrame(telecom_null_check)

# Renaming Columns
telecom_null_df.rename(columns={0: 'Null_Percentage'}, inplace=True)

# Sorting by Null Percentage
telecom_null_df_sorted = telecom_null_df.sort_values('Null_Percentage', ascending=False)

print(telecom_null_df_sorted)


# There are columns with null values more than 70 % so we can remove columns with more than 70% Null values

# In[15]:



##Showing Columns with more than 70% Null Value
Missing_Vals_Column_70=list(telecom_null_df_sorted.index[telecom_null_df_sorted['Null_Percentage'] > 70])
(Missing_Vals_Column_70)


# In[16]:


##Excluding Max_recharge, total_recharge data and avg recharge amount data for future analysis


# In[17]:


# Removing the columns as per above condition
Missing_Vals_Column_70=telecom_df.columns[round(100*telecom_df.isnull().sum()/len(telecom_df),2)> 70]
data_col=['max_rech_data_6','max_rech_data_7','max_rech_data_8','max_rech_data_9','total_rech_data_6','total_rech_data_7','total_rech_data_8','total_rech_data_9','av_rech_amt_data_6','av_rech_amt_data_7','av_rech_amt_data_8','av_rech_amt_data_9']
Missing_Vals_Column_70=[col for col in Missing_Vals_Column_70 if col not in data_col]
telecom_df=telecom_df.drop(Missing_Vals_Column_70,axis=1)
telecom_df.shape


# In[18]:


# Checking missing values percentages again
def NULL_CHECK(X):
    pd.set_option('display.max_rows', None)
    telecom_null_check_2 = 100*X.isnull().sum()/len(X)
    telecom_df = pd.DataFrame(telecom_null_check_2)
    telecom_df.rename(columns={0:'Null_Percentage'}, inplace=True)
    return telecom_df.sort_values('Null_Percentage', ascending=False)

NULL_CHECK(telecom_df)


# In[19]:


# # Convert mobile number to object so that it does not interfare with out visualizations
telecom_df['mobile_number']=telecom_df['mobile_number'].astype('object')
telecom_df['mobile_number'].head()


# In[20]:


NULL_CHECK(telecom_df)


# In[21]:


# impute 0 in recharge columns
zero_impute_rch = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 'total_rech_data_9',
        'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9',
        'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 'max_rech_data_9'
       ]


# In[22]:


# Now impute missing values with 0
telecom_df[zero_impute_rch] = telecom_df[zero_impute_rch].apply(lambda x: x.fillna(0))


# In[23]:


print(telecom_df[zero_impute_rch].isnull().sum()*100/telecom_df.shape[1])


# In[24]:


# Dropping the ID and Date columns as it not needed. 
Dropping=id_cols + date_cols
initial_cols = telecom_df.shape[1]
Cols_to_drop=[col for col in Dropping if col in telecom_df.columns]
telecom_df=telecom_df.drop(Cols_to_drop,axis=1)
telecom_df.shape


# In[25]:


# Imputing the remaining null columns as 0
telecom_df[telecom_df.select_dtypes(exclude=['datetime64[ns]', 'category']).columns] = telecom_df[telecom_df.select_dtypes(exclude=['datetime64[ns]', 'category']).columns].fillna(0, axis=1)


# In[26]:


NULL_CHECK(telecom_df)


# In[27]:


# Checking value_counts for loc_og_t2o_mou , std_og_t2o_mou , loc_ic_t2o_mou columns
print(telecom_df.loc_og_t2o_mou.value_counts(dropna= False))
print(telecom_df.std_og_t2o_mou.value_counts(dropna= False))
print(telecom_df.loc_ic_t2o_mou.value_counts(dropna= False))


# In[28]:


# Dropping above 3 columns as these have o and missing values
telecom_df.drop(['loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou'], axis= 1, inplace= True)


# In[29]:


telecom_df.shape


# In[30]:


## Counting the number unique values present in the columns.
def unique_nan(s):
    return s.nunique(dropna=False).sort_values(ascending=False)
unique_nan(telecom_df)


# In[31]:


# Dropping columns with only 1 unique value
cols = []
for i in telecom_df.columns:
    if telecom_df[i].nunique() ==1:
        cols.append(i)
        
cols

telecom_df = telecom_df.drop(cols,1)
telecom_df.shape


# In[32]:


NULL_CHECK(telecom_df)


# Finally there are no missing values in the data

# # DATA PREPARATION

# Creating column avg_recharge_6_7 by adding total recharge amount of 6 & 7 month, then take avg of sum.

# In[33]:


# Get the index of null vals for both columns and verify if both matches 
# if the result is false it means all rows of total_rech_data and av_rech_amt_data has null at same rows.
res = telecom_df.total_rech_data_6[telecom_df.total_rech_data_6.isna()].index != telecom_df.av_rech_amt_data_6[telecom_df.av_rech_amt_data_6.isna()].index
print('June :', res.any())
res = telecom_df.total_rech_data_7[telecom_df.total_rech_data_7.isna()].index != telecom_df.av_rech_amt_data_7[telecom_df.av_rech_amt_data_7.isna()].index
print('July :', res.any())
res =telecom_df.total_rech_data_8[telecom_df.total_rech_data_8.isna()].index != telecom_df.av_rech_amt_data_8[telecom_df.av_rech_amt_data_8.isna()].index
print('Aug :', res.any())


# In[34]:


telecom_df['avg_rech_6_7']=(telecom_df['total_rech_amt_6']+telecom_df['total_rech_amt_7'])/2


# Define high value customers as follows

# Those who have recharged 70% or above in the first two months

# In[35]:


#Finding 70th percentile for the new column
X=telecom_df['avg_rech_6_7'].quantile(0.7)
X


# In[36]:


# filtering only the customers who have recharged more than X i.e are HIGH-VALUE Customers.
telecom_df = telecom_df[telecom_df['avg_rech_6_7'] >= X]
telecom_df.head()


# Tagging the Churners
# Those who have not made any calls and have not used any mobile internet even once in the churn phase.  (Churn=1 and else=0)

# In[37]:


# counting the rows having more than 50% missing values.
Missing_rows=telecom_df[(telecom_df.isnull().sum(axis=1)) > (len(telecom_df.columns)//2)]
Missing_rows


# In[38]:


telecom_df['Churn']= np.where((telecom_df['total_ic_mou_9']==0) & (telecom_df['total_og_mou_9']==0) & (telecom_df['vol_2g_mb_9']==0) & (telecom_df['vol_3g_mb_9']==0), 1, 0)


# In[39]:


telecom_df.head()


# In[40]:


telecom_df['Churn'].value_counts()


# After tagging churners, let us remove all the attributes corresponding to the churn phase (all attributes having ‘ _9’, etc. in their names)

# In[41]:


columns_9 = [col for col in telecom_df.columns.to_list() if '_9' in col]
print(columns_9)


# In[42]:


# Deleting the churn month columns
telecom_df = telecom_df.drop(columns_9, axis=1)


# In[43]:


# Dropping sep_vbc_3g column
telecom_df = telecom_df.drop('sep_vbc_3g', axis=1)


# In[44]:


# Checking churn data
plt.figure(figsize= [7,5])
sns.countplot(telecom_df.Churn, palette= 'Paired', label=[1,0])
plt.show()
# Checking for the CHURN rate
round(100*(telecom_df['Churn'].mean()),2)


# This tells us that 8.64% of customers have already churned

# # Outlier Treatment

# In[45]:


# Converting the Churn column to Object data type in order to proceed with Outlier Treatment.

telecom_df['Churn'] = telecom_df['Churn'].astype(object)


# In[46]:


# Listing numeric columns only
num_cols = telecom_df.select_dtypes(exclude=['object']).columns
print(num_cols)


# In[47]:


telecom_df.describe(percentiles=[0.01, 0.10,.25,.5,.75,.90,.95,.99])


# In[48]:


# Removing outliers below 10th and above 90th percentile
for col in num_cols: 
    q1 = telecom_df[col].quantile(0.10)
    q3 = telecom_df[col].quantile(0.90)
    iqr = q3-q1
    range_low  = q1-1.5*iqr
    range_high = q3+1.5*iqr
    # Assigning the filtered dataset into new_DF
    new_telecom_df = telecom_df.loc[(telecom_df[col] > range_low) & (telecom_df[col] < range_high)]

new_telecom_df.shape


# # Adding Some New Columns with some insights

# In[49]:


# Avg recharge number at action phase
# We are taking average because there are two months(7 and 8) in action phase
new_telecom_df['avg_rech_num_action'] = (new_telecom_df['total_rech_num_7'] + new_telecom_df['total_rech_num_8'])/2
# Difference between total_rech_num_6 and avg_rech_action
new_telecom_df['diff_rech_num'] = new_telecom_df['avg_rech_num_action'] - new_telecom_df['total_rech_num_6']


# In[50]:


# Checking if recharge number has decreased in action phase, 1=Yes, 0=No
new_telecom_df['dec_rech_action'] = np.where((new_telecom_df['diff_rech_num'] < 0), 1, 0)


# In[51]:


new_telecom_df.head()


# Adding dec_rech_amt_action
# 
# Indicating if recharge amount of customers when compared to good phase is decreased in action phase or not
# 
# 

# In[52]:


# Avg recharge amount in action phase
# We are taking average because there are two months(7 and 8) in action phase
new_telecom_df['avg_rech_amt_action'] = (new_telecom_df['total_rech_amt_7'] + new_telecom_df['total_rech_amt_8'])/2
# Difference of action phase recharge amount and good phase recharge amount
new_telecom_df['diff_rech_amt'] = new_telecom_df['avg_rech_amt_action'] - new_telecom_df['total_rech_amt_6']


# In[53]:


# Checking if recharge amount has decreased in action phase, 1=Yes, 0=No
new_telecom_df['dec_rech_amt_action'] = np.where((new_telecom_df['diff_rech_amt'] < 0), 1, 0) 


# In[54]:


# average ARUP in action phase
# We are taking average because there are two months(7 and 8) in action phase
new_telecom_df['avg_arpu_action'] = (new_telecom_df['arpu_7'] + new_telecom_df['arpu_8'])/2
# Difference of good and action phase ARPU
new_telecom_df['diff_arpu'] = new_telecom_df['avg_arpu_action'] - new_telecom_df['arpu_6']


# In[55]:


# Checking whether the arpu has decreased on the action month, 1=Yes, 0=No
new_telecom_df['dec_avg_revenuePC_action'] = np.where((new_telecom_df['diff_arpu'] < 0), 1, 0)


# In[56]:


new_telecom_df.head()


# In[57]:


# Total mou at good phase incoming and outgoing
new_telecom_df['total_mou_good'] = (new_telecom_df['total_og_mou_6'] + new_telecom_df['total_ic_mou_6'])
# Avg. mou at action phase
# We are taking average because there are two months(7 and 8) in action phase
new_telecom_df['avg_mou_action'] = (new_telecom_df['total_og_mou_7'] + new_telecom_df['total_og_mou_8'] + new_telecom_df['total_ic_mou_7'] + new_telecom_df['total_ic_mou_8'])/2

# Difference avg_mou_good and avg_mou_action
new_telecom_df['diff_mou'] = new_telecom_df['avg_mou_action'] - new_telecom_df['total_mou_good']


# In[58]:


# Checking whether the mou has decreased in action phase, 1=Yes, 0=No
new_telecom_df['dec_MOU_action'] = np.where((new_telecom_df['diff_mou'] < 0), 1, 0)
new_telecom_df.head()


# # Exploratory Data Anlaysis

# In[ ]:





# In[ ]:





# In[59]:


# Converting churn column to int in order to do aggfunc in the pivot table
new_telecom_df['Churn'] = new_telecom_df['Churn'].astype('int64')


# In[60]:


new_telecom_df.pivot_table(values='Churn', index='dec_MOU_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# The churn is more for customers whose minutes of Usase (MOU)  decreased in the action phase than the good phase

# Churn rate on the basis whether the customer decreased her/his amount of recharge in action month
# 
# 

# In[61]:


new_telecom_df.pivot_table(values='Churn', index='dec_rech_amt_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# Here also dthe churn rate is high for customer whose amount of recharge is lesser than the amount in the good phase

# In[62]:


#Churn rate on the basis whether the customer decreased her/his number of recharge in action month
new_telecom_df.pivot_table(values='Churn', index='dec_rech_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# The number of recharge is lesser in action phase than in good phase

# Analysis of the minutes of usage MOU (churn and not churn) in the action phase

# In[63]:


# Creating churn dataframe
Data_churn = new_telecom_df[new_telecom_df['Churn'] == 1]
# Creating not churn dataframe
Data_Non_churn = new_telecom_df[new_telecom_df['Churn'] == 0]


#  #Distribution plot
# ax = sns.distplot(Data_churn['total_mou_good'],label='churn',hist=False)
# ax = sns.distplot(Data_Non_churn['total_mou_good'],label='non churn',hist=False)
# plt.legend(loc='best')
# ax.set(xlabel='Action phase MOU')

# Minutes of usage is higher in churn population which means higher the MOU lesser than churn probability

# In[64]:


#Analysing recharge amount and number of recharge in action month

plt.figure(figsize=(10,6))
fig = sns.scatterplot('avg_rech_num_action','avg_rech_amt_action', hue='Churn', data=new_telecom_df)


# In[65]:


new_telecom_df.pivot_table(values='Churn', index='dec_rech_amt_action', columns='dec_rech_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# Decrease in recharge amount higher in the action phase 

# In[66]:


# Function to create Box Plot for month 6,7 and 8
def plot_box_chart(attribute):
    plt.figure(figsize=(20,16))
    df = new_telecom_df
    plt.subplot(2,3,1)
    sns.boxplot(data=df, y=attribute+"_6",x="Churn",hue="Churn",
                showfliers=False,palette=("plasma"))
    plt.subplot(2,3,2)
    sns.boxplot(data=df, y=attribute+"_7",x="Churn",hue="Churn",
                showfliers=False,palette=("plasma"))
    plt.subplot(2,3,3)
    sns.boxplot(data=df, y=attribute+"_8",x="Churn",hue="Churn",
                showfliers=False,palette=("plasma"))
    plt.show()


# In[67]:


recharge_amnt_columns =  new_telecom_df.columns[new_telecom_df.columns.str.contains('rech_amt')]
recharge_amnt_columns.tolist()


# In[68]:


plot_box_chart('total_rech_amt')
plot_box_chart('max_rech_amt')
plot_box_chart('av_rech_amt_data')


#     From the above plots we can see clearly that the reacharge amounts (Total & Maximum) started to fall in the month 8 i.e near to the churn phase.

# In[69]:


# Dropping the Some of the Derived columns that are not needed furthur.

new_telecom_df = new_telecom_df.drop(['total_mou_good','avg_mou_action','diff_mou','avg_rech_num_action','diff_rech_num','avg_rech_amt_action',
                 'diff_rech_amt','avg_arpu_action','diff_arpu','avg_rech_6_7'], axis=1)


# In[70]:


#function for box plot
def bx_plot(*args,data): 
    
    m=math.ceil(len(args)/2)  # getting the length f arguments to determine the shape of subplots                   
    fig,axes = plt.subplots(m,2,squeeze=False, figsize = (16, 8*m))
    ax_li = axes.flatten()       # flattening the numpy array returned by subplots
    i=0
    for col in args:
        
        sns.boxplot(col, data, ax=ax_li[i])  # plotting the box plot
        ax_li[i].set_title(col)
        #ax_li[i].set_xscale('log')
        plt.tight_layout()
        i=i+1


# In[71]:


# plotting the distribution for recharge amount columns
col_rech = [col for col in new_telecom_df.columns if 'rech' in col]
col_rech


# In[72]:


fig,axes = plt.subplots(1,1,squeeze=False, figsize = (20, 10))
ax=axes[0][0]

new_telecom_df.pivot(columns='Churn')[col_rech].plot(kind = 'box',ax=ax)

ax.xaxis.set_tick_params(rotation=90)
plt.yscale('log')
 


# # TRAIN TEST SPLIT

# In[73]:


from sklearn.model_selection import train_test_split

# Putting feature variables into X
X = new_telecom_df.drop(['Churn'], axis=1)

# Putting target variable to y
y = new_telecom_df.pop('Churn')

# Splitting data into train and test set 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100, stratify=y)


# In[74]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[75]:


print(X_train.columns)


# In[76]:


# Check if '6/27/2014' exists in any column of the DataFrame
mask = X_train.isin(['6/27/2014'])

# Get the rows where '6/27/2014' exists in any column
rows_with_value = X_train[mask.any(axis=1)]

# Print the rows with the value '6/27/2014'
print(rows_with_value)


# In[77]:





# In[ ]:


#Dealing with Class Imbalance using SMOTE (Synthetic Minority Oversampling Technique)

#We are creating synthetic samples by doing upsampling using SMOTE

# Imporing SMOTE

from imblearn.over_sampling import SMOTE

# Instantiating SMOTE
smt = SMOTE(random_state=42)

# Fittign SMOTE to the train set
X_train, y_train = smt.fit_resample(X_train, y_train)


# In[ ]:


get_ipython().system('pip install imbalanced-learn')


# In[80]:


X_train.shape


# # Scaling numeric features

#  During EDA we have observed few outliers in numeric features. So, using Robust Scaling using median and quantile values instead of Standard Scaling using mean and standard deviation.

# In[81]:


# Standardization method
from sklearn.preprocessing import StandardScaler

# Instantiate the Scaler
scaler = StandardScaler()


# In[82]:


X_train.head()


# In[83]:


# List of the numeric columns
cols_scale = X_train.columns.to_list()
print("Total scalable columns: ", len(cols_scale))
# Removing the derived binary columns 
binary_cols_to_remove=['dec_MOU_action','dec_rech_action','dec_rech_amt_action','dec_avg_revenuePC_action']
for col in binary_cols_to_remove:
    cols_scale.remove(col)

print("Scalable cols after removing : ", len(cols_scale))


# In[84]:


# Fit the data into scaler and transform
X_train[cols_scale] = scaler.fit_transform(X_train[cols_scale])
X_train.head()


# In[85]:


# Transform the test set
X_test[cols_scale] = scaler.transform(X_test[cols_scale])
X_test.head()


# # Model building with PCA(Principal Component Analysis)

# In[86]:


#Import PCA
from sklearn.decomposition import PCA

# Instantiate PCA
pca = PCA(svd_solver='randomized', random_state=100)

# Fit train set on PCA
pca.fit(X_train)


# In[87]:


# Looking at the Principal components
pca.components_


# In[88]:


# Cumuliative varinace of the principal components.
variance_cumu = np.round(np.cumsum(pca.explained_variance_ratio_)*100, decimals=4)
print(variance_cumu)


# In[89]:


# # Plotting scree plot
fig = plt.figure(figsize=[12,8])
plt.axhline(y=95, color='g', linestyle='-.')
plt.axvline(x=70, color='r', linestyle='-.')
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),variance_cumu)
plt.xlabel('Number of Components')
plt.ylabel("Cumulative variance explained")
plt.show()


# This shows that 70% of components are enough to make predictions

# Using incremental PCA for better efficiency

# In[90]:


from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components = 70)

X_train_pca = pca_final.fit_transform(X_train)


# In[91]:


print("Size of earlier dataset was :",X_train.shape)
print("Size of dataset after PCA is:", X_train_pca.shape)


# In[92]:


#creating correlation matrix for the given data
corrmat = np.corrcoef(X_train_pca.transpose())

#Make a diagonal matrix with diagonal entry of Matrix corrmat
p = np.diagflat(corrmat.diagonal())

# subtract diagonal entries making all diagonals 0
corrmat_diag_zero = corrmat - p
print("max positive corr:",round(corrmat_diag_zero.max(),3), ", min negative corr: ", round(corrmat_diag_zero.min(),3))


# In[93]:


from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components = 70)

X_train_pca = pca_final.fit_transform(X_train)
print("Size of earlier dataset was :",X_train.shape)
print("Size of dataset after PCA is:", X_train_pca.shape)


# In[94]:


#creating correlation matrix for the given data
corrmat = np.corrcoef(X_train_pca.transpose())

#Make a diagonal matrix with diagonal entry of Matrix corrmat
p = np.diagflat(corrmat.diagonal())

# subtract diagonal entries making all diagonals 0
corrmat_diag_zero = corrmat - p
print("max positive corr:",round(corrmat_diag_zero.max(),3), ", min negative corr: ", round(corrmat_diag_zero.min(),3))


# In[95]:


X_test_pca = pca_final.transform(X_test)
X_test_pca.shape


# # MODEL BUILDING

# 1. Logistic regression with PCA

# In[96]:


# Importing scikit logistic regression module
from sklearn.linear_model import LogisticRegression

# Impoting metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# default solver is very slow so changed to 'lbfgs'
logreg = LogisticRegression(solver = 'lbfgs',class_weight="balanced")
# Training the model on the data
logreg.fit(X_train_pca, y_train)


# In[97]:


#prediction on test data
y_pred = logreg.predict(X_test_pca)

#create confusion matrix
cm = confusion_matrix(y_test,y_pred)
print("confusoin matrix \t\n",cm)

#checking sesitivity 
print("sensitivity \t", (cm[1,1]/(cm[1,0]+cm[1,1])).round(2))


#checking  specificity
print("specificity \t", (cm[0,0]/(cm[0,0]+cm[0,1])).round(2))

#check area under the curve
from sklearn.metrics import roc_auc_score
print("area under the curve \t",round(roc_auc_score(y_test,y_pred),2))


# In[98]:


# Importing libraries for cross validation
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[99]:


# Creating StratifiedKFold object with 5 splits
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

# GridSearch
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True) 

# Fit the model
model_cv.fit(X_train_pca, y_train)


# In[100]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[101]:


# plotting C versus train and validation scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('sensitivity')
plt.legend(['test result', 'train result'], loc='best')
plt.xscale('log')


# In[102]:


# Best score with best C
best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']

print(" The highest test sensitivity is {0} at C = {1}".format(best_score, best_C))


# # MODEL WITH OPTIMUM Paramaters

# In[103]:


# Instantiate the model with best C
logistic_pca = LogisticRegression(C=best_C)

# Fit the model on the train set
log_pca_model = logistic_pca.fit(X_train_pca, y_train)


# Prediction on the train set

# In[105]:


# Predictions on the train set
y_train_pred = log_pca_model.predict(X_train_pca)
## Confusion Matrix.

actual = np.random.binomial(1,.9,size = 1000)
predicted = np.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_train, y_train_pred)
confusion_matrix

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()  


# In[106]:


TP = confusion_matrix[1,1] # true positive 
TN = confusion_matrix[0,0] # true negatives
FP = confusion_matrix[0,1] # false positives
FN = confusion_matrix[1,0] # false negatives
# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# Recall
print("Recall:-" , TP/float(TP+FN))

# check area under curve
y_pred_prob = log_pca_model.predict_proba(X_train_pca)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_train, y_pred_prob),2))


# In[107]:


# Prediction on the test set
y_test_pred = log_pca_model.predict(X_test_pca)


# In[108]:


# Confusion matrix
actual = np.random.binomial(1,.9,size = 1000)
predicted = np.random.binomial(1,.9,size = 1000)

confusion = metrics.confusion_matrix(y_test, y_test_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()


# In[109]:


# Accuracy
print("Accuracy:-    \t",round(metrics.accuracy_score(y_test, y_test_pred),2))

# Sensitivity
print("Sensitivity:-  \t",round(TP / float(TP+FN),2))

# Specificity
print("Specificity:-  \t", round(TN / float(TN+FP),2))

# Recall
print("Recall:-    \t" , round(TP/float(TP+FN),2))

# check area under curve
y_pred_prob = log_pca_model.predict_proba(X_test_pca)[:, 1]
print("AUC:-    \t", round(roc_auc_score(y_test, y_pred_prob),2))


# # 2. Decision tree with PCA

# In[110]:


# Importing decision tree classifier
from sklearn.tree import DecisionTreeClassifier
##Hyperparameter tuning

# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model for best results.
dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, 
                           param_grid = param_grid, 
                           scoring= 'recall',
                           cv = 5, 
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_pca,y_train)


# In[111]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[113]:


# Printing the optimal sensitivity score and hyperparameters
print("Best sensitivity:-", grid_search.best_score_)
print(grid_search.best_estimator_)


# In[114]:


##Model with optimal hyperparameters

# Model with optimal hyperparameters
dt_pca_model = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=10, 
                                  min_samples_leaf=50,
                                  min_samples_split=10)

dt_pca_model.fit(X_train_pca, y_train)


# In[115]:


# Predictions on the train set
y_train_pred = dt_pca_model.predict(X_train_pca)


# Creating Confusion matrix

confusion=metrics.confusion_matrix(y_train, y_train_pred)
confusion


# In[116]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[117]:


# Accuracy
print("Accuracy:-",round(metrics.accuracy_score(y_train, y_train_pred),2))

# Sensitivity
print("Sensitivity:-",round(TP / float(TP+FN),2))

# Specificity
print("Specificity:-", round(TN / float(TN+FP),2))

# Recall
print("Recall:-" , round(TP/float(TP+FN),2))

# AUC
print("Area under curve is:", round(metrics.roc_auc_score(y_train, y_train_pred),2))


# In[118]:


# Prediction on the test set
y_test_pred = dt_pca_model.predict(X_test_pca)


# In[119]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[120]:


# Accuracy
print("Accuracy:-",round(metrics.accuracy_score(y_test, y_test_pred),2))

# Sensitivity
print("Sensitivity:-",round(TP / float(TP+FN),2))

# Specificity
print("Specificity:-", round(TN / float(TN+FP),2))

# Recall
print("Recall:-" , round(TP/float(TP+FN),2))

# AUC
print("Area under curve is:", round(metrics.roc_auc_score(y_test, y_test_pred),2))


# # RANDOM FOREST MODELLING

# In[121]:


# Importing random forest classifier
from sklearn.ensemble import RandomForestClassifier


# In[122]:


# Running the random forest with default parameters.
rfc = RandomForestClassifier()


# In[123]:


# fit
rfc.fit(X_train_pca,y_train)


# In[124]:


# Making predictions
predictions = rfc.predict(X_test_pca)


# In[127]:


# Checking the report of our default model
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))


# In[128]:


# Printing confusion matrix
confusion=metrics.confusion_matrix(y_test,predictions)
confusion


# In[130]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


# In[131]:


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(2, 20, 5)}

# instantiate the model
rf = RandomForestClassifier(class_weight= 'balanced', random_state=100)


# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, 
                   scoring="accuracy",verbose =1, n_jobs = -1, return_train_score=True)
rf.fit(X_train_pca, y_train)

print(rf.best_score_)
print(rf.best_params_)


# In[132]:


# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()


# In[133]:


# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


#  We see that as we increase the value of max_depth, both train and test scores increase till a point. The ensemble tries to overfit as we increase the max_depth.
# 
# Thus, controlling the depth of the constituent trees will help reduce overfitting in the forest.

# In[134]:


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'n_estimators': range(100, 1500, 500)}

# instantiate the model (note we are specifying a max_depth)
rf = RandomForestClassifier(max_depth=4, class_weight ='balanced', random_state=100)


# fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, verbose = 1,
                   scoring="accuracy",return_train_score=True, n_jobs = -1)
rf.fit(X_train_pca, y_train)

print(rf.best_score_)
print(rf.best_params_)


# In[135]:


# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()


# In[136]:


# plotting accuracies with n_estimators
plt.figure()
plt.plot(scores["param_n_estimators"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_n_estimators"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[137]:


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_features': [4, 8, 14, 20, 24]}

# instantiate the model
rf = RandomForestClassifier(max_depth=4,class_weight='balanced',random_state=100)


# fit tree on training data
rf = GridSearchCV(rf, parameters, verbose = 1,cv=n_folds, 
                   scoring="accuracy",n_jobs = -1, return_train_score = True)
rf.fit(X_train_pca, y_train)

print(rf.best_score_)
print(rf.best_params_)


# In[138]:


scores = rf.cv_results_
pd.DataFrame(scores).head()


# In[140]:


# plotting accuracies with max_features
plt.figure()
plt.plot(scores["param_max_features"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_features"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_features")
plt.ylabel("Accuracy")
plt.legend()
plt.show()# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(100, 400, 50)}

# instantiate the model
rf = RandomForestClassifier(class_weight ='balanced',random_state=100)


# fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",verbose =1, return_train_score = True, n_jobs = -1)
rf.fit(X_train_pca, y_train)
print(rf.best_score_)
print(rf.best_params_)


# In[141]:


# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()


# In[142]:


# plotting accuracies with min_samples_leaf
plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[143]:


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_split': range(200, 500, 50)}

# instantiate the model
rf = RandomForestClassifier(class_weight = 'balanced',random_state=100)


# fit tree on training data
rf = GridSearchCV(rf, parameters, verbose = 1,
                    cv=n_folds, 
                   scoring="accuracy",n_jobs =-1, return_train_score = True)
rf.fit(X_train_pca, y_train)

print(rf.best_score_)
print(rf.best_params_)


# In[144]:


# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()


# In[145]:


# plotting accuracies with min_samples_split
plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[146]:


# Create the parameter grid based on the results of random search  
param_grid = {
    'max_depth': [4,8,10],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100,200, 300], 
    'max_features': [5, 10]}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1)


# In[ ]:


# Fitting the grid search to the data

grid_search.fit(X_train_pca, y_train)


# In[148]:


# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,class_weight = 'balanced',
                             max_depth=grid_search.best_params_['max_depth'],
                             min_samples_split=grid_search.best_params_['min_samples_split'],
                             max_features=grid_search.best_params_['max_features'],
                             n_estimators =grid_search.best_params_['n_estimators'],
                             random_state=100, oob_score=True)


# In[ ]:


# fitting the Model
rfc.fit(X_train_pca,y_train)


# In[ ]:


# predicting using test data

predictions = rfc.predict(X_test_pca)


# In[ ]:


# predicting using test data

predictions = rfc.predict(X_test_pca)


# In[ ]:


rfc.oob_score_


# In[ ]:


#create confusion matrix
cm = metrics.confusion_matrix(y_test,predictions)
print("confusoin matrix \t\n",cm)
# Accuracy
print("Accuracy:-",round(metrics.accuracy_score(y_test, predictions),2))

#checking sesitivity 
print("sensitivity \t", (cm[1,1]/(cm[1,0]+cm[1,1])).round(2))


#checking  specificity
print("specificity \t", (cm[0,0]/(cm[0,0]+cm[0,1])).round(2))

# check area under curve
y_pred_prob = rfc.predict_proba(X_test_pca)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2


# # Conclusion and STRATEGY ahead

# 1) From the exploratory data analysis we have seen that there is considerable decrease in call and data usage, recharge during the action phase which is the 8th month.
# 2)  The following are identified as important features loc_og_t2m_mou_7
# total_og_mou_6
# loc_og_t_7
# roam_ic_mou_7
# onnet_mou_7
# loc_og_t2c_mou_7
# onnet_mou_8
# roam_og_mou_8
# arpu_7
# 3) A sudden drop in average revenue  per user in the 7th month plays a vital role in deciding the churn.
# 4) Outgoing ,roaming and total minutes of outgoing are all important factors  affecting the churn.
#   
#     
#     The following startegies can be used to avoid churning
#     1)  Imporve the customer experience by giving proper network connection to the customers who you see decrease in usage of outgoing in the 7th month.
#     2)  Directly call the customer and ask for the feedback , if they still decide to leave try to ask for a chance and provide some offers. 
#     3)  If they are leaving because of specific recharge plan provided by the another network provider, provide them a customised plan according to their convenience.

# In[ ]:




