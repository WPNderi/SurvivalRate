#%% [markdown]
# ## Task 1: Data Retrieving

#%%
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%%
#Reading the data
data1 = pd.read_csv('haberman.data.csv', sep=',', header=None, names=['age', 'year', 'nodes', 'status'])


#%%
#Viewing the data
data1.head()


#%%
#Viewing the data
data1


#%%
#Understanding the rows and columns making up the data
data1.shape


#%%
#Check for missing values
data1.isnull().values.any()
data1.isnull().sum()


#%%
#Data types of the features
data1.dtypes


#%%
# Age attribute
data1['age'].value_counts()


#%%
#There does not seem to be any impossible values in this attribute


#%%
data1['age'].plot(kind='hist',bins=20)
plt.title('Age info')
plt.xlabel('Age')
plt.show()


#%%
data1['age'].plot(kind='box')
plt.title('Age info Box-plot')
plt.ylabel('Age')
plt.show()


#%%
#The distribution looks fairly symmetric


#%%
#Year attribute
data1['year'].value_counts()


#%%
#This study contains observations between the year 1958 and 1970. 


#%%
data1['year'].value_counts().plot(kind='bar')
plt.title('Observations per year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()


#%%
#Nodes attribute
data1['nodes'].value_counts()


#%%
# We do not observe any impossible values.
# It seems that in general it is not common to detect a large number of positive axillary nodes.


#%%
data1['nodes'].plot(kind='hist',bins=10)
plt.title('Number of Positive Axillary nodes detected')
plt.xlabel('No. of nodes')
plt.show()


#%%
# This further confirms our earlier observations and we see a left skew in the data


#%%
#Status - Target attribute
data1['status'].value_counts()


#%%
data1['status'].value_counts().plot(kind='bar')
plt.title('Survival status of patient')
plt.xlabel('Survives longer than 5 years vs Does not survive longer than 5 years')
plt.ylabel('Count')
plt.show()


#%%
def get_bounds(val,range_years,upper=True):
    # get remainder
    rem = val % range_years
    bounds = 0
    if rem != 0:
        bounds = range_years - rem
    
    return val + bounds

def get_ranges(data,range_years):
    # Get the lower and upper bounds of the data
    lower, upper = min(data), max(data)
    
    # Split the data by the years
    lower_bounds = get_bounds(lower,range_years,upper=False)
    upper_bounds = get_bounds(upper,range_years)
    
    # Create the ranges dict
    ranges = []
    for i in range((lower_bounds + range_years),(upper_bounds + range_years),range_years):
        d = {
            'upper_age': i,
            'lower_age': i - range_years,
            'count': 0
        }
        ranges.append(d)
        
    return ranges

def age_count(data,range_years=5):
    # Get the ranges array
    l = get_ranges(data['age'],range_years)
    # age group array
    group_array = []
    # Categrorize the ages based on the age groups
    for a in l:
        for d in data['age']:
            if d >= a['lower_age'] and d < a['upper_age']:
                # Increment the count of the age group by 1
                a['count'] += 1
                # Add the age to the age group array
                group_array.append('{} - {}'.format(a['lower_age'],a['upper_age']))
    # Add the age group array as column to the dataset
    data['age_group'] = group_array
    return data
        
print(age_count(data1,range_years=5))
        


#%%
data1['age_group'].value_counts().sort_index()

#%% [markdown]
# ## Task 2: Data Exploration

#%%
#Do number of nodes vary with age?


#%%
data1.boxplot(column='nodes',by='age_group',figsize=(14,6),grid=False)
plt.ylabel('Nodes per year')
plt.show()


#%%
# Does age affect survival status?


#%%
bar_data = data1

grouped = bar_data.groupby(['status','age_group'])
grouped.size()


#%%
status_table =  pd.crosstab(index=bar_data['status'],columns=bar_data['age_group'])
status_table


#%%
status_table.plot(kind='bar',figsize=(14,6))
plt.title('Survival status of patient by age-group')


#%%
ax = sns.violinplot(x="status", y="nodes",data=data1)

# Calculate number of obs per group & median to position labels
medians = data1.groupby(['status'])['nodes'].median().values
nobs = data1['status'].value_counts().values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]
 
    
 # Add it to the plot
pos = range(len(nobs))
for tick,label in zip(pos,ax.get_xticklabels()):
   ax.text(pos[tick], medians[tick] + 0.03, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')

plt.title('Survival status by no. of nodes detected')


#%%
#import seaborn  as se

sns.set_style("darkgrid");
sns.FacetGrid(data1,col="status",hue='status',size=6,col_wrap=2,)    .map(plt.scatter,"nodes","age")    .add_legend()

#plt.show()
#plt.title('YOUR TITLE HERE')
plt.subplots_adjust(top=0.9)
plt.suptitle('Survival status by no. of nodes detected and age of patient')

#%% [markdown]
# ## Task 3: Data Modelling
#%% [markdown]
# ### Model 1: k Nearest Neighbor

#%%
# We shall use age, nodes and the year and use a backward elimination technique to 
#find the feature that gives the best score
X = data1.iloc[:, [0,1,2]]
y = data1.iloc[:, 3]



#%%
# Fitting classifier
#We start with arbitrary parameters
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,weights='distance',metric='minkowski',p=2)


#When p=1 equal to Manhattan distance
#When p= 2 equal to Euclidean distance
#Even with p change, values remain the same with a distance metric


#%%
#Validating the model


#%%
# We shall use a k folds cross validation technique


#%%
#Necessary imports
from sklearn.model_selection import cross_val_score,cross_val_predict


#%%
# Perform 10 fold cross validation
scores = cross_val_score(classifier, X,y, cv=10)
print 'Cross-validated scores:', scores


#%%
# Print the mean score and variance
print('Accuracy: %0.2f (+/- %0.2f)'% (scores.mean(), scores.std()*2))


#%%
# Confusion matrix with k=5, p=2
from sklearn.metrics import classification_report, confusion_matrix
predicted = cross_val_predict(classifier, X, y, cv=10)

print classification_report(y,predicted)

#%% [markdown]
# ### Parameter Tuning
#%% [markdown]
# ### Optimum value for k 

#%%
# We can find the optimal value of k for KNN


#%%
# range of k we want to try
k_ranges = range(1, 51)
# empty list to store scores
k_scores = []

# 1. we will loop through reasonable values of k
for k in k_ranges:
    # 2. run KNeighborsClassifier with k neighbours
    knnClass = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knnClass, X, y, cv=10, scoring='accuracy')
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())


print(k_scores)


#%%
print('Length of list', len(k_scores))
print('Max of list', max(k_scores))


#%%
# plot how accuracy changes as we vary k

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(x_axis, y_axis)
plt.figure(figsize=[13,8])
plt.plot(k_ranges, k_scores)
plt.title('Number of neighbors(value of K) VS Accuracy')
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')

Value of k seems to be at about 32
#%%
# Now use this k value in the classifier
classifier_remod = KNeighborsClassifier(n_neighbors=32,weights='distance',metric='minkowski',p=2)


#%%
# Using remod
scoresmod = cross_val_score(classifier_remod, X,y, cv=10)
print 'Cross-validated scores:', scoresmod


#%%
# Print the mean score and variance
print('Accuracy: %0.2f (+/- %0.2f)'% (scoresmod.mean(), scoresmod.std()*2))


#%%
# Confusion matrix with k=32, p=2
predicted_remod = cross_val_predict(classifier_remod, X, y, cv=10)

print classification_report(y,predicted_remod)

#%% [markdown]
# ### Optimum value of p

#%%
# Now find best value of p


#%%
# We started with a p value of 2 which is the Euclidean distance
# We shall now try with a p value of 1 which is the Manhattan or taxi distance


#%%
classifier_remod2 = KNeighborsClassifier(n_neighbors=35,weights='distance',p=1)


#%%
# Using remod
scoresmod2 = cross_val_score(classifier_remod2, X,y, cv=10)
print 'Cross-validated scores:', scores


#%%
# Print the mean score and variance
print('Accuracy: %0.2f (+/- %0.2f)'% (scoresmod2.mean(), scoresmod2.std()*2))


#%%
# The Euclidean distance(p=2) seems to be a better measure


#%%
# Confusion matrix with p=1
predicted_remod2 = cross_val_predict(classifier_remod2, X, y, cv=10)

print classification_report(y,predicted_remod2)

#%% [markdown]
# ### Finding the feature with the highest accuracy score
# 

#%%
# We started with all three descriptive features-age, year and nodes
# We will start with age and year and compare the scores
X = data1.iloc[:, [0,1]]
y = data1.iloc[:, 3]


#%%
classifier_remod3 = KNeighborsClassifier(n_neighbors=32,weights='distance',metric='minkowski',p=2)


#%%
# Using remod
scoresmod3 = cross_val_score(classifier_remod3, X,y, cv=10)
print 'Cross-validated scores:', scores


#%%
# Print the mean score and variance
print('Accuracy: %0.2f (+/- %0.2f)'% (scoresmod3.mean(), scoresmod3.std()*2))


#%%
# We will continue with age and nodes and compare the scores


#%%
X = data1.iloc[:, [0,2]]
y = data1.iloc[:, 3]


#%%
classifier_remod4 = KNeighborsClassifier(n_neighbors=32,weights='distance',metric='minkowski',p=2)


#%%
# Using remod
scoresmod4 = cross_val_score(classifier_remod4, X,y, cv=10)
print 'Cross-validated scores:', scores


#%%
# Print the mean score and variance
print('Accuracy: %0.2f (+/- %0.2f)'% (scoresmod4.mean(), scoresmod4.std()*2))


#%%
# We will continue with year and nodes and compare the scores


#%%
X = data1.iloc[:, [1,2]]
y = data1.iloc[:, 3]


#%%
classifier_remod5 = KNeighborsClassifier(n_neighbors=32,weights='distance',metric='minkowski',p=2)


#%%
# Using remod
scoresmod5 = cross_val_score(classifier_remod5, X,y, cv=10)
print 'Cross-validated scores:', scores


#%%
# Print the mean score and variance
print('Accuracy: %0.2f (+/- %0.2f)'% (scoresmod5.mean(), scoresmod5.std()*2))


#%%
predicted_remod5 = cross_val_predict(classifier_remod5, X, y, cv=10)

print classification_report(y,predicted_remod5)


#%%
# We shall proceed to try with individual features


#%%
#Using age
X = data1.iloc[:, [0]]
y = data1.iloc[:, 3]


#%%
classifier_remod6 = KNeighborsClassifier(n_neighbors=32,weights='distance',metric='minkowski',p=2)


#%%
# Using remod
scoresmod6 = cross_val_score(classifier_remod6, X,y, cv=10)
print 'Cross-validated scores:', scores


#%%
# Print the mean score and variance
print('Accuracy: %0.2f (+/- %0.2f)'% (scoresmod6.mean(), scoresmod6.std()*2))


#%%
predicted_remod6 = cross_val_predict(classifier_remod6, X, y, cv=10)

print classification_report(y,predicted_remod6)


#%%
#Using year
X = data1.iloc[:, [1]]
y = data1.iloc[:, 3]


#%%
classifier_remod7 = KNeighborsClassifier(n_neighbors=32,weights='distance',metric='minkowski',p=2)


#%%
# Using remod
scoresmod7 = cross_val_score(classifier_remod7, X,y, cv=10)
print 'Cross-validated scores:', scores


#%%
# Print the mean score and variance
print('Accuracy: %0.2f (+/- %0.2f)'% (scoresmod7.mean(), scoresmod7.std()*2))


#%%
predicted_remod7 = cross_val_predict(classifier_remod7, X, y, cv=10)

print classification_report(y,predicted_remod7)


#%%
#Using nodes
X = data1.iloc[:, [2]]
y = data1.iloc[:, 3]


#%%
classifier_remod8 = KNeighborsClassifier(n_neighbors=32,weights='distance',metric='minkowski',p=2)


#%%
# Using remod
scoresmod8 = cross_val_score(classifier_remod8, X,y, cv=10)
print 'Cross-validated scores:', scores


#%%
# Print the mean score and variance
print('Accuracy: %0.2f (+/- %0.2f)'% (scoresmod8.mean(), scoresmod8.std()*2))


#%%
predicted_remod8 = cross_val_predict(classifier_remod8, X, y, cv=10)

print classification_report(y,predicted_remod8)

#%% [markdown]
# ### Model 2: Decision Tree

#%%
#Load Packages

from sklearn.tree import DecisionTreeClassifier, export_graphviz


#%%
# We shall use age and nodes
A = data1.iloc[:, [0,2]]
b = data1.iloc[:, 3]

from sklearn.model_selection import train_test_split

#Split the datset into training set and test set
A_train, A_test, b_train, b_test = train_test_split(A,b,test_size=0.2,random_state=0)


#%%
#fit the data
#Select decision tree classfier
clf=DecisionTreeClassifier(criterion = "gini", random_state =100,
                              max_depth=3, min_samples_leaf=5)

fit=clf.fit(A_train,b_train)


#%%
#predicting the results
b_pred=fit.predict(A_test)
b_pred


#%%
#Measure confusion matrix
from sklearn.metrics import confusion_matrix
bm=confusion_matrix(b_test,b_pred)
print bm


#%%
#Classification precision/recall/f1score
from sklearn.metrics import classification_report
print classification_report(b_test,b_pred)


#%%
#Visualising the tree
from sklearn import tree


#%%
#def visualize_tree(tree, feature_names):
with open('haberman_data2.dot', 'w') as f:
    f = tree.export_graphviz(clf,feature_names=["age","node"],out_file=f, filled=True, rounded=True, special_characters=True)

    


#%%
from sklearn.metrics import accuracy_score
print "Accuracy is ", accuracy_score(b_test,b_pred)*100


