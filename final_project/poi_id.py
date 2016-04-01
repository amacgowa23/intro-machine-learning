#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

"""
STEP 1: UNDERSTAND THE DATASET
"""
## Initial feature list prior to tuning.
## The first feature must be "poi".

features_list = ['poi','salary', 'bonus', 'total_payments',
                 'exercised_stock_options', 'restricted_stock', 
                 'total_stock_value'] 

## Load the dictionary containing the dataset.
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## Find out more about persons and features.
print
print "Total number of persons is", len(data_dict)  

persons = []
features = []
feature_count = 0
nan_count = 0
for person in data_dict:
    persons.append(person)
    feature_count = len(data_dict[person])
    if person == 'METTS MARK':
        print person, "has", feature_count, "features"        
    if feature_count != 21:  #confirm everyone has 21 features 
        print person, "has ", feature_count, "features"     
    for feature in data_dict[person]:  # create list of features
        if feature not in features:
            features.append(feature)
        if data_dict[person][feature] == 'NaN':
            nan_count += 1

print
print "List of features in dataset is", features
print
print "Total number of features is", len(features)
print
print "Number of features that have a value of NaN is", nan_count
print

## Find out more about persons of interest (POIs)
poi_count = 0
poi_nan_count = 0
not_poi_count = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
        poi_count += 1
    if data_dict[person]['poi'] == 'NaN':
        poi_nan_count += 1
    if data_dict[person]['poi'] == False:
        not_poi_count += 1
        
print "Number of persons of interest (POI) is", poi_count  
print "Number of persons with POI value of NaN is", poi_nan_count 
print "Number of persons who aren't POIs is", not_poi_count 
print

## Find out who the POIs are
poi_list = []
for person in data_dict:
    if data_dict[person]['poi'] == True:
        poi_list.append(person)
print "List of POIs"
print poi_list
print

## Determine the number of 'NaN' values per feature.
nan_feature_dict = {}
for feature in features:
    nan_feature_dict[feature] = 0 # initialize feature dictionary to have 0 'NaN' values
for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature] == 'NaN':
            nan_feature_dict[feature] += 1
print "Number of NaN values per feature"
print nan_feature_dict

## Group together all features that have integer values
feature_dict = {}    
for feature in features:
    feature_dict[feature] = []

for person in data_dict:
    for feature in data_dict[person]:
        if type(data_dict[person][feature]) == int:
           feature_dict[feature].append(data_dict[person][feature]) 

# print "Details of features with integer values", feature_dict

## Identify and remove outliers

# Remove outlier called 'TOTAL' (after initial visualization below)
data_dict.pop('TOTAL',0)  

## Visualize to identify the name 'TOTAL' as an outlier
data1 = featureFormat(data_dict, ['salary','bonus', 'poi'])

for point in data1:
    salary = point[0]
    bonus = point[1]
    color = 'blue'
    if point[2] == 1:
        color = 'red'
    matplotlib.pyplot.scatter(salary, bonus, c = color)

print
print "Scatterplot of Bonus vs Salary (POIs in red)"
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

"""
STEP 2: ADD AND OPTIMIZE FEATURES
"""

## Create new features "fraction_from_poi" and "fraction_to_poi"  

def computeFraction(poi_messages, all_messages):
    """ given a number of messages to/from POI (numerator) 
        and total number of messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if poi_messages != 'NaN' and all_messages != 'NaN':
        if all_messages > 0:
            fraction = float(poi_messages) / float(all_messages)        
    return fraction

for person in data_dict:
    newfeat_dict = data_dict[person]
    
    from_poi_to_this_person = newfeat_dict["from_poi_to_this_person"]
    to_messages = newfeat_dict["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    newfeat_dict["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = newfeat_dict["from_this_person_to_poi"]
    from_messages = newfeat_dict["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    newfeat_dict["fraction_to_poi"] = fraction_to_poi

    data_dict[person] = newfeat_dict

## Display a visualization of these two new features

data2 = featureFormat(data_dict, ['fraction_from_poi','fraction_to_poi', 'poi'])

print "Scatterplot of New Features (POIs in Red)"
for point in data2:
    from_POI = point[0]
    to_POI = point[1]
    color = 'blue'
    if point[2] == 1:
        color = 'red'  
    matplotlib.pyplot.scatter(from_POI, to_POI, c = color)

matplotlib.pyplot.xlabel("fraction_from_poi")
matplotlib.pyplot.ylabel("fraction_to_poi")
matplotlib.pyplot.show()

## Add new features to features_list

features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')

## Try different features and evaluate algorithm performance
features_list_try1 = ['poi', 'bonus', 'exercised_stock_options', 
'restricted_stock', 'total_stock_value', 
'fraction_from_poi', 'fraction_to_poi']  
# Result: recall=0, precision=0, accuracy < 0.86 for all algorithms

features_list_try2 = ['poi','salary', 'to_messages', 'deferral_payments', 
'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 
'shared_receipt_with_poi', 'restricted_stock_deferred', 
'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 
'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 
'long_term_incentive', 'from_poi_to_this_person',
'fraction_from_poi', 'fraction_to_poi'] 
# Result: Result: accuracy = 0.87 for all algorithms, 
# recall, precision = 0 for SVM, precision = 0.5 for Naive Bayes 
# and Decision Tree, and recall = 1 for Decision Tree, 
# recall = 0.5 for Naive Bayes

features_list_try3 = ['poi','bonus', 'restricted_stock', 
'expenses', 'deferred_income',
'fraction_from_poi', 'fraction_to_poi'] 
# Result: recall, precision = 0 for Decision Tree & SVM, 
# recall = 0.5, precision = 0.33 for Naive Bayes, accuracy = 0.71 to 0.86

## Final list of features
features_list = ['poi', 'bonus', 'total_payments',
                 'exercised_stock_options', 'restricted_stock',
                 'fraction_from_poi', 'fraction_to_poi'] 

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

"""
STEP 3: ALGORITHM SELECTION & TUNING
"""
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

## Separate data into training and test sets using cross validation.

"""
## Option 1
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
# Decision Tree Classifier Results with train_test_split cross validation
# {'True Neg': 34, 'False Neg': 4, 'False Pos': 3, 'True Pos': 2}
# {'Recall': 0.3333333333333333, 'Precision': 0.4, 'Accuracy': 0.8372093023255814}
"""

## Option 2 - training/test set split tuned for better algorithm performance

from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels, 1000,random_state = 42)
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

## Decision tree supervised learning classifier with parameter tuning
from sklearn import tree
from sklearn import grid_search
parameters = {'criterion':("gini", "entropy"), 
              'splitter':("best", "random"), 
              'max_features':("auto", "sqrt", "log2")}
dtr = tree.DecisionTreeClassifier()
clf = grid_search.GridSearchCV(dtr, parameters)
clf = clf.fit(features_train, labels_train)

""" 
# Feature importance for decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
print
print "Feature importance for decision tree classifier"
print features_list[1:]
count = 0
for feature in clf.feature_importances_:
    print [feature, count]
    count += 1
"""

## Naive Bayes supervised learning classifier 
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
clf1 = clf1.fit(features_train, labels_train)

## Random Forest supervised learning classifier with manual parameter tuning
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators=10, min_samples_split=10, 
                             criterion='entropy', min_samples_leaf=10,
                             bootstrap=False)
clf2 = clf.fit(features_train, labels_train)

        
"""
STEP 4: EVALUATION & VALIDATION
"""

# Calculate accuracy, recall and precision
def classifier_eval(clf, features_test, labels_test): 
    pred = clf.predict(features_test)
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for num in range(0,len(pred)):
        if pred[num] == 1 and labels_test[num] == 1:
            true_pos += 1
        if pred[num] == 1 and labels_test[num] == 0:
            false_pos +=1
        if pred[num] == 0 and labels_test[num] == 0:
            true_neg +=1
        if pred[num] == 0 and labels_test[num] == 1:
            false_neg +=1
 #   return true_pos, false_pos, true_neg, false_neg
    total_pred = true_neg + false_neg + false_pos + true_pos
    accuracy = 1.0*(true_pos + true_neg)/max(1,total_pred)
    precision = 1.0*true_pos/max(1,(true_pos+false_pos))
    recall = 1.0*true_pos/max(1,(true_pos+false_neg))
    results = {"Accuracy":accuracy, "Precision":precision, "Recall":recall}
    print {"True Pos":true_pos, "False Pos":false_pos, 
    "True Neg":true_neg, "False Neg":false_neg}
    return results
    
print
print "Decision Tree Classifier Results"
print classifier_eval(clf, features_test, labels_test)
print
print "Naive Bayes Classifier Results"
print classifier_eval(clf1, features_test, labels_test)
print
print "Random Forest Classifier Results"
print classifier_eval(clf2, features_test, labels_test)
print
 
### Dump classifier, dataset, and features_list so anyone can
### check the results. Make sure that the version of poi_id.py that you 
### submit can be run on its own and generates the necessary .pkl files 
### for validating the results.

dump_classifier_and_data(clf, my_dataset, features_list)
