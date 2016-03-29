#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import tree

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

accuracy = clf.score(features_test, labels_test)
print "accuracy is", accuracy

# of POIs in test set
count = 0
for poi in labels_test:
    if poi == 1:
        count +=1
print "Number of POIs in test set is", count
print "Number of people in test set is", len(labels_test)

# look at predictions of classifier
pred = clf.predict(features_test)
true_pos = 0
false_pos = 0
for num in range(0,len(pred)):
    if pred[num] == 1 and labels_test[num] == 1:
        true_pos += 1
    if pred[num] == 1 and labels_test[num] == 0:
        false_pos +=1
print "Number of true positives is", true_pos
print "Number of false positives is", false_pos

# calculate recall and precision
from sklearn.metrics import precision_score
precision = precision_score(labels_test, pred) 
print "Precision is", precision

from sklearn.metrics import recall_score
recall = recall_score(labels_test, pred)
print "Recall is", recall
        


