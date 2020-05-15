#!/usr/bin/python
import sys
import pickle
import pandas as pd
import numpy as np
from pprint import pprint 
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'expenses', 'fraction_to_poi','long_term_incentive']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers

data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('TOTAL')


### Task 3: Create new feature(s)

def fraction_of_emails( sent_messages, total_messages ): #function to divide and find fraction

    fraction = 0
    if sent_messages == "NaN":  # change NaN to 0 for fraction function to work properly
        sent_messages = 0 
    if total_messages == "NaN":
        return 0
    fraction = float(sent_messages)/float(total_messages)


    return fraction


 

new_dict = {} # create dictionary to hold values
for name in data_dict: # each person in dictionary, add to dict

    enron_persons = data_dict[name] # enron_persons = names of the people
    
    poi_to_person = enron_persons["from_poi_to_this_person"] # representing feature
    to_messages = enron_persons["to_messages"] # representing feature
    
    fraction_from_poi = fraction_of_emails(poi_to_person, to_messages) # fraction of poi to person message / to messages
    
    enron_persons["fraction_from_poi"] = fraction_from_poi
  
    person_to_poi = enron_persons["from_this_person_to_poi"] # representing feature of person in dictionary
    from_messages = enron_persons["from_messages"]
    fraction_to_poi = fraction_of_emails(person_to_poi, from_messages) # fraction of person to poi / from messages
   
    new_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    
    enron_persons["fraction_to_poi"] = fraction_to_poi






### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier 
clf = DecisionTreeClassifier(min_samples_split=3, min_samples_leaf=11)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)