#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Number of persons is", len(enron_data)  # of persons

#for key in enron_data:
 #   print len(enron_data[key]) # of features per person
    
poicount = 0
poitpnan = 0
for key in enron_data:
    if enron_data[key]['poi'] == True:
        poicount = poicount + 1
        if enron_data[key]['total_payments'] == 'NaN':
            poitpnan = poitpnan + 1
print "Number of persons of interest is", poicount  # of persons of interest

print "Number of POIs with NaN for total payments is", poitpnan 
            
#print enron_data['PRENTICE JAMES']['total_stock_value']        

#print enron_data['COLWELL WESLEY']['from_this_person_to_poi']

#print enron_data['SKILLING JEFFREY K']['exercised_stock_options']  

#print 'Skilling', enron_data['SKILLING JEFFREY K']['total_payments']

#print 'Fastow', enron_data['FASTOW ANDREW S']['total_payments']

#print 'Lay', enron_data['LAY KENNETH L']['total_payments']

salcount = 0
emaddcount = 0
totalpaycount = 0
for key in enron_data:
    if enron_data[key]['salary'] != 'NaN':
        salcount = salcount + 1
    if enron_data[key]['email_address'] != 'NaN':
        emaddcount = emaddcount + 1
    if enron_data[key]['total_payments'] == 'NaN':
        totalpaycount = totalpaycount + 1

print "Number of people with quantified salaries", salcount
print "Number of people with email addresses", emaddcount
print "Number of people who have NaN for total payments", totalpaycount

import math
numpersons = len(enron_data)
pctg = float(totalpaycount * 100 / numpersons)
print "Percentage of people with NaN as TP is", pctg
    
print enron_data['SKILLING JEFFREY K']
