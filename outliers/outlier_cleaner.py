#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    
    tlist = []

    for num in range(0,90):
        
        error = net_worths[num][0] - predictions[num][0]
        tlist.append((ages[num][0], net_worths[num][0], error))
    
    cleaned_data = sorted(tlist, key=lambda x: x[2], reverse=True)
    
    cleaned_data = cleaned_data[:80]

    
    return cleaned_data

