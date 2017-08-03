import graphlab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def polynomial_sframe(feature,degree):
    poly_sframe = graphlab.SFrame()
    poly_sframe['power_1'] = feature
    if degree > 1:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for pow in range(2,degree+1): 
            name = 'power_' + str(pow)
            poly_sframe[name] = feature.apply(lambda x:x**pow)
    return poly_sframe

sales = graphlab.SFrame('C:\\Machine_Learning\\Rg_wk4\\kc_house_data.gl\\')
sales = sales.sort(['sqft_living','price'])
l2_small_penalty = 1e-5

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None,l2_penalty = l2_small_penalty)
model15.get("coefficients")
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
        poly15_data['power_1'], model15.predict(poly15_data),'-')
plt.show()

(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

fig, ax = plt.subplots(nrows=2,ncols=2)
plt.subplot(2,2,1)
poly15_set1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_set1.column_names() # get the name of the features
poly15_set1['price'] = set_1['price'] # add price to the data since it's the target
model15_set1 = graphlab.linear_regression.create(poly15_set1, target = 'price', features = my_features, validation_set = None,l2_penalty = l2_small_penalty)
model15_set1.get("coefficients")
plt.plot(poly15_set1['power_1'],poly15_set1['price'],'.',
        poly15_set1['power_1'], model15_set1.predict(poly15_set1),'-')
#plt.show()

plt.subplot(2,2,2)
poly15_set2 = polynomial_sframe(set_2['sqft_living'], 15)
my_features = poly15_set2.column_names() # get the name of the features
poly15_set2['price'] = set_2['price'] # add price to the data since it's the target
model15_set2 = graphlab.linear_regression.create(poly15_set2, target = 'price', features = my_features, validation_set = None,l2_penalty = l2_small_penalty)
model15_set2.get("coefficients")
plt.plot(poly15_set2['power_1'],poly15_set2['price'],'.',
        poly15_set2['power_1'], model15_set2.predict(poly15_set2),'-')
#plt.show()

plt.subplot(2,2,3)
poly15_set3 = polynomial_sframe(set_3['sqft_living'], 15)
my_features = poly15_set3.column_names() # get the name of the features
poly15_set3['price'] = set_3['price'] # add price to the data since it's the target
model15_set3 = graphlab.linear_regression.create(poly15_set3, target = 'price', features = my_features, validation_set = None,l2_penalty = l2_small_penalty)
model15_set3.get("coefficients")
plt.plot(poly15_set3['power_1'],poly15_set3['price'],'.',
        poly15_set3['power_1'], model15_set3.predict(poly15_set3),'-')
#plt.show()

plt.subplot(2,2,4)
poly15_set4 = polynomial_sframe(set_4['sqft_living'], 15)
my_features = poly15_set4.column_names() # get the name of the features
poly15_set4['price'] = set_4['price'] # add price to the data since it's the target
model15_set4 = graphlab.linear_regression.create(poly15_set4, target = 'price', features = my_features, validation_set = None,l2_penalty = l2_small_penalty)
model15_set4.get("coefficients")
plt.plot(poly15_set4['power_1'],poly15_set4['price'],'.',
        poly15_set4['power_1'], model15_set4.predict(poly15_set4),'-')
plt.show()

l2_penalty=1e5

fig, ax = plt.subplots(nrows=2,ncols=2)
plt.subplot(2,2,1)
poly15_set1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_set1.column_names() # get the name of the features
poly15_set1['price'] = set_1['price'] # add price to the data since it's the target
model15_set1 = graphlab.linear_regression.create(poly15_set1, target = 'price', features = my_features, validation_set = None,l2_penalty = l2_penalty)
model15_set1.get("coefficients")
plt.plot(poly15_set1['power_1'],poly15_set1['price'],'.',
        poly15_set1['power_1'], model15_set1.predict(poly15_set1),'-')
#plt.show()

plt.subplot(2,2,2)
poly15_set2 = polynomial_sframe(set_2['sqft_living'], 15)
my_features = poly15_set2.column_names() # get the name of the features
poly15_set2['price'] = set_2['price'] # add price to the data since it's the target
model15_set2 = graphlab.linear_regression.create(poly15_set2, target = 'price', features = my_features, validation_set = None,l2_penalty = l2_penalty)
model15_set2.get("coefficients")
plt.plot(poly15_set2['power_1'],poly15_set2['price'],'.',
        poly15_set2['power_1'], model15_set2.predict(poly15_set2),'-')
#plt.show()

plt.subplot(2,2,3)
poly15_set3 = polynomial_sframe(set_3['sqft_living'], 15)
my_features = poly15_set3.column_names() # get the name of the features
poly15_set3['price'] = set_3['price'] # add price to the data since it's the target
model15_set3 = graphlab.linear_regression.create(poly15_set3, target = 'price', features = my_features, validation_set = None,l2_penalty = l2_penalty)
model15_set3.get("coefficients")
plt.plot(poly15_set3['power_1'],poly15_set3['price'],'.',
        poly15_set3['power_1'], model15_set3.predict(poly15_set3),'-')
#plt.show()

plt.subplot(2,2,4)
poly15_set4 = polynomial_sframe(set_4['sqft_living'], 15)
my_features = poly15_set4.column_names() # get the name of the features
poly15_set4['price'] = set_4['price'] # add price to the data since it's the target
model15_set4 = graphlab.linear_regression.create(poly15_set4, target = 'price', features = my_features, validation_set = None,l2_penalty = l2_penalty)
model15_set4.get("coefficients")
plt.plot(poly15_set4['power_1'],poly15_set4['price'],'.',
        poly15_set4['power_1'], model15_set4.predict(poly15_set4),'-')
plt.show()

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

#get batch b index if total recored is n and k fold cv
def getStartEnd(k,b,n):
    for i in xrange(k):
        if(i==b):
            start = (n*i)/k
            end = (n*(i+1))/k-1
            print i, (start, end)
            return (start,end)

(start,end) = getStartEnd(k,3,n)

#validation4 = train_valid_shuffled[start:end+1]
#print int(round(validation4['price'].mean(), 0))

def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    nLen = len(data)
    print(nLen)
    rss_lst = []
    for i in xrange(k):
        (start,end)= getStartEnd(k,i,nLen)
        validation = data[start:end+1]
        #print validation.head()
        train = data[0:start].append(data[end+1:nLen])
        #print train.head()
        model = graphlab.linear_regression.create(train, target = output_name, features = features_list, validation_set = None,l2_penalty = l2_penalty)
        error = validation[output_name] - model.predict(validation)
        error = error * error
        rss = error.sum()
        rss_lst.append(rss)    
    rss_lst = np.asarray(rss_lst)    
    return rss_lst.mean()

poly15_data = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = train_valid_shuffled['price'] # add price to the data since it's the target
rss_lst = k_fold_cross_validation(k,l2_penalty,poly15_data,'price',my_features)

l2_penalty_lst =  np.logspace(3, 9, num=13)

l2_rss = []
for l2 in l2_penalty_lst:
    rss_lst = k_fold_cross_validation(k,l2,poly15_data,'price',my_features)
    l2_rss.append(rss_lst)


l2_optimal = 1.00000000e+03
final_model = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None,l2_penalty = l2_optimal)
test_result = final_model.predict(test)
final_err = test_result - test['price']
final_err = final_err * final_err
final_rss = final_err.sum()


