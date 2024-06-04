import csv
import numpy as np
import math
import numpy.linalg as la


def createDataSets():
    # Specify the path to your CSV file
    csv_file_path = 'Housing.csv'

    # Initialize lists to store the labels, row vector, and the remaining float values


    # Open the CSV file
    with open(csv_file_path, mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        first_row = next(csv_reader) 
        m = len(first_row)
        Features_names = []
        Y_train = np.ndarray(m,)
        X_train = np.ndarray(m,)

        # Read the first row as labels
        Features_names = next(csv_reader)
        
        # Iterate over the remaining rows in the CSV file
        for row in csv_reader:
            np.append(Y_train,row[0])  # Append the first column value to the row vector
            trainEg = [float(value) for value in row[1:]]  # Convert the remaining values to floats
            np.append(X_train,trainEg)

    Y_train = [float(value) for value in Y_train]
    Y_train = scaleFeatures(Y_train)
    first_col = [row[0] for row in X_train]
    scaled_first_col = scaleFeatures(first_col)
    for i in range(len(X_train)):
        X_train[i][0] = scaled_first_col[i]
    
    print(X_train)
    print(Y_train)
    print(Features_names)
    #return Features_names, Y_train, X_train

def scaleFeatures(vector):
    max_val = max(vector)
    vector/=max_val
    return vector

def prediction(X_predict, w ,b):
    return np.dot(w, X_predict) + b

def costFuction(w,b, X_train, Y_train):
    m = len(w)
    cost =0
    for i in range(m):
        y_ = np.dot(m,X_train[i]) +b
        cost += math.pow(( y_ - Y_train),2)
    cost/=(2*m)


def gradientDescent(X_train, Y_train,w,b):
    m = len(X_train)
    dw =0
    db=0
    for i in range(m):
        f = np.dot(m,X_train[i]) +b
        dw_i = (f - Y_train[i]) * X_train[i]
        db_i = (f - Y_train[i])
        dw+= dw_i
        db+= db_i
    dw/=m
    db/=m
    return dw,db

def optimize(num_iters,X_train, Y_train,learning_rate):
    m = len(dw)
    w = np.zeros(m, dtype= float)
    b=0.0
    for k in range(num_iters):
        dw,db = gradientDescent(X_train, Y_train,w,b)
        w-= learning_rate*dw
        b-= learning_rate*db
        if(k%100==0):
            print(costFuction(w,b,X_train,Y_train))
    return w,b


def model():
    features , X_train, Y_train = createDataSets()
    w ,b = optimize(1000,X_train,Y_train,0.5) 
    X_predict = [] #add values to make prediction from 
    prediction = prediction(X_predict,w,b)  
    print(prediction)








    





