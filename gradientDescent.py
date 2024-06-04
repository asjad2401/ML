import csv
import numpy as np
import math
import numpy.linalg as la

def createDataSets():
    # Specify the path to your CSV file
    csv_file_path = 'Housing.csv'

    # Initialize lists to store the labels, row vector, and the remaining float values
    Features_names = []
    Y_train = []
    X_train = []

    # Open the CSV file
    with open(csv_file_path, mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        
        # Read the first row as labels
        Features_names = next(csv_reader)
        
        # Iterate over the remaining rows in the CSV file
        for row in csv_reader:
            Y_train.append(float(row[0]))  # Append the first column value to the row vector and convert to float
            trainEg = [float(value) for value in row[1:]]  # Convert the remaining values to floats
            X_train.append(trainEg)

    # Convert lists to numpy arrays
    Y_train = np.array(Y_train)
    X_train = np.array(X_train)
    
    # Normalize the Y_train vector
    Y_train = scaleFeatures(Y_train)
    
    # Extract the first column from X_train for normalization
    first_col = X_train[:, 0]
    
    # Normalize the first column
    scaled_first_col = scaleFeatures(first_col)
    
    # Replace the first column in X_train with the normalized values
    X_train[:, 0] = scaled_first_col
    
    print(X_train)
    print(Y_train)
    print(Features_names)
    return Features_names, Y_train, X_train

def scaleFeatures(vector):
    max_val = max(vector)
    print(max_val)
    return vector / max_val

def prediction(X_predict, w, b):
    return np.dot(w, X_predict) + b

def costFunction(w, b, X_train, Y_train):
    m = len(Y_train)
    cost = 0
    for i in range(m):
        y_ = np.dot(w, X_train[i]) + b
        cost += (y_ - Y_train[i]) ** 2
    return cost / (2 * m)

def gradientDescent(X_train, Y_train, w, b):
    m = len(Y_train)
    dw = np.zeros(w.shape)
    db = 0
    for i in range(m):
        f = np.dot(w, X_train[i]) + b
        dw_i = (f - Y_train[i]) * X_train[i]
        db_i = (f - Y_train[i])
        dw += dw_i
        db += db_i
    dw /= m
    db /= m
    return dw, db

def optimize(num_iters, X_train, Y_train, learning_rate):
    n = X_train.shape[1]  # number of features
    w = np.zeros(n, dtype=float)
    b = 0.0
    for k in range(num_iters):
        dw, db = gradientDescent(X_train, Y_train, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        if k % 10 == 0:
            print(f"Iteration {k}: Cost {costFunction(w, b, X_train, Y_train)}")
    return w, b

def model():
    Features_names, Y_train, X_train = createDataSets()
    w, b = optimize(10, X_train, Y_train, 0.1)
    X_predict = [0.5432098,3,2,2,1,0,0,0,1,2,0,1]  # add values to make prediction from
    prediction_val = prediction(X_predict, w, b)
    print(prediction_val)

# Call the model function to run the entire process
model()
