import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import math
from sklearn.multiclass import OneVsRestClassifier
import plotly.graph_objects as go


array = [2,4]
tensor = torch.Tensor(array)

from torch.autograd import Variable
x = Variable(tensor, requires_grad= True)

print(x)

y = x**2
print(" y =  ",y)


# recap o equation o = 1/2*sum(y)
o = (1/2)*sum(y)
print(" o =  ",o)

# backward
o.backward() # calculates gradients

# As I defined, variables accumulates gradients. In this part there is only one variable x.
# Therefore variable x should be have gradients
# Lets look at gradients with x.grad
print("gradients: ",x.grad)




# As a car company we collect this data from previous selling
# lets define car prices
car_prices_array = [3,4,5,6,7,8,9]
car_price_np = np.array(car_prices_array,dtype=np.float32)
#print(car_price_np)
car_price_np = car_price_np.reshape(-1,1)
car_price_tensor = Variable(torch.from_numpy(car_price_np))
print(car_price_tensor)

# lets define number of car sell
number_of_car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)
number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))

# lets visualize our data
import matplotlib.pyplot as plt
plt.scatter(car_prices_array,number_of_car_sell_array)
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Car Price$ VS Number of Car Sell")
plt.show()

'''
    Now this plot is our collected data
    We have a question that is what will be number of car sell if the car price is 100$
    In order to solve this question we need to use linear regression.
    We need to line fit into this data. Aim is fitting line with minimum error.
    Steps of Linear Regression
        create LinearRegression class
        define model from this LinearRegression class
        MSE: Mean squared error
        Optimization (SGD:stochastic gradient descent)
        Backpropagation
        Prediction
    Lets implement it with Pytorch

'''


# Linear Regression with Pytorch

# libraries
import torch      
from torch.autograd import Variable     
import torch.nn as nn 
import warnings
warnings.filterwarnings("ignore")

# create class
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(LinearRegression,self).__init__()
        # Linear function.
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.linear(x)
    
# define model
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim,output_dim) # input and output size are 1


# MSE
mse = nn.MSELoss()

# Optimization (find parameters that minimize error)
learning_rate = 0.02   # how fast we reach best parameters
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

# train model
loss_list = []
iteration_number = 1001
for iteration in range(iteration_number):
        
    # optimization
    optimizer.zero_grad() 
    
    # Forward to get output
    results = model(car_price_tensor)
    
    # Calculate Loss
    loss = mse(results, number_of_car_sell_tensor)
    
    # backward propagation
    loss.backward()
    
    # Updating parameters
    optimizer.step()
    
    # store loss
    loss_list.append(loss.data)
    
    # print loss
    if(iteration % 50 == 0):
        print('epoch {}, loss {}'.format(iteration, loss.data))

plt.plot(range(iteration_number),loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.show()



'''

    Number of iteration is 1001.
    Loss is almost zero that you can see from plot or loss in epoch number 1000.
    Now we have a trained model.
    While usign trained model, lets predict car prices.

'''


# predict our car price 
predicted = model(car_price_tensor).data.numpy()
plt.scatter(car_prices_array,number_of_car_sell_array,label = "original data",color ="red")
plt.scatter(car_prices_array,predicted,label = "predicted data",color ="blue")


# predict if car price is 10$, what will be the number of car sell
print(np.array([10],dtype=np.float64).dtype)
predicted_10 = model(torch.from_numpy(np.array([10]).astype(np.float32))).data.numpy()
print(predicted_10)
plt.scatter(10,predicted_10.data,label = "car price 10$",color ="green")
plt.legend()
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Original vs Predicted values")
plt.show()





class NNClassifier(nn.Module):
    def __init__ (self,input,output):
        super(NNClassifier,self).__init__()
        self.linear = nn.Linear(input,output)

    def forward(self, x):
        out = self.linear(x)
        return out


data = pd.read_csv("PTTL/drug200.csv")
print(data)

data_label =  LabelEncoder().fit_transform(data[["Drug"]])#pd.get_dummies(data["Drug"])
data["Sex_Encoded"] = LabelEncoder().fit_transform(data[["Sex"]])
data["BP_Encoded"] = LabelEncoder().fit_transform(data[["BP"]])
data["Cholesterol_Encoded"] = LabelEncoder().fit_transform(data[["Cholesterol"]])
u_data = data[["Age","Sex_Encoded","BP_Encoded","Cholesterol_Encoded","Na_to_K"]]
print(u_data)

result = pd.concat([u_data, pd.DataFrame(data_label)], axis=1)
u_data = result
x_data = u_data[["Age","Sex_Encoded","BP_Encoded","Cholesterol_Encoded","Na_to_K"]]
y_data = u_data.drop(["Age","Sex_Encoded","BP_Encoded","Cholesterol_Encoded","Na_to_K"], axis=1)
y_data

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=4)



# Instantiate Model Class
input_dim =  5# size of image px*px
output_dim = 5  # labels 0,1,2,3,4,5,6,7,8,9

# create logistic regression model
model = NNClassifier(input_dim, output_dim)

# Cross Entropy Loss  
error = nn.CrossEntropyLoss()

# SGD Optimizer 
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



# batch_size, epoch and iteration
batch_size = 10
n_iters = 1000
num_epochs = n_iters / (input_dim / batch_size)
num_epochs = int(num_epochs)

X_train = X_train.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)

X_test = X_test.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

featuresTrain = torch.from_numpy(X_train)
targetsTrain = torch.from_numpy(y_train) .type(torch.LongTensor)# data type is long
# Traning the Model
count = 0
loss_list = []
iteration_list = []
for epoch in range(num_epochs):
    for i in range(0,len(X_train)):
        
        # Define variables
        x = featuresTrain[i]
        
        y = targetsTrain[i]
        #print(x)
        train = Variable(x)
        labels = Variable(y)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        #train = train.double()
        outputs = model(train)
        
        # Calculate softmax and cross entropy loss
        # print(len(outputs))
        # print(len(labels))
        loss = error(outputs.reshape(1,-1), labels)
        
        # Calculate gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        # Prediction
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Predict test dataset
            for images, labels in zip(X_test,y_test):
                test = Variable(images)
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.reshape(1,-1).data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))
