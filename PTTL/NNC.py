
# libraries
import torch      
from torch.autograd import Variable     
import torch.nn as nn 
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class NNClassifier(nn.Module):
    def __init__ (self,input,hidden,output):
        super(NNClassifier,self).__init__()
        # Linear function 1: 784 --> 150
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()
        
        # Linear function 2: 150 --> 150
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        #self.tanh2 = nn.Tanh()
        self.relu2 = nn.ReLU()
        
        # Linear function 3: 150 --> 150
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        #self.elu3 = nn.ELU()
        self.relu3 = nn.ReLU()
        # Linear function 4 (readout): 150 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        #out = self.tanh2(out)
        out = self.relu2(out)

        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        #out = self.elu3(out)
        out = self.relu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
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
hidden_dim = 10
output_dim = 5  # labels 0,1,2,3,4,5,6,7,8,9

# create logistic regression model
model = NNClassifier(input_dim, hidden_dim,output_dim)

# Cross Entropy Loss  
error = nn.CrossEntropyLoss()

# SGD Optimizer 
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



# batch_size, epoch and iteration
batch_size = 100
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


from torchviz import make_dot

batch = next(iter(featuresTrain))
yhat = model(batch) # Give dummy batch to forward().
make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

# Saving & Loading Model for Inference
# Save/Load state_dict (Recommended)

# Save:
torch.save(model.state_dict(), "PTTL/NNC.pt")

# Load:
Loded_model = NNClassifier(input_dim,hidden_dim,output_dim)
Loded_model.load_state_dict(torch.load("PTTL/NNC.pt"))
Loded_model.eval()
o_data = Loded_model(Variable(X_test))
o_predicted = torch.max(o_data.reshape(1,-1).data, 1)[1]

# Total number of labels
o_total = len(Variable(y_test))

# Total correct predictions
correct += (o_predicted == Variable(y_test)).sum()
            
accuracy = 100 * correct / float(o_total)
            
######################
#determining the architecutre of neural network
plt.scatter(pd.DataFrame(iteration_list),pd.DataFrame(np.asarray(loss_list)))
plt.show()

plt.scatter(y_data,x_data["Age"])
plt.show()


plt.scatter(x_data["Sex_Encoded"],y_data)
plt.show()


plt.scatter(x_data["BP_Encoded"],y_data)
plt.show()


plt.scatter(x_data["Cholesterol_Encoded"],y_data)
plt.show()


plt.scatter(x_data["Na_to_K"],y_data)
plt.show()