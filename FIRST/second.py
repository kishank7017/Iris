import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mlt
import pandas as pd
from sklearn.model_selection import train_test_split


#class to predict flower type (three type of flower) by lengh width of sepal and petal (4 input) and 3 output as name of flower
#setosa versicolor virginica
class Flower(nn.Module):
    #constructor 
    #neural network with two hidden layer h1 and h2
    def __init__(self,input_feature=4,h1=8,h2=8,output_feature=3 ):
        super().__init__()  #instanciate out nn.Module
        #fully connected1 goint from input to h1
        self.fcc1=nn.Linear(input_feature,h1)
        self.fcc2=nn.Linear(h1,h2)
        self.out=nn.Linear(h2,output_feature)
    #now to move data forward we create a funtion
    def forward(self,x):
        # relu return the output 0 if input is less than 0 and return input if input bigger than 0
        # relu=Max(0,input)
        x=F.relu(self.fcc1(x))
        x=F.relu(self.fcc2(x))
        x=self.out(x)
        return x
    
#making a seed for randomisation
torch.manual_seed(42)

#creating an instance of Flower
flower1=Flower()

#creating a url to improt data
url="https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
#df=data frame
my_df=pd.read_csv(url)

#changing last column (species) to integer

my_df['species']=my_df['species'].replace('setosa',0)
my_df['species']=my_df['species'].replace('versicolor',1)
my_df['species']=my_df['species'].replace('virginica',2)

# Train the nn X is conventionally uppercase and y is lowercase
X=my_df.drop('species',axis=1)
y=my_df['species']

#convert into numpy arrays
X=X.values
y=y.values


#train
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=41)

#CONVERTING X features INTO FLOAT
X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)

#converting y lable into tensor long
Y_train=torch.LongTensor(Y_train)
Y_test=torch.LongTensor(Y_test)

#set the criterian to mearusre the error
criterian=nn.CrossEntropyLoss()

#choosee optimiser and learning rate
optimiser=torch.optim.Adam(flower1.parameters(),lr=0.1)

#train your model
# epoch is one run through the neural network

epochs=100
losses=[]
for i in range(epochs):
    # going forward and getting a prediction
    y_pred=flower1.forward(X_train)

    #calculating loss/error
    loss=criterian(y_pred,Y_train)

    #keep track of losses
    losses.append(loss.detach().numpy())

    #print every 10 epoch
    if(i%10==0):
        print(f'Epoch: {i}and loss {loss}')
    

    #do backpropogation
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
