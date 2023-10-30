#!/usr/bin/env python
# coding: utf-8

# In[144]:


import numpy as np
import pandas as pd


# # 3

# ### 3.0

# In[145]:


# Create a list of characters from 'a' to 'z' and space
char_list = [chr(x) for x in range(ord('a'), ord('z')+1)] + [' ']
print(char_list)
smoothing = 0.5


# ### 3.1 - Prior

# In[146]:


prior = {}
for language in ["e","j","s"]:
    prior[language] = np.log((10+1/2)/((10+1/2)*3))


# In[147]:


# Result
for l in prior:
    print(np.exp(prior[l]))
    


# ### 3.2 and 3.3 - Conditional probabilities

# In[148]:


dict_count = {}
for language in ["e","j","s"]:
    dict_count[language] = {}
    for character in char_list:
        dict_count[language][character] = smoothing
    for n in range(10):
        file = 'languageID/' + language + str(n) + '.txt'
        with open(file, 'r') as file:
            content = file.read()
            for c in content:
                if c in char_list:
                    dict_count[language][c] +=1
                else:
                    pass      


# In[149]:


cond_probability = {}
for language in dict_count:
    cond_probability[language] = {}
    total = sum(dict_count[language].values())
    vector = []
    for c in dict_count[language]:
        c_count = dict_count[language][c]
        vector.append(round((c_count/total),4))
        cond_probability[language][c] = np.log(c_count/total)
    print(language)
    print(vector)
    print(sum(vector))
#cond_probability


# In[150]:


cond_probability


# ### 3.4 - Test data

# In[151]:


def test_bag(file):
    doc_dict = {}
    char_list = [chr(x) for x in range(ord('a'), ord('z')+1)] + [' ']
    for character in char_list:
        doc_dict[character] = 0
    with open(file, 'r') as file:
        content = file.read()
        for c in content:
            if c in char_list:
                doc_dict[c] +=1
            else:
                pass   
    return doc_dict

file = 'languageID/e10.txt'

# Result
vector = []
doc_dict = test_bag(file)
for c in doc_dict:
    vector.append(doc_dict[c])
    
print(vector)


# In[152]:


doc_dict


# ### 3.5 - Likelihood

# In[153]:


def likelihood(doc_dict,cond_probability):        
    dict_likelihood = {}
    for language in ["e","j","s"]:
        likelihood_total = 0
        for c in cond_probability[language]:
            count = doc_dict[c]
            prob = cond_probability[language][c]
            likelihood_total += count*prob
        dict_likelihood[language] = likelihood_total
    return dict_likelihood

# Results
dict_likelihood =  likelihood(doc_dict,cond_probability)
for l in dict_likelihood:
    print(l, np.exp(dict_likelihood[l]))
        


# In[154]:


dict_likelihood


# ### 3.6 - Posterior

# In[155]:


def bayes(prior,dict_likelihood):
    dict_bayes = {}
    for language in ["e","j","s"]:
        dict_bayes[language] = dict_likelihood[language] + prior[language]
    return dict_bayes

dict_bayes = bayes(prior,dict_likelihood)

# Results
for l in dict_bayes:
    print(l, np.exp(dict_bayes[l]))


# In[156]:


dict_bayes


# ### 3.7 - Performance

# In[157]:


def prediction(dict_bayes):
    pred = max(dict_bayes, key=lambda k: dict_bayes[k])
    return pred 


# In[158]:


# Run
dict_results = {}
for language in ["e","j","s"]:
    for n in range(10,20,1):
        file = 'languageID/' + language + str(n) + '.txt'
        doc_dict = test_bag(file)
        #print(doc_dict)
        dict_likelihood =  likelihood(doc_dict,cond_probability)
        dict_bayes = bayes(prior,dict_likelihood)
        #print(dict_bayes["e"]/dict_bayes["s"])
        pred = prediction(dict_bayes)
        truth = language
        dict_results[file] = (pred,truth)

dict_results


# In[159]:


# Create Dataframe
final_df = pd.DataFrame()
for l1 in ["e","j","s"]: #["English True","Japanese True","Spanish True"]:
    for l2 in ["e","j","s"]: #["English Predicted","Japanese Predicted","Spanish Predicted"]:
        final_df.loc[l1,l2] = 0

for i in dict_results:
    l1 = dict_results[i][0] # predicted
    l2 = dict_results[i][1] # true
    final_df.loc[l1,l2] += 1

print(final_df)
    


# In[ ]:





# In[ ]:





# # 4.2

# In[173]:


import numpy as np
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt


# ### Functions

# In[174]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

# Cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

# Compute derivatives for the sigmoid activation function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def accuracy_calculation(W1,W2,X_test,Y_test):
    z1 = np.dot(W1, X_test.T)
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1)
    y_hat = softmax(z2).astype(np.int32)
    pred = y_hat.T
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(pred)):
        index = np.where(pred[i] == 1)
        index2 = np.where(Y_test[i] == 1)
        if index == index2:
            TP +=1
    accuracy = (TP) / len(pred)   
    return accuracy        


# In[175]:


# Model Function
# Disclaimer: I am simplifiin the derivative of the loss and the softmax (line 41)
def model_nn(lr, k, d1, batch_size, X_train, Y_train, epochs, X_test,Y_test, random):
    learning_rate = lr
    np.random.seed(10)
    d = X_train.shape[1] # dimentions of the data # flatening images
    output_size = k
    # different weights initialization
    if random == 0:
        W1 = np.random.rand(d1, d) # Initialize W1 randomly (-1,1)
        W2 = np.random.rand(k, d1)
    elif random == 1:    
        W1 = 2 * np.random.rand(d1, d) -1 # Initialize W1 randomly (-1,1)
        W2 = 2 * np.random.rand(k, d1) -1
    elif random == 2:
        W1 = np.zeros((d1, d))  # Initialize W1 zeros
        W2 = np.zeros((k, d1))
    num_batches = len(X_train) // batch_size
    loss_list = []
    accuracy_list = []
    error_list = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = (batch + 1) * batch_size
            # Batch data
            X_batch = X_train[start_idx:end_idx]
            Y_batch = Y_train[start_idx:end_idx]
            
            # Forward (for the entire batch)
            z1 = np.dot(W1, X_batch.T)
            a1 = sigmoid(z1)
            z2 = np.dot(W2, a1)
            y_hat = softmax(z2)
    
            # Loss Computation
            loss = cross_entropy_loss(Y_batch, y_hat.T)
            total_loss += loss
            
            # Backward pass (compute derivatives)
            delta2 = y_hat.T - Y_batch
            delta1 = np.dot(W2.T, delta2.T) * sigmoid_derivative(z1)
            
            grad_W2 = np.dot(delta2.T, a1.T)
            grad_W1 = np.dot(delta1, X_batch)
            
            # Update weights and biases
            W2 -= learning_rate * grad_W2
            W1 -= learning_rate * grad_W1

        accuracy = accuracy_calculation(W1,W2,X_test,Y_test)
        accuracy_list.append(accuracy)
        error_list.append(1-accuracy)
        
        # Compute average loss for this epoch
        average_loss = total_loss / num_batches
        loss_list.append(average_loss)
        
        # Print the loss every 1000 epochs
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {average_loss}, Error: {1-accuracy}, Accuracy: {accuracy}')  

    plt.figure(figsize=(10, 6))  # Set the size of the plot (optional)
    plt.plot(loss_list, marker='o', linestyle='-', color='b')
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.title('Learning Curve (Loss)')
    plt.grid(True)  # Show grid (optional)
    plt.savefig('learning_curve.png')
    plt.show()  # Display the plot

    plt.figure(figsize=(10, 6))  # Set the size of the plot (optional)
    plt.plot(error_list, marker='o', linestyle='-', color='b')
    plt.xlabel('Index')
    plt.ylabel('Error')
    plt.title('Learning Curve (Error)')
    plt.grid(True)  # Show grid (optional)
    plt.savefig('error.png')
    plt.show()  # Display the plot
    
    return loss_list, error_list


# ### Data

# In[176]:


k = 10
data = pd.read_csv("train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
Y_test = to_categorical(Y_dev, num_classes=k).astype(np.int32)
X_dev = data_dev[1:n]
X_dev = X_dev / 255.
X_test = X_dev.T

data_train = data[1000:m].T
Y_train = data_train[0]
Y_train = to_categorical(Y_train, num_classes=k).astype(np.int32)

X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
X_train = X_train.T


# # 4.2

# In[178]:


lr = 0.5
d1 = 300
batch_size = 64 #X_train.shape[0] # 1 is SGD/ 60000 is Full GD
epochs = 1000
random = 1
loss_list_4_4, accuracy_list_4_4 = model_nn(lr, k, d1, batch_size, X_train, Y_train, epochs, X_test, Y_test, random)


# # 4.3

# In[161]:


from sklearn.neural_network import MLPClassifier
import numpy as np

# Create and train the MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(300,), max_iter=1000, random_state=42, activation='logistic',solver='sgd',batch_size = 64, learning_rate='constant', learning_rate_init=0.5)
model.fit(X_train, Y_train)

# Get the loss after each iteration
loss_values = model.loss_curve_
print("Loss after each iteration:", loss_values)


# In[181]:


from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy on test data:", accuracy)


# In[180]:


plt.figure(figsize=(10, 6))  # Set the size of the plot (optional)
plt.plot(loss_values, marker='o', linestyle='-', color='b')
plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('Learning Curve (Loss)')
plt.grid(True)  # Show grid (optional)
plt.savefig('error.png')
plt.show()  # Display the plot


# # 4.4

# In[182]:


lr = 0.5
d1 = 300
batch_size = 64 #X_train.shape[0] # 1 is SGD/ 60000 is Full GD
epochs = 10
random = 2
loss_list44, error_list44 = model_nn(lr, k, d1, batch_size, X_train, Y_train, epochs, X_test, Y_test, random)


# In[183]:


plt.figure(figsize=(10, 6))  # Set the size of the plot (optional)
plt.plot(loss_list44[1:], marker='o', linestyle='-', color='b')
plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('Learning Curve (Loss)')
plt.grid(True)  # Show grid (optional)
plt.savefig('learning_curve.png')
plt.show()  # Display the plot

plt.figure(figsize=(10, 6))  # Set the size of the plot (optional)
plt.plot(error_list44[1:], marker='o', linestyle='-', color='b')
plt.xlabel('Index')
plt.ylabel('Error')
plt.title('Learning Curve (Error)')
plt.grid(True)  # Show grid (optional)
plt.savefig('error.png')
plt.show()  # Display the plot


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




