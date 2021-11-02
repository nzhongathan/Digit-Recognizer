import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

## get_training_data and get_validation_data adapted from the Agency
def get_training_data():
    f = open('../data/mnist_train.csv', 'r')
    
    lines = f.readlines()
    
    training_images = np.zeros((len(lines), 784))
    training_labels = np.zeros((len(lines), 10))
    index = 0
    for line in lines:
        line = line.strip()
        label = int(line[0])
        training_images[index, :] = np.fromstring(line[2:], dtype=int, sep=',')
        training_labels[index, label - 1] = 1.0
        index += 1
        

    f.close()
    
    # now, instantiate torch tensors
    training_images = torch.tensor(training_images, dtype=torch.float)
    training_labels = torch.tensor(training_labels, dtype=torch.float)
    
    # reshape for minibatch size 20
    # note that if num of total samples is not divisible by minibatch size, we may have to throw out some samples
    training_images = training_images.view(-1, 20, 784)
    training_labels = training_labels.view(-1, 20, 10)
    
    return training_images / 255, training_labels

def get_validation_data():
    f = open('../data/mnist_test.csv', 'r')
    
    lines = f.readlines()
    
    val_images = np.zeros((len(lines), 784))
    val_labels = np.zeros((len(lines), 10))
    index = 0
    for line in lines:
        line = line.strip()
        label = int(line[0])
        val_images[index, :] = np.fromstring(line[2:], dtype=int, sep=',')
        val_labels[index, label - 1] = 1.0
        index += 1
        

    f.close()
    
    val_images = torch.tensor(val_images, dtype=torch.float)
    val_labels = torch.tensor(val_labels, dtype=torch.float)
    
    
    val_images = val_images.view(-1, 20, 784)
    val_labels = val_labels.view(-1, 20, 10)
    
    return val_images / 255, val_labels
  
  
  def train(model, training_images, training_labels, val_images, val_labels, epochs=5, lr=0.0001):
    optimizer = optim.Adam(model.parameters(), lr)
    
    loss_function = nn.MSELoss()
    
    for i in range(epochs):
        model.train()
        
        print("Epoch", i)
        for n in range(training_images.shape[0]):
            x = training_images[n]
            y = training_labels[n]
            
            out = model(x)
            loss = loss_function(out,y)
            #print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        evaluate(model,val_images,val_labels)
        print()
def evaluate(model, val_images, val_labels):
    model.eval()
    num_correct = 0
    total_num = val_labels.shape[0] * val_labels.shape[1]
    with torch.no_grad():
        for n in range(val_images.shape[0]):
            x = training_images[n] 
            y = training_labels[n]
            y = y.numpy()
            truth = np.argmax(y, axis=1)
            
            out = model(x)
            out = out.numpy()
            out = np.argmax(out, axis=1)
            results = np.equal(out, truth)
            num_correct += np.sum(results)
    print("    Percent correct: {}".format(num_correct * 100 / total_num))
    
training_images, training_labels = get_training_data()
val_images, val_labels = get_validation_data()

model = NeuralNet()
train(model, training_images, training_labels, val_images, val_labels, 15, 0.0001)
