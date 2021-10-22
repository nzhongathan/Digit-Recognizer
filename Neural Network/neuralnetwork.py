class NeuralNetwork():
    """
    A Fully Connected Neural Network. There are 784 input layer nodes, 12 hidden layer nodes, and 10 output layer
    nodes.
    """
    def __init__(self):
        
        
        # Nodes
        self.A = np.zeros((784, ))
        self.B = np.zeros((256, ))
        self.C = np.zeros((128, ))
        self.D = np.zeros((64, ))
        self.E = np.zeros((32, ))
        self.F = np.zeros((10, ))
        
        # Weights
        self.M = 2 * np.random.rand(784, 256) - 1
        self.N = 2 * np.random.rand(256, 128) - 1
        self.O = 2 * np.random.rand(128, 64) - 1
        self.P = 2 * np.random.rand(64, 32) - 1
        self.Q = 2 * np.random.rand(32, 10) - 1
        
        # Biases
        self.R = 2 * np.random.rand(256) - 1
        self.S = 2 * np.random.rand(128) - 1
        self.T = 2 * np.random.rand(64) - 1
        self.U = 2 * np.random.rand(32) - 1
        self.V = 2 * np.random.rand(10) - 1
        
        # Before Values
        self.B_before = np.zeros((256, ))
        self.C_before = np.zeros((128, ))
        self.D_before = np.zeros((64, ))
        self.E_before = np.zeros((32, ))
        self.F_before = np.zeros((10, ))
        
        self.M_grad = np.zeros((784, 256))
        self.N_grad = np.zeros((256, 128))
        self.O_grad = np.zeros((128, 64))
        self.P_grad = np.zeros((64, 32))
        self.Q_grad = np.zeros((32, 10))
        
        self.R_grad = np.zeros((256, ))
        self.S_grad = np.zeros((128, ))
        self.T_grad = np.zeros((64, ))
        self.U_grad = np.zeros((32, ))
        self.V_grad = np.zeros((10, ))
        
        
    def forward(self, x): # A->F  R->V
        self.A = x
        self.B_before = np.dot(self.A, self.M) + self.R
        self.B = np.tanh(self.B_before)
        self.C_before = np.dot(self.B, self.N) + self.S
        self.C = np.tanh(self.C_before)
        self.D_before = np.dot(self.C, self.O) + self.T
        self.D = np.tanh(self.D_before)
        self.E_before = np.dot(self.D, self.P) + self.U
        self.E = np.tanh(self.E_before)
        self.F_before = np.dot(self.E, self.Q) + self.V
        self.F = 1 / (1 + np.exp(-1 * self.F_before))
        
    def calculate_loss(self, x, y):
        out = self.forward(x)
        loss = np.sum((self.F - y) ** 2)
        return loss
        
    def backpropagate(self, label):
        dF = -2 * (label - self.F)
        dF_before = dF * sigmoid(self.F_before) * (1 - sigmoid(self.F_before))
        self.Q_grad = np.outer(self.E, dF_before)
        self.C_grad = dF_before
        
        dE = np.dot(dF_before, np.transpose(self.Q))
        dE_before = dE * (1 / (np.cosh(self.E_before) ** 2))
        self.P_grad = np.outer(self.D, dE_before)
        self.U_grad = dE_before
        
        dD = np.dot(dE_before, np.transpose(self.P))
        dD_before = dD * (1 / (np.cosh(self.D_before) ** 2))
        self.O_grad = np.outer(self.C, dD_before)
        self.T_grad = dD_before
        
        dC = np.dot(dD_before, np.transpose(self.O))
        dC_before = dC * (1 / (np.cosh(self.C_before) ** 2))
        self.N_grad = np.outer(self.B, dC_before)
        self.S_grad = dC_before
        
        dB = np.dot(dC_before, np.transpose(self.N))
        dB_before = dB * (1 / (np.cosh(self.B_before) ** 2))
        self.M_grad = np.outer(self.A, dB_before)
        self.R_grad = dB_before
    
    def update(self, lr):
        self.M -= lr * self.M_grad
        self.N -= lr * self.N_grad
        self.O -= lr * self.O_grad
        self.P -= lr * self.P_grad
        self.Q -= lr * self.Q_grad
        
        self.R -= lr * self.R_grad
        self.S -= lr * self.S_grad
        self.T -= lr * self.T_grad
        self.U -= lr * self.U_grad
        self.V -= lr * self.V_grad
        
    def train(self, epochs, train_images, train_labels, val_images, val_labels, lr):
        error = []
        for i in range (epochs):
            print("Epoch", i)
            for j in range (len(train_images)):
                self.forward(train_images[j])
                if j % 10000 == 0:
                    print(np.sum((self.F - train_labels) ** 2) / len(train_images))
                self.backpropagate(train_labels[j])
                self.update(lr)
                #self.decay = (lr / (i+1))/2;
                #lr *= (1. / (1. + self.decay * i))
            error.append(self.evaluate(val_images, val_labels))
            print("Accuracy: ", error[i], "; Learning Rate: ", lr)
            print("============================================")
            if i >= 2 and abs(error[i] - error[i-2]) <= 0.0001:
                break;
            
            
    def evaluate(self, val_images, val_labels):
        total = 0
        for i in range(len(val_images)):
            self.forward(val_images[i])
            if np.argmax(self.F) == np.argmax(val_labels[i]):
                total+=1
            else:
                continue
        return total/len(val_images)

    
