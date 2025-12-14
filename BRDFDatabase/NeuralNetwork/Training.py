import numpy as np
import matplotlib.pyplot as plt


data_path = "alum.txt"
Hidden_layer_size = 64 
LR = 1e-3
EPOCHS = 50
BATCH = 1024
Stopping = 8  #how many EPOCHS we do w/ no testing before stopping
SEED = 42

np.random.seed(SEED)

#Load all the data from alum.txt
#theta_i, phi_i, theta_o, phi_o, red, green, blue

data = np.loadtxt(data_path, dtype=np.float32) #use 4 byte numbers instead of the 8 byte with the 64bit loads

theta_i = data[:, 0]
phi_i = data[:, 1]
theta_o = data[:, 2]
phi_o = data[:, 3]

red = data[:, 4:5]
green = data[:, 5:6]
blue = data[:, 6:7]

y = red.astype(np.float32) 


sin_theta_i, cos_theta_i = np.sin(theta_i), np.cos(theta_i)
sin_phi_i, cos_phi_i = np.sin(phi_i), np.cos(phi_i)
sin_theta_o , cos_theta_o = np.sin(theta_o), np.cos(theta_o)
sin_phi_o , cos_phi_o = np.sin(phi_o), np.cos(phi_o)

#can use stack to have one input matrix X w/ shape (samples, features)
#example: sin(theta)_i, cos(theta)_i, sin(phi)_i, cos(phi)_i, sin(theta)_o, cos(theta)_o, sin(phi)_o, cos(phi)_o]
X = np.stack(
    [sin_theta_i, cos_theta_i,
     sin_theta_o, cos_theta_o,
     sin_phi_i, cos_phi_i,
     sin_phi_o, cos_phi_o], axis=1).astype(np.float32)

N, D = X.shape

# ____We are using this section for training much below_____
idx = np.arange(N)
np.random.shuffle(idx)

X = X[idx]
y = y[idx]
split = N // 2
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

index_list = np.arange(len(X_train))
#___________till here _________________________________________

#reference from page 109 in our textbook
def forward_pass(x):
    global hidden_layer_y
    global output_layer_y
    #Activation for Hidden
    for i, w in enumerate(hidden_layer_w):
        z = np.dot(w,x)
        hidden_layer_y[i] = np.tanh(z)
    hidden_layer_array = np.concatenate((np.array([1.0]), hidden_layer_y))
    #Activation for Outer
    for i, w in enumerate(output_layer_w):
        z = np.dot(w, hidden_layer_array)
        output_layer_y[i] = z
    return output_layer_y[0]


#reference from page 110 in our textbook
def backward_pass(y_truth):
    global hidden_layer_error
    global output_layer_error

    # For our output layer
    for i, y in enumerate(output_layer_y):
        error_prime = (y - y_truth[i]) # MSE loss instead of typical derivative loss that was used in our textbook
        output_layer_error[i] = error_prime
    for i, y in enumerate(hidden_layer_y):
        # Create array weights 
        # hidden neuron i to neurons in the output layer.
        error_weights = []
        for w in output_layer_w:
            error_weights.append(w[i+1])
        error_weight_array = np.array(error_weights)
        # Backpropagate error
        derivative = 1.0 - y**2 # tanh derivative
        weighted_error = np.dot(error_weight_array,output_layer_error)
        hidden_layer_error[i] = weighted_error * derivative
        
#reference from page 110 in our textbook
def adjust_weights(x):
    global output_layer_w
    global hidden_layer_w

    for i, error in enumerate(hidden_layer_error):
        hidden_layer_w[i] -= (x * LR * error) # Update all weights
    hidden_output_array = np.concatenate(
        (np.array([1.0]), hidden_layer_y))
    for i, error in enumerate(output_layer_error):
        output_layer_w[i] -= hidden_output_array * LR * error # Update all weights

#Instantiation and Initialization of our Neurons within our NN
def layer_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count+1))
    for i in range(neuron_count):
        for j in range(1, (input_count+1)):
            weights[i][j] = np.random.uniform(-0.1, 0.1)
    return weights


#matrices and vectors for neurons.
#Part of our chapter 4 page 108
H = Hidden_layer_size
D = X.shape[1] 

#the hidden layers
hidden_layer_w = layer_w(H, D)
hidden_layer_y = np.zeros(H, dtype=np.float32)
hidden_layer_error = np.zeros(H, dtype=np.float32)

# Output layer: 1 neuron
output_layer_w = layer_w(1, H)
output_layer_y = np.zeros(1, dtype=np.float32)
output_layer_error = np.zeros(1, dtype=np.float32)

chart_x = []
chart_y_train = []
chart_y_test = []

#For reporting progress like example in page 108
def show_learning(epoch, train_mse, test_mse):
    global chart_x
    global chart_y_train
    global chart_y_test

    print( "epoch no:", epoch + 1, ", train_mse:", f"{train_mse:6.4f}", ", test_mse:", f"{test_mse:6.4f}")
    chart_x.append(epoch + 1)
    chart_y_train.append(train_mse)
    chart_y_test.append(test_mse)

def plot_learning():
    plt.plot(chart_x, chart_y_train, 'r-',label='training error')
    plt.plot(chart_x, chart_y_test, 'b-', label='test error')
    plt.xlabel('training epochs')
    plt.ylabel('error')
    plt.title("Training vs Testing Error")
    plt.legend()
    plt.grid(True)
    plt.show() 

#We have this section here for stopping to prevent overftting
best_test_mse = np.inf
epochs_since_improve = 0

for epoch in range(EPOCHS):

    show_learning(epoch, train_mse, test_mse)

    # Early stopping here
    if test_mse < best_test_mse:
        best_test_mse = test_mse
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1

    if epochs_since_improve >= Stopping:
        print(f"Early stopping at epoch {epoch + 1}")
        break




#Based on the structure for training loop in page 112 of our textbook

for epoch in range(EPOCHS): # Training the EPOCHS iterations

    np.random.shuffle(index_list) # first step; Randomize order

    for j in index_list:
       
        x = np.concatenate(([1.0], X_train[j]))  
        
        y_truth = np.array([y_train[j, 0]], dtype=np.float32)

        y_pred = forward_pass(x)
        backward_pass(y_truth)
        adjust_weights(x)
       
    train_preds = []
    for j in range(len(X_train)): # Second step; Randomize order
        x = np.concatenate(([1.0], X_train[j]))
        y_pred = forward_pass(x)
        train_preds.append(y_pred)
    train_preds = np.array(train_preds).reshape(-1, 1)
    train_mse = np.mean((train_preds - y_train)**2)

    test_preds = []
    for j in range(len(X_test)):# Third step; Evaluate network
        x = np.concatenate(([1.0], X_test[j]))
        y_pred = forward_pass(x)
        test_preds.append(y_pred)
    test_preds = np.array(test_preds).reshape(-1, 1)
    test_mse = np.mean((test_preds - y_test)**2)

    show_learning(epoch, train_mse, test_mse)# Fourth step; Evaluate network

plot_learning() #Fifth step; Create plot