import numpy as np

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