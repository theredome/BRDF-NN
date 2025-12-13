import numpy as np



def forward_pass(x):
    global hidden_layer_y
    global output_layer_y
    #Activation for Hidden
    for i, w in enumerate(hidden_layer_w):
        z = np.dot(w.x)
        hidden_layer_y[i] = np.tanh(z)
    hidden_layer_array = np.concatenate(np.array([1.0], hidden_layer_y))
    #Activation for Outer
    for i, w in enumerate(output_layer_w):
        z = np.dot(w, hidden_output_array)
        output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))