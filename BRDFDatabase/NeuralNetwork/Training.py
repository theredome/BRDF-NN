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

