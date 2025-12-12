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
