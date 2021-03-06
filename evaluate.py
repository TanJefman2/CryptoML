from os import urandom
import numpy as np
from keras.models import model_from_json, load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from pickle import dump

from lime.lime_text import LimeTextExplainer

from keras import backend as K
from keras.models import Model

import matplotlib.pyplot as plt

import pathlib
import csv
import os
import csv

##### INPUTS #####

bit_diff = (0x0020, 0)  # Bit difference
file = '4-plaintext_models/round6-4_plaintext-bit(32, 0).h5'  # Model file location
rounds = 6

##################



################################ GOHR'S SPECK.PY ################################################
# CODE RETRIEVED FROM: https://github.com/agohr/deep_speck/blob/master/speck.py

def WORD_SIZE():
    return(16);

def ALPHA():
    return(7);

def BETA():
    return(2);

MASK_VAL = 2 ** WORD_SIZE() - 1;

def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    c0 = ror(c0, ALPHA());
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA());
    c1 = c1 ^ c0;
    return(c0,c1);

def dec_one_round(c,k):
    c0, c1 = c[0], c[1];
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA());
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA());
    return(c0, c1);

def expand_key(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i);
    return(ks);

def encrypt(p, ks):
    x, y = p[0], p[1];
    for k in ks:
        x,y = enc_one_round((x,y), k);
    return(x, y);

def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);

def check_testvector():
  key = (0x1918,0x1110,0x0908,0x0100)
  pt = (0x6574, 0x694c)
  ks = expand_key(key, 22)
  ct = encrypt(pt, ks)
  if (ct == (0xa868, 0x42f2)):
    print("Testvector verified.")
    return(True);
  else:
    print("Testvector not verified.")
    return(False);

#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
  X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(4 * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

#takes a text file that contains encrypted block0, block1, true diff prob, real or random
#data samples are line separated, the above items whitespace-separated
#returns train data, ground truth, optimal ddt prediction
def readcsv(datei):
    data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s,16) for x in range(2)});
    X0 = [data[i][0] for i in range(len(data))];
    X1 = [data[i][1] for i in range(len(data))];
    Y = [data[i][3] for i in range(len(data))];
    Z = [data[i][2] for i in range(len(data))];
    ct0a = [X0[i] >> 16 for i in range(len(data))];
    ct1a = [X0[i] & MASK_VAL for i in range(len(data))];
    ct0b = [X1[i] >> 16 for i in range(len(data))];
    ct1b = [X1[i] & MASK_VAL for i in range(len(data))];
    ct0a = np.array(ct0a, dtype=np.uint16); ct1a = np.array(ct1a,dtype=np.uint16);
    ct0b = np.array(ct0b, dtype=np.uint16); ct1b = np.array(ct1b, dtype=np.uint16);
    
    #X = [[X0[i] >> 16, X0[i] & 0xffff, X1[i] >> 16, X1[i] & 0xffff] for i in range(len(data))];
    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b]); 
    Y = np.array(Y, dtype=np.uint8); Z = np.array(Z);
    return(X,Y,Z);

#baseline training data generator
# n = data size, nr = , diff = bit difference
# & is bitwise-AND, ^ is bitwise-XOR
def make_train_data(n, nr, diff=(0x0040,0)):
    # & means bitwise-AND, what happen here is random generate numbers e.g. 10010110, and AND with 1, so will get last bit random 1 1 0 0 1....
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  # create key
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  # ^ means bitwise-XOR
  # plain0l/plain0r
  # plain1l/plain1r
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  # count Y=0
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key(keys, nr);

  
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);

#real differences data generator
def real_differences_data(n, nr, diff=(0x0040,0)):
  #generate labels
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  #generate keys
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  #generate plaintexts
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  #apply input difference
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  #expand keys and encrypt
  ks = expand_key(keys, nr);
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  #generate blinding values
  k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  #apply blinding to the samples labelled as random
  ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0; ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1;
  ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0; ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1;
  #convert to input data for neural networks
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);

##########################################################################################






################################ NEW FUNCTIONS ################################################

def convert_to_binary_4plaintext(arr):
  """
  Pre-process the dataset by converting the numbers to binary and combine them into a single line of binary values (for 4-plaintext)
  Inputs: Array of dataset
  Returns: Processed dataset
  """
  if len(arr) == 0: raise Exception("No empty arrays")
  # (8x16, dataset_size)
  X = np.zeros((8 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  # for i in 128
  for i in range(8 * WORD_SIZE()):
    # index = 0, 1, 2, 3, 4, 5, 6, 7 every 16 bit
    index = i // WORD_SIZE();
    # count index of the array to retrieve
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    # X[i] --------------> 64
    # arr[index] >> offset & 1 => get specific index from the array

    # basically what it does is that
    # arr[0] --------------------> for every bit
    # arr[1] --------------------> append into X
    # arr[2] --------------------> accordingly for each bit in the order of the array
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

def make_4plaintext_train_data(n, nr, diff=(0x0040,0), diff2=(0,0)):
  """
  Generate training/validation/testing dataset for 4-plaintext scenario
  Inputs: Dataset size, Number of rounds, Bit difference 1, Bit difference 2
  Returns: Dataset X and Y
  """
  if nr <= 0: raise Exception("Round value can't be less than 0")
    # & means bitwise-AND, what happen here is random generate numbers e.g. 10010110, and AND with 1, so will get last bit random 1 1 0 0 1....
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  # create key
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);

  # ^ means bitwise-XOR
  # plain0l/plain0r -- plain1l/plain1r
  #       |                   |
  #       |                   |
  #       |                   |
  # plain2l/plain2r -- plain3l/plain3r

  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  plain2l = plain0l ^ diff2[0]; plain2r = plain0r ^ diff2[1];
  plain3l = plain2l ^ diff[0]; plain3r = plain2r ^ diff[1];

  # count Y=0
  num_rand_samples = np.sum(Y==0);
  # randomize data with Y==0
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain2l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain2r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain3l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain3r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);

  # generate key
  ks = expand_key(keys, nr);

  # encryption
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  ctdata2l, ctdata2r = encrypt((plain2l, plain2r), ks);
  ctdata3l, ctdata3r = encrypt((plain3l, plain3r), ks);
  
  # data pre-processing
  X = convert_to_binary_4plaintext([ctdata0l, ctdata0r, ctdata1l, ctdata1r, ctdata2l, ctdata2r, ctdata3l, ctdata3r]);
  return(X,Y);

# Display the model distinct outputs 
def check_model_outputs(file, num_rounds, bit_diff):
    """
    Check distinct outputs from a model prediction
    Inputs: Model file name, Number of rounds, Bit difference
    Returns: Prints a line of values counting by their occurence in the predictions
    """
    if num_rounds <= 0: raise Exception("Round value can't be less than 0")

    rnet = load_model(file)

    # make model predictions
    X,Y = make_train_data(10**6,num_rounds, diff=bit_diff)
    Z = rnet.predict(X,batch_size=10000).flatten();
    print("output array = ",Z)

    # count unique values
    unique, counts = np.unique(Z, return_counts=True)
    print(dict(zip(unique, counts)))
    print(' --------------------------------------------------------------------------------- ')

# Display the second-to-last layer of the model predictions
def evaluate_model_layers_output(file, num_rounds, bit_diff):
    """
    Visualize the second to last layer from the model (for 2-plaintext)
    Inputs: Model file name, Number of rounds, Bit difference
    Returns: 2 png formats of visualization based on the model layer outputs
    """
    
    if num_rounds <= 0: raise Exception("Round value can't be less than 0")
        
    # load model
    rnet = load_model(file)                       

    # generate test dataset
    X,Y = make_train_data(10**6,num_rounds, diff=bit_diff);
    c1_xor_c2(X,Y)

    # extract outputs from second to last model layer
    intermediate_layer_model = Model(inputs=rnet.input,
                                 outputs=[rnet.layers[-2].output, rnet.layers[-1].output])
    intermediate_output, final_output = intermediate_layer_model.predict(X)

    # flatten the outputs
    final_output = final_output.flatten()
    valid_bits = intermediate_output

    # average all values in bit positions throughout the dataset
    ret1 = np.sum(valid_bits[final_output>0.5], axis=0)/len(valid_bits[final_output>0.5])  
    ret0 = np.sum(valid_bits[final_output<=0.5], axis=0)/len(valid_bits[final_output<=0.5])

    # plot visualization for ciphertext 1
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = [i for i in range(32)]
    ax.bar(x,ret1[:32],color='tab:blue',width=0.25)
    ax.bar([i+0.25 for i in x],ret0[:32],color='tab:orange',width=0.25)
    plt.xticks(x)
    plt.xlabel("Bit Positions")
    plt.ylabel("Significance Values")
    plt.legend(["Y=1", "Y=0"])
    plt.title("Average Significance in Each Bit Position for Ciphertext 1")
    fig.savefig('model_layer_output_ciphertext1.png', dpi=100)

    # plot visualization for ciphertext 2
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = [i for i in range(32)]
    ax.bar(x,ret1[32:],color='tab:blue',width=0.25)
    ax.bar([i+0.25 for i in x],ret0[32:],color='tab:orange',width=0.25)
    plt.xticks(x)
    plt.xlabel("Bit Positions")
    plt.ylabel("Significance Values")
    plt.legend(["Y=1", "Y=0"])
    plt.title("Average Significance in Each Bit Position for Ciphertext 2")
    fig.savefig('model_layer_output_ciphertext2.png', dpi=100)

# Display the second-to-last layer of the model predictions
def evaluate_4_plaintext_model_layers_output(file, num_rounds, bit_diff):
    """
    Visualize the second to last layer from the model (for 4-plaintext)
    Inputs: Model file name, Number of rounds, Bit difference
    Returns: 4 png formats of visualization based on the model layer outputs
    """
    if num_rounds <= 0: raise Exception("Round value can't be less than 0")
    
    # load model
    rnet = load_model(file)                       

    # generate test dataset
    X,Y = make_4plaintext_train_data(10**6,rounds,diff2=bit_diff);
    c1_xor_c2_4plaintext(X,Y)

    # extract outputs from second to last model layer
    intermediate_layer_model = Model(inputs=rnet.input,
                                 outputs=[rnet.layers[-2].output, rnet.layers[-1].output])
    intermediate_output, final_output = intermediate_layer_model.predict(X)

    # flatten the outputs
    final_output = final_output.flatten()
    valid_bits = intermediate_output

    # average all values in bit positions throughout the dataset
    ret1 = np.sum(valid_bits[final_output>0.5], axis=0)/len(valid_bits[final_output>0.5])
    ret0 = np.sum(valid_bits[final_output<=0.5], axis=0)/len(valid_bits[final_output<=0.5])

    # plot visualization for ciphertext 1
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = [i for i in range(32)]
    ax.bar(x,ret1[:32],color='tab:blue',width=0.25)
    ax.bar([i+0.25 for i in x],ret0[:32],color='tab:orange',width=0.25)
    plt.xticks(x)
    plt.xlabel("Bit Positions")
    plt.ylabel("Significance Values")
    plt.legend(["Y=1", "Y=0"])
    plt.title("Average Significance in Each Bit Position for Ciphertext 1")
    fig.savefig('model_layer_output_ciphertext1.png', dpi=100)

    # plot visualization for ciphertext 2
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = [i for i in range(32)]
    ax.bar(x,ret1[32:64],color='tab:blue',width=0.25)
    ax.bar([i+0.25 for i in x],ret0[32:64],color='tab:orange',width=0.25)
    plt.xticks(x)
    plt.xlabel("Bit Positions")
    plt.ylabel("Significance Values")
    plt.legend(["Y=1", "Y=0"])
    plt.title("Average Significance in Each Bit Position for Ciphertext 2")
    fig.savefig('model_layer_output_ciphertext2.png', dpi=100)

    # plot visualization for ciphertext 3
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = [i for i in range(32)]
    ax.bar(x,ret1[64:96],color='tab:blue',width=0.25)
    ax.bar([i+0.25 for i in x],ret0[64:96],color='tab:orange',width=0.25)
    plt.xticks(x)
    plt.xlabel("Bit Positions")
    plt.ylabel("Significance Values")
    plt.legend(["Y=1", "Y=0"])
    plt.title("Average Significance in Each Bit Position for Ciphertext 3")
    fig.savefig('model_layer_output_ciphertext3.png', dpi=100)

    # plot visualization for ciphertext 4
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = [i for i in range(32)]
    ax.bar(x,ret1[96:],color='tab:blue',width=0.25)
    ax.bar([i+0.25 for i in x],ret0[96:],color='tab:orange',width=0.25)
    plt.xticks(x)
    plt.xlabel("Bit Positions")
    plt.ylabel("Significance Values")
    plt.legend(["Y=1", "Y=0"])
    plt.title("Average Significance in Each Bit Position for Ciphertext 4")
    fig.savefig('model_layer_output_ciphertext4.png', dpi=100)
    

# c1 xor c2 test
def c1_xor_c2(X,Y):
    """
    Conduct c1-xor-c2 analysis on the training dataset (for 2-plaintext)
    Inputs: Dataset of X and Y
    Returns: 1 png format of visualization based on the c1-xor-c2 analysis
    """
    if len(X)==0 or len(Y)==0: raise Exception("No empty arrays")
    if len(X[0]) != 64: raise Exception("Not training dataset from 2 plaintext")

    True_X = X[Y==1]
    Random_X = X[Y==0]

    data = []
    # Y=1 and Y=0
    for i in (True_X, Random_X):
        left = i[:, :32]
        right = i[:, 32:]
        diff = left ^ right

        # Average occurences
        ret = np.sum(diff, axis=0)/len(diff)
        data.append(ret)

    # plot visualization for c1_xor_c2
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = [i for i in range(32)]
    ax.bar(x,data[0],color='tab:blue',width=0.25)
    ax.bar([i+0.25 for i in x],data[1],color='tab:orange',width=0.25)
    plt.xticks(x)
    plt.xlabel("Bit Positions")
    plt.ylabel("Average Occurence of Differences in Bit Position")
    plt.legend(["Y=1", "Y=0"], loc='lower right')
    plt.title("C1 xor C2")
    fig.savefig('c1_xor_c2.png', dpi=100)
    
# c1 xor c2 test
def c1_xor_c2_4plaintext(X,Y):
    """
    Conduct c1-xor-c2 analysis on the training dataset (for 4-plaintext)
    Inputs: Dataset of X and Y
    Returns: 3 png format of visualization based on the c1-xor-c2 analysis
    """
    if len(X)==0 or len(Y)==0: raise Exception("No empty arrays")
    if len(X[0]) != 128: raise Exception("Not training dataset from 4 plaintext")

    True_X = X[Y==1]
    Random_X = X[Y==0]

    data = []
    # Y=1 and Y=0
    for i in (True_X, Random_X):

        
        # Average occurences between c1 and c2
        left = i[:, :32]
        right = i[:, 32:64]
        diff = left ^ right

        ret = np.sum(diff, axis=0)/len(diff)
        data.append(ret)
        
        # Average occurences between c1 and c3
        left = i[:, :32]
        right = i[:, 64:96]
        diff = left ^ right

        ret = np.sum(diff, axis=0)/len(diff)
        data.append(ret)
        
        # Average occurences between c1 and c4
        left = i[:, :32]
        right = i[:, 96:]
        diff = left ^ right

        ret = np.sum(diff, axis=0)/len(diff)
        data.append(ret)

    # plot visualization for c1_xor_c2
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = [i for i in range(32)]
    ax.bar(x,data[0],color='tab:blue',width=0.25)
    ax.bar([i+0.25 for i in x],data[3],color='tab:orange',width=0.25)
    plt.xticks(x)
    plt.xlabel("Bit Positions")
    plt.ylabel("Average Occurence of Differences in Bit Position")
    plt.legend(["Y=1", "Y=0"], loc='lower right')
    plt.title("C1 xor C2 for Bit Difference 1")
    fig.savefig('c1_xor_c2.png', dpi=100)

    # plot visualization for c1_xor_c3
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = [i for i in range(32)]
    ax.bar(x,data[1],color='tab:blue',width=0.25)
    ax.bar([i+0.25 for i in x],data[4],color='tab:orange',width=0.25)
    plt.xticks(x)
    plt.xlabel("Bit Positions")
    plt.ylabel("Average Occurence of Differences in Bit Position")
    plt.legend(["Y=1", "Y=0"], loc='lower right')
    plt.title("C1 xor C3 for Bit Difference 2")
    fig.savefig('c1_xor_c3.png', dpi=100)

    # plot visualization for c1_xor_c4
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = [i for i in range(32)]
    ax.bar(x,data[2],color='tab:blue',width=0.25)
    ax.bar([i+0.25 for i in x],data[5],color='tab:orange',width=0.25)
    plt.xticks(x)
    plt.xlabel("Bit Positions")
    plt.ylabel("Average Occurence of Differences in Bit Position")
    plt.legend(["Y=1", "Y=0"], loc='lower right')
    plt.title("C1 xor C4 for Both Bit Differences")
    fig.savefig('c1_xor_c4.png', dpi=100)

    
# Normal Evaluate
def evaluate(file,rounds,bit_diff):
    """
    Evaluate model (accuracy, TPR, TNR, MSE) for 2-plaintext
    Inputs: Model file name, Number of rounds, Bit difference
    Returns: Prints the performances of the model
    """
    if rounds <= 0: raise Exception("Round value can't be less than 0")
    # load model
    net = load_model(file)
    # generate test dataset
    X,Y = make_train_data(10**6,rounds,diff=bit_diff);
    # make model predictions
    Z = net.predict(X,batch_size=10000).flatten();
    # only >0.5 counted as Y=1
    Zbin = (Z > 0.5);
    # calculate mse
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    # calculate acc/tpr/tnr
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", str(acc), "TPR: ", str(tpr), "TNR: ", str(tnr), "MSE:", str(mse));
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);


# 4-plaintext evaluate
def evaluate_4_plaintext(file, rounds, bit_diff):
    """
    Evaluate model (accuracy, TPR, TNR, MSE) for 4-plaintext
    Inputs: Model file name, Number of rounds, Bit difference
    Returns: Prints the performances of the model
    """
    if rounds <= 0: raise Exception("Round value can't be less than 0")
    # load model
    net = load_model(file)
    # generate test dataset
    X,Y = make_4plaintext_train_data(10**6,rounds,diff2=bit_diff);
    # make model predictions
    Z = net.predict(X,batch_size=10000).flatten();
    # only >0.5 counted as Y=1
    Zbin = (Z > 0.5);
    # calculate mse
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    # calculate acc/tpr/tnr
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", str(acc), "TPR: ", str(tpr), "TNR: ", str(tnr), "MSE:", str(mse));
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);



    
#evaluate_model_layers_output(file, rounds, bit_diff)
evaluate_4_plaintext_model_layers_output(file, rounds, bit_diff)

