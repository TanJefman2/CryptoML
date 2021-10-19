from os import urandom
import numpy as np
from keras.models import model_from_json, load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from pickle import dump

from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

from keras import backend as K
from keras.models import Model


import pathlib
import csv
import os
import csv

##### INPUTS #####

bit_diff = (0x0020, 0)  # Bit difference
file = '2-plaintext_models/round6/round6bit(32, 0).h5'  # Model file location
rounds = 6

##################



################################ GOHR'S SPECK.PY ################################################

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

# Display the second-to-last layer of the model predictions
def evaluate_model_layers_output(file, num_rounds, bit_diff):

    
    if num_rounds <= 0: raise Exception("Round value can't be less than 0")
        
    rnet = load_model(file)                       

    X,Y = make_train_data(10**6,num_rounds, diff=bit_diff);
    c1_xor_c2(X,Y)

    intermediate_layer_model = Model(inputs=rnet.input,
                                 outputs=[rnet.layers[-2].output, rnet.layers[-1].output])
    intermediate_output, final_output = intermediate_layer_model.predict(X)

    final_output = final_output.flatten()
    valid_bits = intermediate_output
    # valid_bits[intermediate_output>0] = 1

    ret = np.sum(valid_bits[final_output>0.5], axis=0)/len(valid_bits[final_output>0.5])
    print("Y=1 model layers outputs: ")
    print(ret[:32].tolist())
    print(ret[32:].tolist())
    print(' --------------------------------------------------------------------------------- ')
    
    ret = np.sum(valid_bits[final_output<=0.5], axis=0)/len(valid_bits[final_output<=0.5])
    print("Y=0 model layers outputs: ")
    print(ret[:32].tolist())
    print(ret[32:].tolist())
    print(' --------------------------------------------------------------------------------- ')

# Display the model distinct outputs 
def check_model_outputs(file, num_rounds, bit_diff):

    if num_rounds <= 0: raise Exception("Round value can't be less than 0")

    rnet = load_model(file)

    X,Y = make_train_data(10**6,num_rounds, diff=bit_diff)
    Z = rnet.predict(X,batch_size=10000).flatten();
    print("output array = ",Z)

    unique, counts = np.unique(Z, return_counts=True)
    print(dict(zip(unique, counts)))
    print(' --------------------------------------------------------------------------------- ')

# c1 xor c2 test
def c1_xor_c2(X,Y):
    if len(X)==0 or len(Y)==0: raise Exception("No empty arrays")
    if len(X[0]) != 64: raise Exception("Not training dataset from 2 plaintext")

    True_X = X[Y==1]
    Random_X = X[Y==0]

    print("c1_xor_c2 test: ")

    for i in (True_X, Random_X):
        left = i[:, :32]
        right = i[:, 32:]
        diff = left ^ right

        ret = np.sum(diff, axis=0)/len(diff)
        print(ret.tolist())
        print(' --------------------------------------------------------------------------------- ')

# c1 xor c2 test
def c1_xor_c2_4plaintext(X,Y):
    if len(X)==0 or len(Y)==0: raise Exception("No empty arrays")
    if len(X[0]) != 128: raise Exception("Not training dataset from 4 plaintext")

    True_X = X[Y==1]
    Random_X = X[Y==0]

    print("c1_xor_c2 test: ")

    for i in (True_X, Random_X):

        print("bit difference 1")
        left = i[:, :32]
        right = i[:, 32:64]
        diff = left ^ right

        ret = np.sum(diff, axis=0)/len(diff)
        print(ret.tolist())
        print(' --------------------------------------------------------------------------------- ')

        print("bit difference 2")
        left = i[:, :32]
        right = i[:, 64:96]
        diff = left ^ right

        ret = np.sum(diff, axis=0)/len(diff)
        print(ret.tolist())
        print(' --------------------------------------------------------------------------------- ')

        print("diagonal")
        left = i[:, :32]
        right = i[:, 96:]
        diff = left ^ right

        ret = np.sum(diff, axis=0)/len(diff)
        print(ret.tolist())
        print(' --------------------------------------------------------------------------------- ')

# Normal Evaluate
def evaluate(file,rounds,bit_diff):

    if rounds <= 0: raise Exception("Round value can't be less than 0")

    net = load_model(file)
    X,Y = make_train_data(10**6,rounds,diff=bit_diff);
    Z = net.predict(X,batch_size=10000).flatten();
    Zbin = (Z > 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", str(acc), "TPR: ", str(tpr), "TNR: ", str(tnr), "MSE:", str(mse));
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);

def convert_to_binary_4plaintext(arr):

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

  ks = expand_key(keys, nr);

  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  ctdata2l, ctdata2r = encrypt((plain2l, plain2r), ks);
  ctdata3l, ctdata3r = encrypt((plain3l, plain3r), ks);
  
  X = convert_to_binary_4plaintext([ctdata0l, ctdata0r, ctdata1l, ctdata1r, ctdata2l, ctdata2r, ctdata3l, ctdata3r]);
  return(X,Y);

# 4-plaintext evaluate
def evaluate_4_plaintext(file, rounds, bit_diff):

    if rounds <= 0: raise Exception("Round value can't be less than 0")

    net = load_model(file)
    X,Y = make_4plaintext_train_data(10**6,rounds,diff2=bit_diff);
    Z = net.predict(X,batch_size=10000).flatten();
    Zbin = (Z > 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", str(acc), "TPR: ", str(tpr), "TNR: ", str(tnr), "MSE:", str(mse));
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);

# Display the second-to-last layer of the model predictions
def evaluate_4_plaintext_model_layers_output(file, num_rounds, bit_diff):
    
    if num_rounds <= 0: raise Exception("Round value can't be less than 0")
    
    rnet = load_model(file)                       

    X,Y = make_4plaintext_train_data(10**6,rounds,diff2=bit_diff);
    c1_xor_c2_4plaintext(X,Y)

    intermediate_layer_model = Model(inputs=rnet.input,
                                 outputs=[rnet.layers[-2].output, rnet.layers[-1].output])
    intermediate_output, final_output = intermediate_layer_model.predict(X)

    final_output = final_output.flatten()
    valid_bits = intermediate_output

    ret = np.sum(valid_bits[final_output>0.5], axis=0)/len(valid_bits[final_output>0.5])
    print("Y=1 model layers outputs: ")
    print(ret[:32].tolist())
    print(ret[32:64].tolist())
    print(ret[64:96].tolist())
    print(ret[96:].tolist())
    print(' --------------------------------------------------------------------------------- ')
    
    ret = np.sum(valid_bits[final_output<=0.5], axis=0)/len(valid_bits[final_output<=0.5])
    print("Y=0 model layers outputs: ")
    print(ret[:32].tolist())
    print(ret[32:64].tolist())
    print(ret[64:96].tolist())
    print(ret[96:].tolist())
    print(' --------------------------------------------------------------------------------- ')

evaluate_model_layers_output(file, rounds, bit_diff)
#evaluate_4_plaintext_model_layers_output(file, rounds, bit_diff)

