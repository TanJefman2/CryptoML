from os import urandom
import numpy as np
from keras.models import model_from_json, load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from pickle import dump

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2

import pathlib
import csv
import os


##### INPUTS #####

bit_differences = [(0x0040,0)]   # List of bit differences
main_wdir='./kaggle/working/'    # Custom directory
wdir = main_wdir+'test_models/'  # Directory to save models
num_rounds = 5

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
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key(keys, nr);

  # why 01 and 0r?
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

################################# GOHR'S TRAIN_NETS.PY ###########################################
# CODE RETRIEVED FROM: https://github.com/agohr/deep_speck/blob/master/speck.py

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

#make residual tower of convolutional blocks
def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num_blocks * word_size * 2,));
  rs = Reshape((2 * num_blocks, word_size))(inp);
  perm = Permute((2,1))(rs);
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  conv0 = BatchNormalization()(conv0);
  conv0 = Activation('relu')(conv0);
  #add residual blocks
  shortcut = conv0;
  for i in range(depth):
    conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
    conv1 = BatchNormalization()(conv1);
    conv1 = Activation('relu')(conv1);
    conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
    conv2 = BatchNormalization()(conv2);
    conv2 = Activation('relu')(conv2);
    shortcut = Add()([shortcut, conv2]);
  #add prediction head
  flat1 = Flatten()(shortcut);
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1);
  dense1 = BatchNormalization()(dense1);
  dense1 = Activation('relu')(dense1);
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
  dense2 = BatchNormalization()(dense2);
  dense2 = Activation('relu')(dense2);
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
  model = Model(inputs=inp, outputs=out);
  return(model);

# Make checkpoint
def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);

##########################################################################################

######################################## NEW FUNCTIONS ############################################


# Model file name
def get_model_file_name(num_rounds, bit):
  return 'round'+str(num_rounds)+'bit'+str(bit)+'.h5'

def train_speck_distinguisher(bit_diff, num_epochs, num_rounds, depth=1, initial_epoch=0):

    bs = 5000;
    #generate training and validation data
    X, Y = make_train_data(10**7,num_rounds, bit_diff);
    X_eval, Y_eval = make_train_data(10**6, num_rounds, bit_diff);
    
    #set up model checkpoint
    fname = get_model_file_name(num_rounds,bit_diff)
    logdir = pathlib.Path(main_wdir+"logs/history.csv")

    # If model exists, load the model, else create new model
    try:
      net = load_model(wdir+fname);
    except:
      print("Creating new model")
      initial_epoch = 0
      #create the network
      net = make_resnet(depth=depth, reg_param=10**-5);
      net.compile(optimizer='adam',loss='mse',metrics=['acc']);
    check = make_checkpoint(wdir+fname);

    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));

    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), initial_epoch=initial_epoch, callbacks=[lr,check,CSVLogger(logdir , append=True)]);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    dump(h.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));

    return(net, np.max(h.history['val_acc']));

def evaluate(f, rnet, bit_diff, num_rounds):

    def eval(net,X,Y):
        Z = net.predict(X,batch_size=10000).flatten();
        Zbin = (Z > 0.5);
        diff = Y - Z; mse = np.mean(diff*diff);
        n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
        acc = np.sum(Zbin == Y) / n;
        tpr = np.sum(Zbin[Y==1]) / n1;
        tnr = np.sum(Zbin[Y==0] == 0) / n0;
        mreal = np.median(Z[Y==1]);
        high_random = np.sum(Z[Y==0] > mreal) / n0;
        f.writelines("Accuracy: "+ str(acc) + " TPR: " + str(tpr) + " TNR: "+ str(tnr) + " MSE:" + str(mse) +"\n");
        f.writelines("Percentage of random pairs with score higher than median of real pairs: " + str(100*high_random) +"\n\n");

    f.writelines("\nBit Difference: "+str(bit_diff)+"\n")

    X,Y = make_train_data(10**6,num_rounds,diff=bit_diff);
    X5r, Y5r = real_differences_data(10**6,num_rounds,diff=bit_diff);

    f.writelines('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting'+"\n");
    eval(rnet, X, Y);

    f.writelines('Testing real differences setting now.'+"\n");
    eval(rnet, X5r, Y5r);

# Read history.csv file to get latest epoch
def read_log():
  with pathlib.Path(main_wdir+"logs/history.csv").open() as fp:
    data = list(csv.DictReader(fp)) 
    ret = int(data[-1]['epoch']) if len(data)>0 else 0
    return ret

def run(num_rounds):

    # Log file name
    log_file = main_wdir+'analysis_round'+str(num_rounds)+'.txt'

    # Create log file if not exist
    pathlib.Path(main_wdir+"logs").mkdir(parents=True, exist_ok=True)
    pathlib.Path(main_wdir+"logs/history.csv").touch(exist_ok=True)
    pathlib.Path(log_file).touch(exist_ok=True)

    # Checks
    train_check, eval_check, done_check = check_progress(log_file)
    start = 0
    model_file = None
    initial_epoch = 0

    if done_check is not None:
      print("Already done running all test cases. Please remove all content in the text file if you want to re-run")
      return 

    elif train_check is not None:
      start = train_check
      last_epoch = read_log()+1
      if last_epoch == 200: 
        eval_check = start
      initial_epoch = last_epoch
      
    if eval_check is not None:
      start = eval_check
      model_file = get_model_file_name(num_rounds, bit_differences[start])

    with open(log_file, 'a') as f:
        
        f.writelines("+++++++++++++++++++++++++++++++++++++\n")

        for i in range(start, len(bit_differences)):
        
            f.writelines(">"+str(i)+"\n")

            if eval_check is None:
              net, val_acc = train_speck_distinguisher(bit_differences[i], 200, num_rounds, depth=10, initial_epoch=initial_epoch)
              f.writelines("=Best validation accuracy: "+str(val_acc)+"\n")

              os.remove(main_wdir+"logs/history.csv")
              pathlib.Path(main_wdir+"logs/history.csv").touch(exist_ok=True)
            else:
              net = load_model(wdir+model_file)

            evaluate(f, net, bit_differences[i],num_rounds)
            eval_check = None
            initial_epoch = 0
        
        f.writelines("-------------------------------------")

# Check log file progress
def check_progress(filename):

  eval_check = None
  train_check = None
  done_check = None

  with open(filename) as f:
    symbol = "/"
    index = -1
    for line in f:
      if line[0] == ">": index = int(line[1:])
      if line[0] in [">","=","-"]: symbol = line[0]

    if symbol == ">":
      train_check = index
      
    elif symbol == "=":
      eval_check = index

    elif symbol == "-":
      done_check = True
  
    return train_check, eval_check, done_check
    
    
run(num_rounds=num_rounds)

