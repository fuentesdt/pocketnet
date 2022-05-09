#!/usr/bin/env python
# coding: utf-8

# # Model Saturation Testing
# 
# Train BraTS and COVIDx models with successively fewer data points to test for model saturation for PocketNet.

# In[ ]:


import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import subprocess
import os

##### Tensorflow #####
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
import os

# Set this environment variable to allow ModelCheckpoint to work
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Set this environment variable to only use the first available GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# For tensorflow 2.x.x allow memory growth on GPU
###################################
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
###################################

# Use this to allow memory growth on TensorFlow v1.x.x
# ###################################
# config = tf.ConfigProto()
 
# # Don't pre-allocate memory; allocate as-needed
# config.gpu_options.allow_growth = True
 
# # Only allow a specified percent of the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 0.75
 
# # Create a session with the above options specified.
# K.tensorflow_backend.set_session(tf.Session(config = config))
# ##################################


# ### L2 Dice Loss
# 
# Dice loss for BraTS models.

# In[ ]:


# L2 Dice loss
def dice_loss_l2(y_true, y_pred):
    smooth = 0.0000001
    
    # (batch size, depth, height, width, channels)
    if len(y_true.shape) == 5:
        num = K.sum(K.square(y_true - y_pred), axis = (1,2,3))
        den = K.sum(K.square(y_true), axis = (1,2,3)) + K.sum(K.square(y_pred), axis = (1,2,3)) + smooth
        
    # (batch size, height, width, channels)
    elif len(y_true.shape) == 4:
        num = K.sum(K.square(y_true - y_pred), axis = (1,2))
        den = K.sum(K.square(y_true), axis = (1,2)) + K.sum(K.square(y_pred), axis = (1,2)) + smooth
        
    return K.mean(num/den, axis = -1)


# ### Architecture Implementation

# In[ ]:


def PocketNet(inputShape, 
              numClasses, 
              mode, 
              net, 
              pocket, 
              initFilters, 
              depth):
    
    '''
    PocketNet - Smaller CNN for medical image segmentation
    
    Inputs:
    inputShape   : Size of network input - (depth, height, width, channels) for 3D
                   (height, width, channels) for 2D
    numClasses   : Number of output classes
    mode         : 'seg' or 'class' for segmenation or classification network
    net          : 'unet', 'resnet', or 'densenet' for U-Net, ResNet or DenseNet blocks
    pocket       : True/False for pocket architectures
    initFilters  : Number of starting filters at input level
    depth        : Number of max-pooling layers
    
    Outputs:
    model        : Keras model for training/predicting
    
    Author: Adrian Celaya
    Last modified: 4.20.2021
    '''
    
    # 3D inputs are (depth, height, width, channels)
    if len(inputShape) == 4:
        dim = '3d'
    # 2D inputs are (height, width, channels)
    elif len(inputShape) == 3:
        dim = '2d'
    
    # Convolution block operator
    def Block(x, filters, params, net, dim):
        ### DenseNet block ###
        if net == 'densenet':
            for _ in range(2):
                if dim == '3d':
                    y = layers.Conv3D(filters, **params[0])(x)
                elif dim == '2d':
                    y = layers.Conv2D(filters, **params[0])(x)
                x = layers.concatenate([x, y])
                
            if dim == '3d':
                x = layers.Conv3D(filters, **params[1])(x)
            elif dim == '2d':
                x = layers.Conv2D(filters, **params[1])(x)
        
        ### ResNet block ###
        elif net == 'resnet':
            if dim == '3d':
                y = layers.Conv3D(filters, **params[0])(x)
                y = layers.Conv3D(filters, **params[0])(y)
            elif dim == '2d':
                y = layers.Conv2D(filters, **params[0])(x)
                y = layers.Conv2D(filters, **params[0])(y)
                
            x = layers.concatenate([x, y])
            
            if dim == '3d':
                x = layers.Conv3D(filters, **params[1])(x)
            elif dim == '2d':
                x = layers.Conv2D(filters, **params[1])(x)
        
        ### U-Net block ###
        elif net == 'unet':
            if dim == '3d':
                x = layers.Conv3D(filters, **params[0])(x)
                x = layers.Conv3D(filters, **params[0])(x)
            elif dim == '2d':
                x = layers.Conv2D(filters, **params[0])(x)
                x = layers.Conv2D(filters, **params[0])(x)
                
        return x

    # Downsampling block - Convolution + maxpooling
    def TransitionDown(x, filters, params, net, dim):
        skip = Block(x, filters, params, net, dim)
        
        if dim == '3d':
            x = layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(skip)
        elif dim == '2d':
            x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(skip)
            
        return skip, x

    # Upsampling block - Transposed convolution + concatenation + convolution
    def TransitionUp(x, skip, filters, params, net, dim):
        
        if dim == '3d':
            x = layers.Conv3DTranspose(filters, **params[2])(x)
        elif dim == '2d':
            x = layers.Conv2DTranspose(filters, **params[2])(x)
            
        x = layers.concatenate([x, skip])
        x = Block(x, filters, params, net, dim)
        return x
    
    # Parameters for each convolution operation
    params = list()
    if dim == '3d':
        params.append(dict(kernel_size = (3, 3, 3), activation = 'relu', padding = 'same'))
        params.append(dict(kernel_size = (1, 1, 1), activation = 'relu', padding = 'same'))
        params.append(dict(kernel_size = (1, 2, 2), strides = (1, 2, 2), padding = 'same'))
    elif dim == '2d':
        params.append(dict(kernel_size = (3, 3), activation = 'relu', padding = 'same'))
        params.append(dict(kernel_size = (1, 1), activation = 'relu', padding = 'same'))
        params.append(dict(kernel_size = (2, 2), strides = (2, 2), padding = 'same'))

        
    # Keep filters constant for PocketNet
    if pocket:
        filters = [initFilters for i in range(depth + 1)]
    else:
        filters = [initFilters * 2 ** (i) for i in range(depth + 1)]
    
    # Input to network
    inputs = layers.Input(inputShape)
 
    # Encoder path
    x = inputs
    skips = list()
    for i in range(depth):
        skip, x = TransitionDown(x, filters[i], params, net, dim)
        skips.append(skip)
        
    # Bottleneck
    x = Block(x, filters[-1], params, net, dim)

    # Apply global max-pooling to output of bottleneck if classification
    if mode == 'class':
        if dim == '3d':
           x = layers.GlobalMaxPooling3D()(x)
        elif dim == '2d':
           x = layers.GlobalMaxPooling2D()(x)
        output = layers.Dense(numClasses, activation = 'softmax')(x)

    
    # Continue with decoder path if segmentation
    elif mode == 'seg':
        
        for i in range(depth - 1, -1, -1):
            x = TransitionUp(x, skips[i], filters[i], params, net, dim)
            
        if dim == '3d':
            output = layers.Conv3D(numClasses, (1, 1, 1), activation = 'softmax')(x)
        elif dim == '2d':
            output = layers.Conv2D(numClasses, (1, 1), activation = 'softmax')(x)
            
    model = Model(inputs = [inputs], outputs = [output])
    return model


# ### Data Generator 
# 
# Stream data for BraTS models from disk to model while training.

# In[ ]:


class data_generator(keras.utils.Sequence):
    def __init__(self, dataframe, batch_size = 1, dim = (240, 240, 5), n_channels = 4, n_classes = 2, shuffle = True):
        self.dim = dim
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac = 1).reset_index(drop = True)
        
    def __data_generation(self, index):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,self.n_classes))

        for i in range(index, index + self.batch_size):
            X[i - index] = np.load(self.dataframe.iloc[i]['image'])
            y[i - index] = self.dataframe.iloc[i]['truthid']
        return X, y


# ### Inference
# 
# Create predictions on BraTS images after training each model.

# In[ ]:



def inference_class(model, df):
    preds = list()
    for i in trange(len(df)):
        img = np.load(df.iloc[i]['image'])
        
        # Apply z-score normalization
        #img = img.reshape((1, *img.shape))
        pred = model.predict(img[np.newaxis,:,:,:,:])
        pred = pred[0][-1]
        preds.append(pred)
    return preds


# ### Run data saturation tests
# 
# train model with kfold

# In[ ]:


# In[ ]:

def GetSetupKfolds(numfolds,idfold,dataidsfull ):
  from sklearn.model_selection import KFold

  if (numfolds < idfold or numfolds < 1):
     raise("data input error")
  # split in folds
  if (numfolds > 1):
     kf = KFold(n_splits=numfolds,random_state=1,shuffle=True)
     allkfolds = [ (list(map(lambda iii: dataidsfull[iii], train_index)), list(map(lambda iii: dataidsfull[iii], test_index))) for train_index, test_index in kf.split(dataidsfull )]
     train_index = allkfolds[idfold][0]
     test_index  = allkfolds[idfold][1]
  else:
     train_index = np.array(dataidsfull )
     test_index  = None  
  return (train_index,test_index)

def run_saturation_pdac(pocket):
    
    # Load main dataframe with images and targets
    # Used clean version of COVIDx dataset. See preprocess.ipynb.
    fulldata = pd.read_csv('dicom/wide_slices_paths.csv')
    pats = np.unique(fulldata['id'])
    nfold = 5
    #for idfold in range(nfold):
    for idfold in [0]:
       (train_validation_index,test_index) = GetSetupKfolds(nfold ,idfold,pats )
       trainPats, valPats, _, _ = train_test_split(train_validation_index, train_validation_index, test_size = 0.10, random_state = 0)

       # get subsets
       val=fulldata[fulldata['id'].isin(valPats)]
       train=fulldata[fulldata['id'].isin(trainPats)]
       test=fulldata[fulldata['id'].isin(test_index)]
   
       # error check
       if( len(np.unique(train['target'])) < 2):
         print("training set error")
         raise RuntimeError
           
       ## # Use COVIDx test set and scale up the size of each training set
       ## train, val, _, _ = train_test_split(train, train['target'], test_size = 0.05, random_state = 0)
       ## train = train.reset_index(drop = True)
       ## val = val.reset_index(drop = True)
       ## val_imbalance = 1 - np.sum(val['target'].map(int)) / len(val)
       ## print('Val class imbalance = ' + str(val_imbalance))
       
       # Define batchsize for models
       batchSize = 4
       
       # Save predictions here
       # TODO 
       preds = test[['image', 'truthid']]
           
       net = 'unet'

       # Create training and validation generators 
       trainGenerator = data_generator(train, batchSize,dim = (96, 256, 256),n_channels=1,n_classes=2)
       validationGenerator = data_generator(val, batchSize,dim = (96, 256, 256),n_channels=1,n_classes=2)
       
       # Create and compile model
       model = PocketNet((96,256, 256, 1), 2, 'class', net , pocket, 16, 1)
       model.summary()
       #model = PocketNet((96,256, 256, 2), 2, 'class', net , pocket, 16, 4)
       myoptim = tf.keras.optimizers.Adadelta()

       model.compile(optimizer = myoptim , loss = 'binary_crossentropy', metrics = ['binary_accuracy', tf.keras.metrics.AUC()])

       # Define callbacks
       # Reduce learning rate when learning stalls
       reduceLr = ReduceLROnPlateau(monitor = 'val_loss', 
                                    mode = 'min',
                                    factor = 0.5, 
                                    patience = 7, 
                                    min_lr = 0.0000001, 
                                    verbose = 1)

       # Save best model based on validation accuracy
       # Name convention: (architecture)_(full/pocket)_(% of training data used).h5 -> unet_pocket_20.h5
       if pocket:
           modelName = 'models/' + net + '_pocket_' + str(idfold ) + '.h5'
       else:
           modelName = 'models/' + net + '_full_' + str(idfold ) + '.h5'
       
       saveBestModel = ModelCheckpoint(filepath = modelName, 
                                       monitor = 'val_loss', 
                                       mode = 'min',
                                       verbose = 1, 
                                       save_best_only = True)
       
       # tensorboard callbacks
       from tensorflow.keras.callbacks import TensorBoard
       os.system('mkdir -p log/%d/' % idfold)
       tensorboard = TensorBoard(log_dir='./log/%d/' % idfold, histogram_freq=0, write_graph=True, write_images=False)

       # Fit model
       model.fit(trainGenerator , 
                 epochs = 20,
                 steps_per_epoch = (len(train)) // batchSize,
                 validation_data = validationGenerator ,
                 validation_steps = (len(val)) // batchSize,
                 callbacks = [tensorboard, saveBestModel]
                 #callbacks = [tensorboard,reduceLr, saveBestModel]
                 #use_multiprocessing = True, 
                # workers = 8
                 )
       
       # Load best model for prediction
       model = load_model(modelName)
       preds['predictions'] = np.array(inference_class(model, test))
       
       # For each network architecture, write scaling results to csv file
       if pocket:
           csvFile = 'preds_' + net + '_pocket'+str(idfold)+'.csv'
       else:
           csvFile = 'preds_' + net + '_full'+str(idfold)+'.csv'
   
       preds.to_csv(csvFile, index = False)

    ### END OF FUNCTION ###


# In[ ]:


run_saturation_pdac(pocket = True)

