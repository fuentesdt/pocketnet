#!/usr/bin/env python
# coding: utf-8

# # Preprocessing for BraTS, NFBS, and COVIDx8B datasets
# 
# #### Change file paths in this notebook to match your system!
# 
# Preprocessing steps taken:
# 
# BraTS and NFBS: Load images with SimpleITK -> z-score intensity normalization -> break into patches
# 
# COVIDx8B: Clean up file names and unzip compressed images

# In[ ]:


import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import trange
import os
import subprocess


# ### Helper functions

# In[ ]:


# Create binary masks for BraTS dataset
def binarize_brats(base_dir):
    def get_files(path):
        files_list = list()
        for root, _, files in os.walk(path, topdown = False):
            for name in files:
                files_list.append(os.path.join(root, name))
        return files_list
    
    files_list = get_files(base_dir)
    for file in files_list:
        if 'seg' in file:
            binary_mask_name = file[0:-7] + '_binary.nii.gz'
            binarize_cmd = ['c3d', file, '-binarize', '-o', binary_mask_name]
            subprocess.call(binarize_cmd)
            
    ################# End of function #################
    

# Create a CSV with file paths for a dataset
def get_paths_csv(base_dir, name_dict, output_csv):
    def get_files(path):
        files_list = list()
        for root, _, files in os.walk(path, topdown = False):
            for name in files:
                files_list.append(os.path.join(root, name))
        return files_list

    cols = ['id'] + list(name_dict.keys())
    df = pd.DataFrame(columns = cols)
    row_dict = dict.fromkeys(cols)

    ids = os.listdir(base_dir)

    for i in ids:
        row_dict['id'] = i
        path = os.path.join(base_dir, i)
        files = get_files(path)

        for file in files:
            for img_type in name_dict.keys():
                for img_string in name_dict[img_type]:
                    if img_string in file:
                        row_dict[img_type] = file

        df = df.append(row_dict, ignore_index = True)

    df.to_csv(output_csv, index = False)
    
    ################# End of function #################
    
# Read a nifti file from a given path and return it as a 3D numpy array
def ReadImagesSITK(images_list, dims):
    
    # Read image, normalize, and get numpy array
    def GetArray(path):
        arr = sitk.ReadImage(path)
        arr = sitk.Normalize(arr)
        arr = sitk.GetArrayFromImage(arr)
        return arr
    
    image = np.empty((*dims, len(images_list)))
    for i in range(len(images_list)):
        image[..., i] = GetArray(images_list[i])

    return image

# Read a segmentation mask from a given path and return one hot representation of mask
def ReadMaskSITK(path, classes):
    num_classes = len(classes)
    mask = sitk.ReadImage(path)
    mask = sitk.GetArrayFromImage(mask)
    mask_onehot = np.empty((*mask.shape, num_classes))
    for i in range(num_classes):
        mask_onehot[..., i] = mask == classes[i]
    return mask_onehot

# Write slices of data from csv
def write_slices(input_csv, image_dest, mask_dest, output_csv, image_dims):
    input_df = pd.read_csv(input_csv)
    num_pats = len(input_df)
    slice_thickness = 5
    
    output_cols = ['id', 'image', 'mask']
    output_df = pd.DataFrame(columns = output_cols)
    
    for i in trange(num_pats):
        
        # Get row of input dataframe
        current_pat = input_df.iloc[i].to_dict()
        
        # Read in images and masks
        images_list = list(current_pat.values())[2:len(current_pat)]
        img = ReadImagesSITK(images_list, dims = image_dims)
        mask_binary = ReadMaskSITK(current_pat['mask'], classes = [0, 1])
        img_depth = image_dims[0]

        for k in range(img_depth - slice_thickness + 1):
            mask_binary_slice = mask_binary[k:(k + slice_thickness), ...]
            
            # Only take slices with foreground in them - this is for training only
            if  np.sum(mask_binary_slice[..., 1]) > 25:
                
                # Get corresponding image slices
                img_slice = img[k:(k + slice_thickness), ...]
                
                # Name the slices and write them to disk
                slice_name = current_pat['id'] + '_' + str(k) + '.npy'
                img_slice_name = image_dest + slice_name
                mask_binary_slice_name = mask_dest + slice_name

                np.save(img_slice_name, img_slice)
                np.save(mask_binary_slice_name, mask_binary_slice)

                # Track slices with output dataframe
                output_df = output_df.append({'id': current_pat['id'], 
                                              'image': img_slice_name, 
                                              'mask': mask_binary_slice_name}, 
                                              ignore_index = True)
    
    # Save dataframe to .csv and use the .csv for training the BraTS model
    output_df.to_csv(output_csv, index = False)
    
    ############## END OF FUNCTION ##############


# ### Create file path CSVs for BraTS and NFBS datasets

# In[ ]:


################# Create binary masks for BraTS #################

# Change this to the appropriate folder on your system 
brats_base_dir = '/rsrch1/ip/aecelaya/data/brats_2020/raw/train/'

binarize_brats(brats_base_dir)

################# Create CSV with BraTS file paths #################
brats_names_dict = {'mask': ['seg_binary.nii.gz'],
                    't1': ['t1.nii.gz'],
                    't2': ['t2.nii.gz'], 
                    'tc': ['t1ce.nii.gz'], 
                    'fl': ['flair.nii.gz']}

brats_output_csv = 'brats_paths.csv'
get_paths_csv(brats_base_dir, brats_names_dict, brats_output_csv)

################# Create CSV with NFBS file paths #################
nfbs_names_dict = {'mask': ['brainmask.nii.gz'],
                   't1': ['T1w.nii.gz']}

# Change this to the appropriate folder on your system 
nfbs_base_dir = '/rsrch1/ip/aecelaya/data/nfbs/raw/'

nfbs_output_csv = 'nfbs_paths.csv'
get_paths_csv(nfbs_base_dir, nfbs_names_dict, nfbs_output_csv)


# ### Preprocess BraTS and NFBS and write slices to disk

# In[ ]:


################# Preprocess and write BraTS slices to disk #################
brats_input_csv = 'brats_paths.csv'

# Change these to the appropriate folder on your system 
brats_image_dest = '/rsrch1/ip/aecelaya/github/NecrosisRecurrence/pocketnet/brats/test/images/'
brats_mask_dest = '/rsrch1/ip/aecelaya/github/NecrosisRecurrence/pocketnet/brats/test/masks/'


brats_output_csv = 'brats_slices_paths.csv'

write_slices(brats_input_csv, brats_image_dest, brats_mask_dest, brats_output_csv, image_dims = (155, 240, 240))

################# Preprocess and write NFBS slices to disk #################
nfbs_input_csv = 'nfbs_paths.csv'

# Change these to the appropriate folder on your system 
nfbs_image_dest = '/rsrch1/ip/aecelaya/github/NecrosisRecurrence/pocketnet/brats/test2/images/'
nfbs_mask_dest = '/rsrch1/ip/aecelaya/github/NecrosisRecurrence/pocketnet/brats/test2/masks/'

nfbs_output_csv = 'nfbs_slices_paths.csv'

write_slices(nfbs_input_csv, nfbs_image_dest, nfbs_mask_dest, nfbs_output_csv, image_dims = (192, 256, 256))


# ### Clean up file names for COVIDx8B

# In[ ]:


'''
Clean up the COVIDx dataset. There are a few glitches in it. This script corrects them.

1) Some files in the COVIDx training set are compressed (i.e., end with .gz). Keras can't read
zipped files with its native image data generators. This script goes through each file
and checks to see if its compressed and unzips it if it is. 

2) The original train.csv file that comes with the COVIDx dataset has incorrect file names for rows
725 - 1667. These rows only contian numbers and not the name of an image. For example, row 725 has 
the entry 1 but it should be COVID1.png.

Before running this, please change the file paths in this code to match your system.
'''

def get_files(dir_name):
    list_of_files = list()
    for (dirpath, dirnames, filenames) in os.walk(dir_name):
        list_of_files += [os.path.join(dirpath, file) for file in filenames]
    return list_of_files

files_list = get_files('/rsrch1/ip/aecelaya/data/covidx/processed/train')
for file in files_list:
    if '(' in file:
        new_file = file.replace('(', '')
        new_file = new_file.replace(')', '')
        print('Renaming ' + file + ' to ' + new_file)
        os.rename(file, new_file)
        file = new_file
        
    if '.gz' in file:
        # Unzip files with gunzip
        print('Unzipping ' + file)
        subprocess.call('gunzip ' + file, shell = True)
        
train_df = pd.read_csv('/rsrch1/ip/aecelaya/data/covidx/raw/data/train.csv')
for i in range(724, 1667):
    number = train_df.iloc[i]['image']
    train_df.at[i, 'image'] = 'COVID' + number + '.png'
    
for i in range(len(train_df)):
    file = '/rsrch1/ip/aecelaya/data/covidx/train/' + train_df.iloc[i]['image']
    if not os.path.isfile(file):
        print('Does not exist: ' + file + ', row = ' + str(i))

train_df.to_csv('covidx_train_clean.csv', index = False)

