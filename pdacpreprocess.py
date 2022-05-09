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
        # arr = sitk.Normalize(arr)
        arr = sitk.GetArrayFromImage(arr)
        return arr
    
    image = np.empty((*dims, len(images_list)))
    for i in range(len(images_list)):
        image[..., i] = GetArray(images_list[i])
        print(images_list[i],'min',np.min(image[..., i]),'max',np.max(image[..., i]),'mean',np.mean(image[..., i]),'std',np.std(image[..., i]) )

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
    
    output_cols = ['id', 'image', 'truthid', 'target']
    output_df = pd.DataFrame(columns = output_cols)
    
    for i in trange(num_pats):
        
        # Get row of input dataframe
        current_pat = input_df.iloc[i].to_dict()
        
        # Read in images and masks
        images_list = list(current_pat.values())[2:3]
        print ( list(current_pat.values())[3:])
        img = ReadImagesSITK(images_list, dims = image_dims)
        #mask_binary = ReadMaskSITK(current_pat['mask'], classes = [0, 1])

        img_slice_name = image_dest + "%d/image.npy"  % current_pat['id'] 
        print(img.shape)
        np.save(img_slice_name, img )
        # Track slices with output dataframe
        output_df = output_df.append({'id': current_pat['id'], 
                                      'image': img_slice_name, 
                                      'truthid': current_pat['truthid'], 
                                      'target': current_pat['target']}, 
                                      ignore_index = True)
    

    # Save dataframe to .csv and use the .csv for training the BraTS model
    output_df.to_csv(output_csv, index = False)
    
    ############## END OF FUNCTION ##############

# ### Create file path CSVs 

################# Preprocess and write PDAC slices to disk #################
pdac_input_csv = 'dicom/wideclassificationroid2.csv'


# Change these to the appropriate folder on your system 
pdac_image_dest = '/rsrch3/ip/dtfuentes/github/pdacclassify/D2Processed/'
pdac_mask_dest  = pdac_image_dest 

pdac_output_csv = 'dicom/wide_slices_paths.csv'

write_slices(pdac_input_csv, pdac_image_dest, pdac_mask_dest, pdac_output_csv, image_dims = (96, 256, 256))


