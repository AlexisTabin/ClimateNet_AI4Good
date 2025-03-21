import numpy as np
import configparser

import xarray as xr
from patchify import patchify
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import os
from collections import Counter
from decouple import config
import json

#read in config file
config = configparser.ConfigParser()
config.read('config.yaml')

#set path to directories from config
DATA_PATH = config['path']['data_path']
REPO_PATH = config["path"]["repo_path"]
CL = f'{REPO_PATH}curriculum.txt'

'''
Functions for extracting patches according to their characteristics

AR_o:   optimal ARs (thresholded)
AR:     random ARs

TC_o:   optimal TCs (thresholded)
TC:     random TCs

BG:     Background

M_o:    optimal mixed (thresholded)
M:      random mixed

R:      random patches

A:      all patches
'''
def AR_o(class_freq, max_exp_patches):
    subset=np.squeeze(np.argwhere((class_freq[:,1]==0.0)& (class_freq[:,2]>0.0))) #AR and no TC
    if subset.size < 2:
            return []

    #draw from set of valid sample patches
    elif subset.size < max_exp_patches: #if subset smaller than requested set sample with replacement
            draws = np.random.choice(subset.size, max_exp_patches) 
            ids = subset[draws]
    else: 
            ids = subset[np.argsort(class_freq[subset,2])[::-1][:max_exp_patches]] #return best patches
    return ids

def AR(class_freq, max_exp_patches):
    subset=np.squeeze(np.argwhere((class_freq[:,1]==0.0)& (class_freq[:,2]>0.0)))  #AR and no TC
    if subset.size < 2:
            return []
    elif subset.size < max_exp_patches:
            draws = np.random.choice(subset.size, max_exp_patches)
    else: 
            draws = np.random.choice(subset.size, max_exp_patches, replace = False) #sample without replacement
    ids = subset[draws]
    return ids

def TC_o(class_freq, max_exp_patches):
    subset=np.squeeze(np.argwhere((class_freq[:,2]==0.0)& (class_freq[:,1]>0.0)))  #TC and no AR
    if subset.size < 2:
            return []
    elif subset.size < max_exp_patches:
            draws = np.random.choice(subset.size, max_exp_patches)
            ids = subset[draws]
    else: 
            ids = subset[np.argsort(class_freq[subset,1])[::-1][:max_exp_patches]]
    return ids

def TC(class_freq, max_exp_patches):
    subset=np.squeeze(np.argwhere((class_freq[:,2]==0.0)& (class_freq[:,1]>0.0))) #TC and no AR
    if subset.size < 2:
            return []
    elif subset.size < max_exp_patches:
            draws = np.random.choice(subset.size, max_exp_patches)
    else: 
            draws = np.random.choice(subset.size, max_exp_patches, replace = False)
    ids = subset[draws]
    return ids
    

def BG(class_freq, max_exp_patches):
    subset=np.squeeze(np.argwhere(class_freq[:,0]==1.0)) #BG
    if subset.size < 2:
            return []
    elif subset.size < max_exp_patches:
            draws = np.random.choice(subset.size, max_exp_patches)
    else: 
            draws = np.random.choice(subset.size, max_exp_patches, replace = False)
    ids = subset[draws]
    return ids     

def M_o(class_freq, max_exp_patches):
    combined = class_freq[:,1]*class_freq[:,2] # TC * AR > 0
    try:
        subset=np.hstack(np.argwhere(combined > 0))
    except:
        subset = np.array([])
    if subset.size < 2:
        return []
    if subset.size < max_exp_patches:
        draws = np.random.choice(subset.size, max_exp_patches)
        ids = subset[draws]
    else: 
        ids = subset[np.argsort(combined[subset])[::-1][:max_exp_patches]]
    return ids

def M(class_freq, max_exp_patches):
    combined = class_freq[:,1]*class_freq[:,2] # TC * AR > 0
    try:
        subset=np.hstack(np.argwhere(combined > 0))
    except:
        subset = np.array([])
    if subset.size < 2:
        return []
    elif subset.size < max_exp_patches:
        draws = np.random.choice(subset.size, max_exp_patches)
    else: 
        draws = np.random.choice(subset.size, max_exp_patches, replace = False)
    ids = subset[draws]
    return ids

#extract random patches
def R(class_freq, max_exp_patches):
    ids = np.random.choice(class_freq.shape[0], max_exp_patches, replace = False)
    return ids

#extract all patches
def A(class_freq, max_exp_patches):
    ids = np.random.choice(class_freq.shape[0], max_exp_patches, replace = False)
    return ids

def patch_image(image, patch_size, stride, vars):
    """
    Splits single input image into square patches of defined size and stride.
    Each patch only contains specified variables of interest.
    
    :param image: input image of type xarray.core.dataset.Dataset
    :param patch_size: pixel size of square patch. Note patch_size must not exceed the image height.
    :param vars: array of variable names, e.g. ['Z1000', 'U850', 'V850']
    :return: image patches as a single variable of shape (output_H * output_W, len(vars) + 1, patch_size, patch_size), with:
    output_H = np.floor((H-patch_size+stride)/stride).astype(int),
    output_W = np.floor((W-patch_size+stride)/stride).astype(int).
    """
    im = np.expand_dims(np.array(image['LABELS']), axis=0)
    for j, var in enumerate(vars):
        im = np.concatenate((im, np.array(image[var])), axis=0)
    im_patches = np.squeeze(patchify(im, (len(vars)+1, patch_size, patch_size), stride), axis=0) # +1 to include labels in patch
    return np.reshape(im_patches, (-1, len(vars)+1, patch_size, patch_size))

def calc_class_freq(im_patches):
    """
    Calculates the class frequency for all patches drawn from the image input.
    :param im_patches: array containing all patches for a single image (output of patch_image function).
    :return: shape is (3, patch_size**2)
    """

    nr_patches = len(im_patches)
    nr_pixels = im_patches.shape[-1]**2
    class_counts = [Counter(list(patch[0,:,:].flatten()))for patch in im_patches.astype(np.uint8)]
    class_freq = np.reshape(np.array([np.array([counts[0], counts[1], counts[2]])/(nr_pixels) for counts in class_counts]), 
                 (nr_patches,-1))

    return class_freq
   

def save_best_patches(set, vars,file_name, image, im_patches, class_freq, max_exp_patches):
    patch_size = im_patches.shape[-1]
    stride = im_patches.shape[1]

    json_file_path = CL

    with open(json_file_path, 'r') as j:
        curriculum = json.loads(j.read())
    stages = list(curriculum.keys())
    subsets = list(curriculum.values())


    paths = [(os.path.join(DATA_PATH,'cl/',str(patch_size)+'/',f'{set}/', stage_name+'/')) for stage_name in stages]
    
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created the folder: {path}")

    idx = np.zeros((len(stages), max_exp_patches), dtype=int)

    #extract patches for all requested groups
    for i, (stage, set) in enumerate(curriculum.items()):
        functions = [globals()[type]for type in set]
        data = np.hstack(np.array([func(class_freq, max_exp_patches) for func in functions], dtype=object))
        if len(data) == 0:
            break
        else:
            draws = np.random.choice(len(data), max_exp_patches)
            idx[i,:] = data[draws]


   
    
    ##### patch the latitude and longitude #####
    H = 768
    W = 1152
    H_out = np.floor((H-patch_size+stride)/stride).astype(int)
    W_out = np.floor((W-patch_size+stride)/stride).astype(int)


    lat_im = np.linspace(-90, 90, 768)
    lon_im = np.linspace(0, 359.7, 1152)

    lat_all = np.empty((H_out, patch_size))
    lon_all = np.empty((W_out, patch_size))

    for i in range(H_out): lat_all[i,:] = lat_im[i*stride:patch_size+i*stride]
    for i in range(W_out): lon_all[i,:] = lon_im[i*stride:patch_size+i*stride]
    # print(lat_all.shape, lon_all.shape)

    ###### select best patches; assign correct lat, lon to each patch; create and save .nc file #####
    for i, (stage, set) in enumerate(curriculum.items()):
        for n in range(max_exp_patches):
            save_patch = im_patches[idx[i,n],:,:]

            lat_idx = np.ceil(idx[i,n]/W_out).astype(int)-1
            lat = lat_all[lat_idx,:]
            lon_idx = np.ceil(idx[i,n]%W_out).astype(int)-1 if np.ceil(idx[0,n]%W_out) !=0 else idx.shape[1]
            lon = lon_all[lon_idx,:]

            coords = {'lat': (['lat'], lat),
                'lon': (['lon'], lon),
                'time': (['time'], [np.array(image['time'])[0][5:-3]])}

            data_vars={}
            for j in range(len(vars)):
                data_vars[vars[j]] = (['time', 'lat', 'lon'], np.expand_dims(save_patch[j+1,:,:].astype(np.float32), axis=0))
            data_vars["LABELS"] = (['lat', 'lon'], save_patch[0,:,:].astype(np.int64))

            xr_patch = xr.Dataset(data_vars=data_vars, coords=coords)
            
            xr_patch.to_netcdf(os.path.join(paths[i]+str(set)+'_'+file_name+"_p"+str(n)+".nc"))
            xr_patch.close()

#save all samples
def save_best_patches_test(set, vars,file_name, image, im_patches, class_freq, max_exp_patches, mode):
    patch_size = im_patches.shape[-1]
    stride = im_patches.shape[1]

    idx = np.arange(len(im_patches))
   
    
    ##### patch the latitude and longitude #####
    H = 768
    W = 1152
    H_out = np.floor((H-patch_size+stride)/stride).astype(int)
    W_out = np.floor((W-patch_size+stride)/stride).astype(int)


    lat_im = np.linspace(-90, 90, 768)
    lon_im = np.linspace(0, 359.7, 1152)

    lat_all = np.empty((H_out, patch_size))
    lon_all = np.empty((W_out, patch_size))

    for i in range(H_out): lat_all[i,:] = lat_im[i*stride:patch_size+i*stride]
    for i in range(W_out): lon_all[i,:] = lon_im[i*stride:patch_size+i*stride]
    # print(lat_all.shape, lon_all.shape)


    ###### select best patches; assign correct lat, lon to each patch; create and save .nc file #####
    if mode == 'True':
        path_exp = os.path.join(DATA_PATH,str(patch_size)+'/',f'{set}/')
    else:
        path_exp = os.path.join(DATA_PATH,'cl/',str(patch_size)+'/',f'{set}/')
    if not os.path.exists(path_exp):
            os.makedirs(path_exp)
            print(f"Created the folder: {path_exp}")
    for n in range(len(im_patches)):
        save_patch = im_patches[idx[n],:,:]

        lat_idx = np.ceil(idx[n]/W_out).astype(int)-1
        lat = lat_all[lat_idx,:]
        lon_idx = np.ceil(idx[n]%W_out).astype(int)-1 
        lon = lon_all[lon_idx,:]

        coords = {'lat': (['lat'], lat),
            'lon': (['lon'], lon),
            'time': (['time'], [np.array(image['time'])[0][5:-3]])}

        data_vars={}
        for j in range(len(vars)):
            data_vars[vars[j]] = (['time', 'lat', 'lon'], np.expand_dims(save_patch[j+1,:,:].astype(np.float32), axis=0))
        data_vars["LABELS"] = (['lat', 'lon'], save_patch[0,:,:].astype(np.int64))

        xr_patch = xr.Dataset(data_vars=data_vars, coords=coords)
        xr_patch.to_netcdf(os.path.join(path_exp+file_name+"_p"+str(n)+".nc"))
        xr_patch.close()

def load_single_image(image_path):
    return xr.load_dataset(image_path)

def process_single_image(set, file_name, image, patch_size, stride, vars, max_exp_patches, mode):
    
    im_patches = patch_image(image, patch_size, stride, vars)
    class_freq = calc_class_freq(im_patches)

    #patch training - extract all patches
    if mode == 'True':
        save_best_patches_test(set, vars,file_name, image, im_patches, class_freq, max_exp_patches, mode)

    #extract and draw for groups, if test extract all
    else:
        if set == 'test':
            save_best_patches_test(set, vars,file_name, image, im_patches, class_freq, max_exp_patches, mode)
        else:
            save_best_patches(set, vars,file_name, image, im_patches, class_freq, max_exp_patches)

    

def process_all_images(patch_size, stride, vars, max_exp_patches, mode):

    for set in ['train','val','test']:

        data_dir = f'{DATA_PATH}{set}/'
        single_file_paths = [data_dir+f for f in listdir(data_dir) if isfile(join(data_dir, f))]
        file_names = [f[:-3] for f in listdir(data_dir) if isfile(join(data_dir, f))]
        print('Load all images')
        data = []
        for p in tqdm(single_file_paths[:]):
            try:
                data.append(xr.load_dataset(p))
            except:
                pass

        print('process images')
        for i, image in enumerate(tqdm(data)):
            process_single_image(set, file_names[i], image, patch_size, stride, vars, max_exp_patches, mode)


if __name__ == "__main__":
   pass