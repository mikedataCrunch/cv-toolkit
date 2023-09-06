"""
Module for exploration of image dataset and accompanying metadata

BY: MDOROSAN 2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import os

def _parse_path_meta(path):
    """Return class and fileid (filename) from path str"""
    class_, id_ = path.split("/")[-2:]
    return class_, id_


def _plot_img(path, fig, rows, cols, pos):        
    
    # read img from path
    image = mpimg.imread(path) 
    
    # plot onto axis
    ax = fig.add_subplot(rows, cols, pos)
    ax.imshow(image)
    
    # formating
    ax.axis('off')
    ax.set_title("Class: {},\nFileID: {}".format(
        *_parse_path_meta(path)), fontsize=6)
    
    return None


def inspect_samples(img_root, num_samples=3, classes=None):
    """Plot and save an image grid of samples images."""
    
    if not classes:
        # set up grid dims from args
        rows, cols = int(np.ceil(num_samples / 3)), 3 
        
        # get paths from img root
        paths = glob(os.path.join(img_root, '*', '*'))
        
        # randomly sample from all paths
        samples = np.random.choice(paths, size=num_samples)
        
        # plot per sample into grid
        fig = plt.figure(1, figsize=(2*cols, 2*rows))
        for index, sample_path in enumerate(samples):
            _plot_img(sample_path, fig, rows, cols, pos=index+1)
        plt.show()
    
    else: 
        if classes == 'all':
            classes = os.listdir(img_root)
        else:
            pass
        
        # set up grid from args 
        rows, cols =  len(classes), num_samples
        
        # get samples per class
        samples = []
        for class_ in classes:
            
            # get img paths for class_
            paths = glob(os.path.join(img_root, class_, '*'))
            
            # extend all samples from class samples
            samples.extend(np.random.choice(paths, size=num_samples))
            
        # plot per sample into grid            
        fig = plt.figure(1, figsize=(2*cols, 2*rows))
        for index, sample_path in enumerate(samples):
            _plot_img(sample_path, fig, rows, cols, pos=index+1)
        plt.show()