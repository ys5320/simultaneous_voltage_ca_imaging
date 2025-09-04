import sys
import os

import functions as canf
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import tifffile as tiff

def make_videos(df_file, top_dir, HPC_num = None):
    df = df_file
    for idx,data in df.iterrows():
        if HPC_num is not None:  # allows running in parallel on HPC
            if idx != HPC_num:
                print(HPC_num, idx)
                continue
        print(data.trial_string)
        save_dir = Path(top_dir,'analysis','results_profiles',f'{data.trial_string}')
        save_dir.mkdir(parents=True, exist_ok=True)
        folder_path = Path(data.folder)

        if not folder_path.exists():
            print(f"Skipping {folder_path}, does not exist.")
            continue

        # Find and sort .tif files in the correct order
        tif_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".ome.tif")])
        # Custom sorting logic:
        def custom_sort(filename):
            if filename.endswith("_Default.ome.tif"):
                return 0  # Highest priority (always comes first)
            elif filename.endswith("_Default_1.ome.tif"):
                return 1  # Comes second
            elif filename.endswith("_Default_2.ome.tif"):
                return 2  # Comes third (if it exists)
            return 3  # Any other files

        tif_files.sort(key=custom_sort)  # Sort dynamically based on existence
        #print(tif_files)
        if not tif_files:
            print(f"No .tif files found in {folder_path}")
            continue
        print(tif_files)
        '''
        # Load Z-stack images using tifffile (instead of OpenCV)
        image_stack = []
        for tif in tif_files:
            img_stack = tiff.imread(str(folder_path / tif))  # Reads full Z-stack
            image_stack.append(img_stack)
            print(img_stack.shape)

        image_stack = np.concatenate(image_stack, axis=0)  # Stack all slices along Z-axis
        '''
        
        image_stack = tiff.imread(str(folder_path / tif_files[0]))

        if image_stack.ndim != 3:
            print(f"Skipping {folder_path}, no valid images loaded.")
            continue
        '''
        n = image_stack.shape[0] / len(tif_files)
        image_stack = image_stack[:n, :, :]
        '''
        print(f"Loaded Z-stack shape: {image_stack.shape}")  # Should be (n, 512, 512)
        
        # Deinterleave: Split into 2 channels
        
        voltage_channel = image_stack[::2, :, :]  # Take every even z
        calcium_channel = image_stack[1::2, :, :]  # Take every odd z
        print(voltage_channel.shape, calcium_channel.shape)
        '''
        # Auto-adjust brightness and enhance sharpness
        enhanced_voltage = np.array([canf.enhance_contrast(slice) for slice in voltage_channel])
        enhanced_calcium = np.array([canf.enhance_contrast(slice) for slice in calcium_channel])
        
        # Save the enhanced channels as AVI
        canf.save_to_avi(enhanced_voltage, "enhanced_voltage_video.avi", save_dir)
        canf.save_to_avi(enhanced_calcium, "enhanced_calcium_video.avi", save_dir)
        '''
        
        voltage_channel = np.array([canf.adjust_brightness(slice) for slice in voltage_channel])
        calcium_channel = np.array([canf.adjust_brightness(slice) for slice in calcium_channel])
        #voltage_channel = np.array([canf.enhance_sharpness(slice) for slice in voltage_channel])
        #calcium_channel = np.array([canf.enhance_sharpness(slice) for slice in calcium_channel])
        print(voltage_channel.shape, calcium_channel.shape)
        canf.save_to_avi(voltage_channel, "enhanced_voltage_video.avi", save_dir)
        canf.save_to_avi(calcium_channel, "enhanced_calcium_video.avi", save_dir)
        

    