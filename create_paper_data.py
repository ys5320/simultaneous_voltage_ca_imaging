import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
import sys

home = Path.home()
cell_line = 'MDA_MB_468'
if "ys5320" in str(home):
    HPC = True
    top_dir = Path(home, "firefly_link/Calcium_Voltage_Imaging",f'{cell_line}')
    df_str = "_HPC"
    HPC_num = (
        int(sys.argv[1]) - 1
    )  # allows running on HPC with data parallelism 
    date = '20250902'
    df = Path(top_dir,'analysis', 'dataframes',f'MDA_MB_468_dataframe_tc_extracted.csv')

    save_dir = Path(top_dir,'analysis', 'results_profiles')

HPC = True
if HPC:
    df_data = pd.read_csv(df)
    df_data = df_data[df_data['multi_tif']>1]
    df_data = df_data[df_data['use'] != 'n']
    toxins = ['ATP', 'TRAM-34', 'L-15', 'Dantrolene', 'dantrolene','Ani9','siRNA_negative','siRNA_kcnn4','PPADS','YM58483','Thapsigargin','heparin','4AP','Ca_free','DMSO']
    
    df_data = df_data[df_data['expt'].apply(lambda x: any(k in x for k in toxins))]

    df_data = df_data.reset_index(drop=True)  
    
    print("Original path:", df_data.iloc[0]['folder'])
    df_data['folder'] = df_data['folder'].str.replace('/user/be320/', '/user/ys5320/')
    print("Fixed path:", df_data.iloc[0]['folder'])
    
    from make_videos import make_videos
    make_videos(df_file = df_data, top_dir = top_dir, HPC_num = HPC_num)
    
    from run_pipeline import run_pipeline
    data_dir = Path(top_dir.parent.parent, 'ca_voltage_imaging_working', 'results_1')
    run_pipeline(df_file = df_data, top_dir = top_dir, data_dir = data_dir, HPC_num = HPC_num)
    