import sys
import os

import functions as canf
from pathlib import Path
import datetime
import numpy as np

home = Path.home()
cell_line = 'MDA_MB_468'

if "ys5320" in str(home):
    HPC = True
    top_dir = Path(home, "firefly_link/Calcium_Voltage_Imaging",f'{cell_line}')
    savestr = "_HPC"
elif os.name == "nt":
    HPC = False
    top_dir = Path(f"R:/firefly_link/Calcium_Voltage_Imaging/{cell_line}")
    savestr = "_HPC"
else:
    HPC = False
    top_dir = Path(home, f"data/Firefly/Calcium_Voltage_Imaging/{cell_line}")
    savestr = ""


save_file = Path(
    top_dir,
    "analysis",
    f"long_acqs_{datetime.datetime.now().year}{datetime.datetime.now().month:02}{datetime.datetime.now().day:02}{savestr}.csv",
)


df = canf.get_tif_smr(
    top_dir, save_file, "20250128", None , cell_line = cell_line,
)

dates = []
slips = []
areas = []
expt = []
trial_string = []
folder = []

for data in df.itertuples():
    s = data.tif_file

    par = Path(s).parts

    dates.append(par[par.index(f"{cell_line}") + 1][-8:])
    #print(dates[-1])
    slips.append(s[s.find("slip") + len("slip") : s.find("slip") + len("slip") + 1])

    if "area" in s:
        areas.append(s[s.find("area") + len("area") : s.find("area") + len("area") + 1])
    else:
        areas.append(s[s.find("cell") + len("cell") : s.find("cell") + len("cell") + 1])

    folder.append(Path(s).parent)
    last_part = par[-1]  # Get the actual filename

    segments = last_part.split("_")  # Split filename by '_'
    
    trial_string.append("_".join(segments[:3]))
    
    extracted_parts = []  # List to store extracted substrings

    # Loop through segments and find the part between "areaX" and "_X"
    for i in range(len(segments)):
        if segments[i].startswith("area") and i < len(segments) - 1:  
            # Take all parts until the next '_X' (which usually indicates MMStack or numbering)
            for j in range(i + 1, len(segments)):
                if segments[j].isdigit():  # Stop if it's a number (like _1)
                    break
                extracted_parts.append(segments[j])  # Store the valid part
            
            break  # Stop processing once we found and extracted the experiment name

    expt.append("_".join(extracted_parts) if extracted_parts else None ) # Join extracted parts with '_'



df['folder'] = folder
df["date"] = dates
df["slip"] = slips
df["area"] = areas
df["trial_string"] = trial_string
df['expt'] = expt

# drop bad goes
df = df[df["multi_tif"] != 0]

df = df.sort_values(by=["date", "slip", "area"])

df.to_csv(
    Path(
        top_dir,
        "analysis", 'dataframes',
        f"long_acqs_{datetime.datetime.now().year}{datetime.datetime.now().month:02}{datetime.datetime.now().day:02}{savestr}_labelled.csv",
    )
)