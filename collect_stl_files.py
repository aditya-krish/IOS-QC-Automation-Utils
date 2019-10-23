# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:00:46 2019

@author: Aditya
A script to move .stl files from the iTero directories sent by Satish Kumar into
the desktop directories I have created 
"""

import os
import shutil

roots ={'bad': r"C:\Users\jerome\Documents\raw_data_leixir\1",
        'manageable': r"C:\Users\jerome\Documents\raw_data_leixir\2",
        'good': r"C:\Users\jerome\Documents\raw_data_leixir\3"}
qualities = ['bad','manageable','good']

stl_files = {qualities[0]:[],qualities[1]:[],qualities[2]:[]}
sub_dirs = {qualities[0]:[],qualities[1]:[],qualities[2]:[]}

for qual, path in roots.items():
    for _, sub_dir_list, _ in os.walk(roots[qual]):
        sub_dirs[qual] += sub_dir_list

for qual in qualities:
    for sub_dir in sub_dirs[qual]:
        for file in os.listdir(os.fsencode(os.path.join(roots[qual],sub_dir))):
            filename = os.fsdecode(file)
            if filename.endswith('.stl'):
                full_name = os.path.join(roots[qual],sub_dir,filename)
                stl_files[qual].append(full_name)
                
target_dirs = {'bad':r"C:\Users\jerome\Desktop\bad",
               'manageable':r"C:\Users\jerome\Desktop\manageable",
               'good':r"C:\Users\jerome\Desktop\good"}

for qual in qualities:
    for file in stl_files[qual]:
        shutil.copy(file,target_dirs[qual])