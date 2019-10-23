# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:12:20 2019

@author: Aditya
A script to automate the segregation process done by tanya and satish
"""

import os
import pandas as pd
import shutil
import warnings

def move_files_to_labelled_folders(root: str,
                                   quality_report: pd.DataFrame) -> None:
    quality_report.set_index('ID',inplace=True)
    quality_report['Scan Quality'] = quality_report['Scan Quality'].astype('str').str.upper()
    
    for _, dir_list, _ in os.walk(root):
        scan_dirs = dir_list
        break
    
    locate_directory = {}    
    for dir_name in scan_dirs:
        scan_id = dir_name.split('_')[2]
        locate_directory[scan_id] = dir_name 
        
    for scan_id in locate_directory.keys():
        
        try:
            quality = quality_report.loc[scan_id]['Scan Quality'].item()
            case_type = quality_report.loc[scan_id]['Case Type'].item()
            
        except ValueError:
            quality = quality_report.loc[scan_id]['Scan Quality'].iloc[0]
            warnings.warn('Multiple entries present for this scan')
            case_type = quality_report.loc[scan_id]['Case Type'].iloc[0]
            
        shutil.move(os.path.join(root,locate_directory[scan_id]), 
                    os.path.join(root,case_type,quality,locate_directory[scan_id]))
        
if __name__ == '__main__':
#   enter root directory of scans here
    root = r"C:\Users\jerome\Documents\raw_data_leixir\1"
#   enter location of quality report excel file here
    quality_report_loc = r"C:\Users\jerome\Downloads\Scan Quality Report Sep 2019 for AI_edited.xlsx"
    quality_report = pd.read_excel(quality_report_loc)
    move_files_to_labelled_folders(root,quality_report)