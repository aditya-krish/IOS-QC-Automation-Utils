# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:04:06 2019

@author: Aditya
A script to merge the lookup tables
"""

import pandas as pd

lookup_table = pd.read_excel(
        r"C:\Users\jerome\Downloads\Scan Quality Report 2019 for AI.xlsx",
        sheet_name = 'Sheet2')

lookup_table.set_index('Scan ID', inplace = True)
lookup_table['Rx Quality'].fillna('',inplace=True)
lookup_table['Rx Quality']=lookup_table['Rx Quality'].str.lower()