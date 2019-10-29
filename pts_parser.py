# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:02:45 2019

@author: Aditya
A script to parse pts files and return the x,y,z coordinates as np arrays 
in a tuple
"""

import numpy as np
from math_functions import rotation_matrix

class PointCloud:
    
    @staticmethod
    def parse_pts(filename):
        file = open(filename, 'rb')
        
        contents = file.read()
        file.close()
        coords = [float(tmp) for tmp in contents.split()]
        new_len = len(coords[1:])//3
        return np.array(coords[1:]).reshape([new_len,3]) # reshape as (x,y,z) coords
    
    def __init__(self,filepath):
        '''
        Initialize PointCloud objects with path to .pts file as the argument
        '''
        self.filepath = filepath
        self.coords = PointCloud.parse_pts(filepath)
    
    def rotate(self,axis,theta):
        R = rotation_matrix(axis,-theta) # negative in line with the clockwise rotations of numpy-stl
        new_coords = np.dot(R,self.coords.T).T
        return new_coords
        
    
