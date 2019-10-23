# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:25:26 2019

@author: Aditya
Convert .stl files in a directory to 3-view png
"""

#import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
from matplotlib import cm

#### set dark background for computational efficiency ####
plt.style.use('dark_background')

class MyDirectory:
    import os
    def __init__(self: "MyDirectory",path: str) -> None:
        self.path = path
        
    def list_stl_files(self: "MyDirectory", full_path: bool=True) -> list:
        '''
        List all .stl files within the directory
        :params full_path: return file names as full path
        '''
        import os
        files_list=[]
        for file in os.listdir(os.fsencode(self.path)):
            filename = os.fsdecode(file)
            if filename.endswith('.stl'):
                if full_path:
                    files_list.append(self.path+'/'+filename)
                else:
                    files_list.append(filename)
        return files_list
    
class MyMesh:
    def __init__(self, path: str) -> None:
        '''
        Initialize the mesh object
        :params path: specifies the path to the .stl file in str
        '''
        self.location = path
        self.mesh = mesh.Mesh.from_file(path)
           
    def locate(self: "MyMesh") -> str:
        return self.location

    def get_mesh(self: "MyMesh") -> mesh.Mesh:
        return self.mesh
    
    def save_3_view(self: "MyMesh",color_scheme=cm.GnBu):
        '''
        Save 3-view .png images of mesh.
        Returns a function call which in turn returns a NoneType
        '''
        return self.make_3_view(color_scheme,save=True)
    
    def make_3_view(self: "MyMesh",color_scheme=cm.GnBu,save=False) -> None:
        '''
        Make 3-view images of mesh object.
        Optionally save them. equivalent to save_3_view() 
        '''
        _mesh = self.get_mesh()
        
        # make first view
        figure1 = plt.figure(figsize=(10,10))
        axes1 = mplot3d.Axes3D(figure1)    # define axis object
        axes1.plot_surface(_mesh.x,_mesh.y,_mesh.z) # plot on the axis
        plt.axis('off') # remove axes
        destination = self.location.replace('.stl','_view1.png')
        if save:    
            plt.savefig(destination) # save figure
        
        # rotate for second view
        _mesh.rotate([1,1,0],math.radians(-80))
        
        # make second view
        figure2 = plt.figure(figsize=(10,10))
        axes2  = mplot3d.Axes3D(figure2)
        axes2.plot_surface(_mesh.x,_mesh.y,_mesh.z)
        plt.axis('off')
        destination = self.location.replace('.stl','_view2.png')
        if save:
            plt.savefig(destination)
        
        # rotate for third view
        _mesh.rotate([1,1,0],math.radians(80))
        _mesh.rotate([0.1,0,1],math.radians(90))
        
        # make thrid view
        figure3 = plt.figure(figsize=(10,10))
        axes3 = mplot3d.Axes3D(figure3)
        axes3.plot_surface(_mesh.x,_mesh.y,_mesh.z)
        plt.axis('off')
        destination = self.location.replace('.stl','_view3.png')
        if save:
            plt.savefig(destination)
        
        #rotate mesh back
        _mesh.rotate([0.1,0,1],math.radians(-90))
        
        
    
    
def save_png_from_directory_path(path: str, color_scheme=cm.GnBu)-> None:
    '''
    Factory function to save 3-view .png images of all .stl files 
    in the directory
    '''
    directory = MyDirectory(path)
    stl_files = directory.list_stl_files()
    for file in stl_files:
        mesh_object = MyMesh(file)
        mesh_object.make_3_view(save=True,color_scheme=color_scheme)