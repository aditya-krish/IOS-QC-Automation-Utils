# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:04:27 2019

@author: Aditya
A better stl2png
"""

from stl import mesh
import matplotlib.pyplot as plt
import math
import vtkplotlib as vpl

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
    
    @staticmethod
    def _make_current_view(mesh_vectors, **kwargs) -> None:
        '''
        Make current view of the mesh. 
        :params mesh_vectors: np.ndarray object containing the vectors of the mesh
        :params 
        '''
        vpl.mesh_plot(mesh_vectors)
        image = vpl.screenshot_fig()
        if kwargs.get('save',False):
            try:
                plt.imsave(kwargs['destination'],image)
            except KeyError:
                raise RuntimeError('Destination has not been mentioned')
        vpl.show(block=False) 
        vpl.close()
    
    def save_4_view(self: "MyMesh"):
        '''
        Save 4-view .jpeg images of mesh.
        Returns a function call which in turn returns a NoneType
        '''
        return self.make_4_view(save=True)
    
    def make_4_view(self: "MyMesh",save=False) -> None:
        '''
        Make 4-view images of mesh object.
        Optionally save them. equivalent to save_4_view() 
        '''
        _mesh = self.get_mesh()
        
        # make first view
        destination = self.location.replace('.stl','_view1.jpeg')
        MyMesh._make_current_view(_mesh.vectors, 
                                  destination=destination, 
                                  save=save
                                  )
        # rotate for second view
        _mesh.rotate([0,1,0],math.radians(120))
        # make second view
        destination = self.location.replace('.stl','_view2.jpeg')
        MyMesh._make_current_view(_mesh.vectors, 
                                  destination=destination, 
                                  save=save
                                  )
        # rotate for third view
        _mesh.rotate([0,1,0],math.radians(-120))
        _mesh.rotate([0,0,1],math.radians(120))
        _mesh.rotate([0,1,0],math.radians(120))
        # make thrid view
        destination = self.location.replace('.stl','_view3.jpeg')
        MyMesh._make_current_view(_mesh.vectors, 
                                  destination=destination, 
                                  save=save
                                  )
        # rotate for fourth view 
        _mesh.rotate([0,1,0],math.radians(-120))# reverse third view
        _mesh.rotate([0,0,1],math.radians(120))
        _mesh.rotate([0,1,0],math.radians(120))
        destination = self.location.replace('.stl','_view4.jpeg')
        MyMesh._make_current_view(_mesh.vectors, destination=destination, save=save)
        
        # rotate mesh back
        _mesh.rotate([0,1,0],math.radians(-120))       
        _mesh.rotate([0,0,1],math.radians(120))      
    
    
def save_images_from_directory_path(path: str)-> None:
    '''
    Factory function to save 4-view .jpeg images of all .stl files 
    in the directory
    '''
    directory = MyDirectory(path)
    stl_files = directory.list_stl_files()
    for file in stl_files:
        mesh_object = MyMesh(file)
        mesh_object.save_4_view()
