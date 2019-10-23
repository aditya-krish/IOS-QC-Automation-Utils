# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:31:53 2019

@author: Aditya
Segregate (bad) scans based on bit and margin problems
"""

import os
import vtkplotlib as vpl
import math
from stl import mesh
import matplotlib.pyplot as plt
import copy
from lookup_table import lookup_table
import warnings
import random
#import shutil

def is_flipped(filename: str) -> bool:
    return ("flipped" in filename)

class Destinations:
    bite_root = r"E:\bite_issue"
    margin_root = r"E:\margin_issue"

class ScanFolder:
    def __init__(self, root: str) -> None:
        self.root = root
    
    @staticmethod    
    def list_stl_files(path: str) -> list:
        stl_files = []
        for file in os.listdir(os.fsencode(path)):
            filename = os.fsdecode(file)
            if filename.endswith('.stl'):
                stl_files.append(filename)
        return stl_files
    
    def count_stl_files(self):
        return len(ScanFolder.list_stl_files(self.root))
    
    @property
    def scanID(self):
        folder_name = self.root.split("\\")[-1]
        _id = folder_name.split('_')[2]
        return int(_id)
    
    @property
    def preparation_scan(self):
        stl_files = ScanFolder.list_stl_files(self.root)
        whether_flipped = is_flipped(stl_files[0])
        prep_identifier = {True: 'upper', False: 'lower'}[whether_flipped]
        if self.count_stl_files()>3:
            warnings.warn(f'There is preparation (probably) on both sides of scan with ID {self.scanID}')

        pretreatment_scans = [file for file in stl_files if 'pretreatment' in file]
        if len(pretreatment_scans)==1:
            # only preparation side has pretreatment if only one pretreatment scan is present
            pretreatment_scan = pretreatment_scans[0]
            return os.path.join(self.root,pretreatment_scan)
        elif len(pretreatment_scans) == 2:
            # if two pretreatment scans are present both sides have preparation without ditch
            pretreatment_scan = (
            pretreatment_scans[0] if prep_identifier in pretreatment_scans[0].lower() 
            else pretreatment_scans[1])
            return os.path.join(self.root,pretreatment_scan)
        
        
        preparation_scan = (
                stl_files[0] if prep_identifier in stl_files[0].lower() 
                else stl_files[1])
        return os.path.join(self.root,preparation_scan)
    
    @preparation_scan.getter
    def get_preparation_scan(self):
        return self.preparation_scan
    
    @property
    def bite_scan(self):
        '''
        Fails if there are more than one pretreatment files in the folder 
        (which hasn't been the case till now)
        '''
        stl_files = ScanFolder.list_stl_files(self.root)
        whether_flipped = is_flipped(stl_files[0])

        pretreatment_scans = [file for file in stl_files if 'pretreatment' in file]
        if len(pretreatment_scans)==1 and len(stl_files)==3: # solves special case with 3 scans - one of which is pretreatment
            stl_files.remove(pretreatment_scans[0]) # bite side doesn't have pre-treatment

        bite_identifier = {False: 'upper', True: 'lower'}[whether_flipped]
        bite_scan = (
                stl_files[0] if bite_identifier in stl_files[0].lower() 
                else stl_files[1])
        return os.path.join(self.root,bite_scan)

    @bite_scan.getter
    def get_bite_scan(self):
        return self.bite_scan            
    
class ScanObject:
    def __init__(self, scanID: int, bite_path: str, preparation_path: str) -> None:
        '''
        Initialize the mesh object
        :params path: specifies the path to the .stl file in str
        '''
        self.scanID = scanID
        self.bite_path = bite_path
        self.preparation_path = preparation_path
        self.mesh = mesh.Mesh.from_file(preparation_path)
        self.full_mesh = mesh.Mesh.from_files([preparation_path, bite_path])
           
    def locate_bite_scan(self: "ScanObject") -> str:
        return self.bite_path
    
    def locate_preparation_scan(self: "ScanObject") -> str:
        return self.preparation_path

    def get_preparation_mesh(self: "ScanObject") -> mesh.Mesh:
        return self.mesh
    
    def get_whole_scan_mesh(self) -> mesh.Mesh:
        return self.full_mesh
    
    @staticmethod
    def _make_current_view(mesh_vectors, **kwargs) -> None:
        '''
        Make current view of the mesh. 
        :params mesh_vectors: np.ndarray object containing the vectors of the mesh
        :params 
        '''
        vpl.mesh_plot(mesh_vectors,color=[0.1,0.7,1])
        image = vpl.screenshot_fig()
        if kwargs.get('save',False):
            try:
                plt.imsave(kwargs['destination'],image)
            except KeyError:
                raise RuntimeError('Destination has not been mentioned')
        vpl.show(block=False) 
        vpl.close()
    
    @staticmethod
    def rotate_for_top_view(mesh_):
        _mesh = copy.deepcopy(mesh_)
        _mesh.rotate([1,0,0],math.radians(-77))
        return _mesh
    
    def save_3_view_of_bite(self,destination_root=None):
        # this type of rotation is ideal for full arch scans
        _meshes = copy.deepcopy(self.full_mesh)
        
        random_filename = '_'+str(random.randint(1000,1200))
        ScanObject._make_current_view(
                _meshes.vectors,
                save=True,
                destination=os.path.join(
                        destination_root,str(self.scanID)+random_filename+'_view1.jpeg')
                )
                
        _meshes.rotate([0,1,0],math.radians(-60))
        ScanObject._make_current_view(
                _meshes.vectors,
                save=True,
                destination=os.path.join(
                        destination_root,str(self.scanID)+random_filename+'_view2.jpeg')
                )
        _meshes.rotate([0,1,0],math.radians(120))
        ScanObject._make_current_view(
                _meshes.vectors,
                save=True,
                destination=os.path.join(
                        destination_root,str(self.scanID)+random_filename+'_view3.jpeg')
                )
        
    def save_2_view_of_bite(self,destination_root):
        # this type of rotation is ideal for quadrant scans
        _meshes = copy.deepcopy(self.full_mesh)
        _meshes.rotate([0,1,0],math.radians(80))
        random_filename = '_'+str(random.randint(1000,1200))
        ScanObject._make_current_view(
                _meshes.vectors,
                save=True,
                destination=os.path.join(
                        destination_root,str(self.scanID)+random_filename+'_view1.jpeg')
                )
        _meshes.rotate([0,1,0],math.radians(-160))
        ScanObject._make_current_view(
                _meshes.vectors,
                save=True,
                destination=os.path.join(
                        destination_root,str(self.scanID)+random_filename+'_view2.jpeg')
                )
    
    def save_top_view(self,destination_root):
        _mesh = ScanObject.rotate_for_top_view(self.get_mesh())
        ScanObject._make_current_view(
                _mesh.vectors,
                save=True,
                destination=os.path.join(
                        destination_root,str(self.scanID)+'_topview.jpeg')
                )
                
    @property
    def rx_quality(self):
        try:
            _rx_quality = lookup_table.loc[self.scanID]['Rx Quality'].item()
        except AttributeError:
            _rx_quality = lookup_table.loc[self.scanID]['Rx Quality']
        except ValueError:
            _rx_quality = lookup_table.loc[self.scanID]['Rx Quality'].iloc[0]
        except KeyError or IndexError:
            warnings.warn(f'Case with scan ID {self.scanID} is not present in the lookup table')
            return None
        return _rx_quality
    
    @property
    def bite_quality(self):
        if 'bite' in self.rx_quality:
            return False
        return True
        
    @property
    def margin_clarity(self):
        if 'noise' in self.rx_quality or 'margin' in self.rx_quality:
            return False
        return True
    
    @property
    def case_type(self):
        try:
            _type = lookup_table.loc[self.scanID]['Case Type'].item()
        except AttributeError:
            # means it is a string type already
            _type = lookup_table.loc[self.scanID]['Case Type']
        except ValueError:
            # means there are more than one values in the series
            _type = lookup_table.loc[self.scanID]['Case Type'].iloc[0]
        except KeyError or IndexError:
            # means scan ID is not present in lookup table
            warnings.warn(f'Case with scan ID {self.scanID} is not present in the lookup table')
            return None
        return _type
    
    @bite_quality.getter
    def has_good_bite(self):
        return self.bite_quality
    
    @margin_clarity.getter
    def has_clear_margin(self):
        return self.margin_clarity
    
    @case_type.getter
    def get_case_type(self):
        return self.case_type
            

if __name__ == "__main__":
    directory = r"E:\unsegregated"
    less_file = []
    no_lookup_value = []
    both_side_prep = []
    num_good_scans=0
    num_bad_scans=0
    for _, scan_folders_all,_ in os.walk(directory):
        break
    scan_folders = scan_folders_all.copy()
    for folder in scan_folders_all:
        scan_folders.remove(folder) # in order to enable checkpoint running in case error is encountered in between
        scan_folder = ScanFolder(os.path.join(directory,folder))
        if scan_folder.count_stl_files() < 2:
            less_file.append(scan_folder.scanID)
            continue
        if scan_folder.count_stl_files() >=4:
            both_side_prep.append(scan_folder.scanID)
            continue
        scan_object = ScanObject(scan_folder.scanID,
                                 scan_folder.bite_scan,
                                 scan_folder.preparation_scan)
        if scan_object.rx_quality is None:
            no_lookup_value.append(scan_object.scanID)
            continue
        bite_quality_of_this_scan = ('Good' if scan_object.has_good_bite 
                                     else 'Bad')
        if bite_quality_of_this_scan == 'Good':
            num_good_scans +=1
        else:
            num_bad_scans += 1
        if scan_object.get_case_type in ['Quadrant','Expanded']:
            scan_object.save_2_view_of_bite(
                    os.path.join(Destinations.bite_root,
                                 bite_quality_of_this_scan)
                    )
        elif scan_object.get_case_type == 'Full Arch':
            scan_object.save_3_view_of_bite(
                    os.path.join(Destinations.bite_root,
                                 bite_quality_of_this_scan)
                    )
        
            
            
        