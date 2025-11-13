import os
import yaml
import logging

class Mapper(object):
    def __init__(self, classes_file, simplify=False):
        
        self.simplify = simplify
        if not simplify:
            # Load the class mapping from a YAML file
            with open(classes_file, 'r') as cf:
                self.ml_project_classes = yaml.safe_load(cf)
        # we also accapt a class dict
        else:
            self.ml_project_classes = classes_file

    def map_aigle_classes_labels(self, x):
        """
        Internal mapping function for dataset version 1.1.
        """
        categories_mapping = {
            0: 'construction en dur',
            1: 'camping car',
            2: 'caravane',
            3: 'construction en dur',
            4: 'container',
            5: 'dechet',
            6: 'installation legere',
            7: 'mobil home',
            8: 'navire',
            9: 'panneau photovoltaique',
            10: 'piscine',
            11: 'pilone electrique',
            12: 'reservoir d eau',
            13: 'tunnel agricole serre',
            14: 'station d epuration',
            15: 'remblais',
            16: 'plan d eau',
            17: 'broussailles a risque'
        }
        return categories_mapping.get(x, 'Unknown class')
    
    def simplify_flair_classes_app(self,x):
        """
        mapping to min required aigles detections classes, ie it simplify classif and remove (-1) unwanted classes
            0: 'construction en dur',
            1: 'camping car',
            2: 'caravane',
            3: 'construction en dur',
            4: 'container',
            5: 'dechet',
            6: 'construction legere yourte etc.',
            7: 'mobil home',
            8: 'navire',
            9: 'panneau photovoltaique',
            10: 'piscine',
            11: 'pilone electrique',
            12: 'reservoir d eau',
            13: 'tunnel agricole serre',
            14: 'station d epuration',
            15: 'remblais',
        """
        categories_mapping = {
            0: -1,
            1: -1,
            2: -1,
            3: -1,
            4: -1,
            5: -1,
            6 : 16,
            7: -1,
            8: -1,
            9: -1,
            10: -1,
            11: -1,
            12: -1,
            13: -1,
            14: -1,
            15: -1,
            16: -1,
            17: -1,
            18: -1
                
        }
        return categories_mapping[x]