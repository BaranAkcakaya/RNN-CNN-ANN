# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:15:29 2020

@author: lenovoz
"""

import inception
import os

inception.download()
model = inception.Inception()

def classify(image_path):
    pred = model.classify(image_path = image_path)
    model.print_scores(pred = pred, k = 5, only_first_name = True)
    
dosya = os.listdir("IncepDeneme/")

classify(image_path = "IncepDeneme/zebra.jpg") 
