#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 12:10:43 2018

@author: richardolij
"""

#from read_data import get_data
import sys;
sys.path.append('../data_preprocessing/')
import read_data

data_loader = read_data.get_data()
data_loader