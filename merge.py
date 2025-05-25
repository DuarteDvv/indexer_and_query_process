import os
import json

def merge_partial_indexes(index_path):
   """
   Merges all partial indexes into a single index file.
   """
   
   # adquire all names of the partial indexes
   partial_indexes_names = os.listdir(index_path) 
   
   print(partial_indexes_names)
   


merge_partial_indexes("./index/")