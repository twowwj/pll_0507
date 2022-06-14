#
# i = 0
# start = 0.999
# end = 0.8
#
# for epoch in range(25):
#    print("{:.02f}".format(0.4 * epoch/ 25 * (end - start) + start), end=',')
# print()
# for epoch in range(25):
#    print("{:.02f}".format(0.4 * (25 - epoch) / 25 * (end - start) + start), end=',')

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import pickle
import os
from pathlib import Path
import numpy as np
import re


def get_img_list(dir, imagelist, ext=None):
   if os.path.isdir(dir):
      for s in os.listdir(dir):
         newdir = os.path.join(dir, s)
         get_img_list(newdir, imagelist, ext)
   elif os.path.isfile(dir):
      if ext is None:
         imagelist.append(dir)
      elif ext in dir[-3:]:
         imagelist.append(dir)
   return imagelist


ferplus = '/data0/wwang/pll_0507/data/alignment3/FER2013Train'
csv_path = '/data0/wwang/pll_0507/data/alignment3/fer2013new.csv'

imgs = [f.split("/")[-1] for f in get_img_list(ferplus, [], ext='png')]

affectnet_partialY = pickle.load(open('/data0/wwang/pll_0507/labelset_ferplus/confi.pkl', "rb"))

with open(csv_path, mode='r') as f:
   reader = csv.reader(f)
   header = next(reader)
   count = 0
   label_dict = {}

   for row in reader:
      labels = np.array(row[2:10]).astype(int)
      for img in imgs:
         if row[1] in img:

            labels = np.array(row[2:10]).astype(int).tolist()
            L = np.sum(labels)
            if not L == 0:
               confidence = labels / L
               label_dict[img] = confidence.tolist()
               count += 1
               print(count)
            else:
               pass

   with open('/data0/wwang/pll_0507/labelset_ferplus/confi.pkl', 'wb') as ff:
      pickle.dump(label_dict, ff)
      print(count)



