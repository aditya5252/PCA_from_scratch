'''Variable nomenclature improve'''
import glob
import matplotlib.pyplot as plt
data = [(plt.imread(file),file) for file in glob.glob('..\data\*.gif')]
img_list=[]
img_targets=[]
for dat in data :
    img_list.append(dat[0])    
    img_targets.append(dat[1])
