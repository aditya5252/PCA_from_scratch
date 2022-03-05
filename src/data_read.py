'''Variable nomenclature improve'''
import glob
import matplotlib.pyplot as plt

def get_data_(train=True):
    state= 'train' if train==True else 'test'
    data = [(plt.imread(file),file) for file in glob.glob('data/emotion_classification/'+state+'/*.gif')]
    img_list=[]
    img_targets=[]
    for dat in data :
        img_list.append(dat[0])    
        img_targets.append(dat[1])
    return img_list,img_targets