
import numpy as np
def norm_highx_y_(img_list,img_targets):
    '''This function with return normalized features and the corresponding labels'''
    feature_len_long=np.prod(img_list[0].shape)
    feature_list=[np.reshape(img,feature_len_long) for img in img_list]
    feature_matrix=np.stack(feature_list)
    label_list=[0. if 'sad' in target else 1 for target in img_targets]
    Y=np.array(label_list)
    X=feature_matrix-feature_matrix.mean(0,keepdims=True)
    return X,Y



