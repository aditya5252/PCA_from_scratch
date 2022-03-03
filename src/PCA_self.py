
import numpy as np
from numpy import linalg as LA
def pca_scratch_high_dim(X,K,N):
    '''
    Here I apply High_Dimensional_PCA for dimensionality
    reduction.
    Function inputs N images each flattened to the dimension high_dim=10201 
    which gets reduced to dimension low_dim=K.
    X - N*high_dim
    K - low_dim
    N - no.of.samples
    
    function returns array of dim N*low_dim
    '''
    
    '''sample_dim_matrix is of dimensions N*N 
    Its Eigenvectors will be of dimension N*1
    Preserving top K principal components 
    ==> Keeping K eigenvectors corresponding to the top K eigenvalues
    '''
    mat_N=np.dot(X,X.T)/N
    eig_val,eig_vec_N=LA.eig(mat_N)
    increase_idx=np.argsort(eig_val)
    decrease_idx=np.flip(increase_idx)
    eig_vec_D_ls=[(  1/(N*eig_val[i])**0.5  )*np.dot(X.T,eig_vec_N[:,i]) for i in range(N)]
    eig_vec_D=np.stack(eig_vec_D_ls)
    K_feature_ls=[np.squeeze(np.dot(eig_vec_D[0:K],X[i].reshape(-1,1))) for i in range(N)]
    K_feature_arr=np.stack(K_feature_ls)
    return K_feature_arr

