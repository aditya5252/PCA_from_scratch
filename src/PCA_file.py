
import numpy as np
from numpy import linalg as LA

class PCA_scratch:
    def __init__(self,n_components):
        self.K=n_components
    def fit(self,X):
        N=X.shape[0]
        self.N=N
        if(self.K>N):
            raise ValueError('no.of.components should be less than no. of samples')
        mat_N=np.dot(X,X.T)/N
        eig_val,eig_vec_N=LA.eig(mat_N)
        increase_idx=np.argsort(eig_val)
        decrease_idx=np.flip(increase_idx)
        eig_vec_D_ls=[(  1/(N*eig_val[i])**0.5  )*np.dot(X.T,eig_vec_N[:,i]) for i in range(self.N)]
        eig_vec_D=np.stack(eig_vec_D_ls)
        self.eig_vec_D=eig_vec_D
    def fit_predict(self,X):
        self.fit(X)
        K_feature_ls=[np.squeeze(np.dot(self.eig_vec_D[0:self.K],X[i].reshape(-1,1))) for i in range(self.N)]
        K_feature_arr=np.stack(K_feature_ls)
        return K_feature_arr
    def predict(self,Xtest):
        K_feature_ls=[np.squeeze(np.dot(self.eig_vec_D[0:self.K],Xtest[i].reshape(-1,1))) for i in range(self.N)]
        K_feature_arr=np.stack(K_feature_ls)
        return K_feature_arr