from src.import_file import *

def FiscLDA(X_low,Y):
    happy_index=np.where(Y==1)[0]
    sad_index=np.where(Y==0)[0]

    ''' Dimensions
    X_happ = N_samples_happ x low_dim
    X_sad = N_samples_sad x low_dim
    '''
    X_happ=X_low[happy_index]
    X_sad=X_low[sad_index]

    mean_happ=np.mean(X_happ, axis=0)
    mean_sad=np.mean(X_sad,axis=0)

    Scatter_between=np.outer(mean_sad-mean_happ,mean_sad-mean_happ)

    Sw1=np.matmul((X_happ-mean_happ).T,(X_happ-mean_happ))/X_happ.shape[1]
    Sw2=np.matmul((X_sad-mean_sad).T,(X_sad-mean_sad))/X_sad.shape[1]
    Scatter_within=Sw1+Sw2
    Swb=np.matmul(LA.inv(Scatter_within),Scatter_between)
    Lval, Lvec = LA.eig(Swb)
    L_inc_idx=np.argsort(Lval)
    L_dec_idx=np.flip(L_inc_idx)
    W=Lvec[:,L_dec_idx][:,0:1].real

    Xhap1=np.matmul(X_happ,W)
    Xsad1=np.matmul(X_sad,W)
    thresh=(np.mean(Xhap1)+np.mean(Xsad1))/2

    plt.plot(Xhap1,'g')
    plt.plot(np.full(11,thresh),'b')
    plt.plot(Xsad1,'r')
    return 