import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sksutils.sksutils import thresh_rm

def sparsity(iu):

    '''
    Get item user matrix sparsity
    '''

    sparsity = iu[iu>0].sum()
    sparsity /= (iu.shape[0] * iu.shape[1])
    sparsity *= 100
    return sparsity


def item_user(file, normalize=True):

    '''
    generate item user matrix from review dataframe
    '''

    df = pd.read_csv('{}/db_reviews.csv'.format(file))
    df = thresh_rm(df,['user_name'],20)

    df = df[['product_id','user_name','rating']]
    df_iu = df.groupby(['product_id','user_name'])['rating'].last().reset_index()

    urm = df_iu.pivot(columns='product_id',index='user_name',values='rating').fillna(0).reset_index(drop=True)

    if normalize:
        urm[urm==0]=np.nan
        urm = urm.sub(urm.mean(axis=1),axis=0).divide(urm.std(axis=1),axis=0)
        # urm += -(urm.min().min()-1)
        urm = urm.fillna(0)

    return urm

def gen_cf(path, save=True):

    '''
    Get item-item cosine similarity
    '''

    urm = item_user(path)
    X = np.array(urm.T)

    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model.fit(X)

    dist_matrix = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        distances, indices = model.kneighbors(X[i,:][np.newaxis], n_neighbors = X.shape[0])
        dist_matrix[:,i] = distances[:,np.argsort(indices)]
        dist_matrix[i,i] = np.nan


    sim = 1-dist_matrix
    sim[sim==0]=np.nanmin(sim[sim>0])
    sim_log = np.log10(sim)
    sim_z = (sim_log-np.nanmean(sim_log,axis=1))/np.nanstd(sim_log,axis=1)
    df_sim = pd.DataFrame(sim_z,columns=urm.columns)

    if save:
        df_sim.to_csv('{}/db_cf.csv'.format(path),index=False)

    print('CF model built')
    return df_sim

if __name__ == '__main__':
    gen_cf('~/Documents/insight/skinsight')

