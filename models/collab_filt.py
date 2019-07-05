import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class KnnRecommender:
    
    def __init__(self,names,data):

        self.model = NearestNeighbors()
        self.names = 'skin_user_item_pid.csv'
        self.ratings = 'skin_user_item.csv'

    def set_model_params(self, n_neighbors, algorithm, metric, n_jobs=None):

        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algorithm,
            'metric': metric,
            'n_jobs': n_jobs})

    def _inference(self, model, data, n_recommendations):
        
        model.fit(data)
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)

        raw_recommends = sorted(list(zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist())),
                key=lambda x: x[1])[:0:-1]

        return raw_recommends

    def make_recommendations(self, names, n_recommendations):


        raw_recommends = self._inference(
            self.model, X, n_recommendations)
        reverse_hashmap = [np.argwhere(l==[i]).flatten()[0] for i in names]


def sparsity(iu):
    sparsity = iu[iu>0].sum()
    sparsity /= (iu.shape[0] * iu.shape[1])
    return sparsity *= 100


def item_user(file):
    df = pd.read_csv('{}/skin_reviews.csv'.format(file))
    df = thresh_rm(df,['user_name'],20)

    df = df[['product_id','user_name','rating']]
    df_iu = df.groupby(['product_id','user_name'])['rating'].last().reset_index()

    df_piv = pd.pivot(data=df_iu,columns='product_id',index='user_name',values='rating').fillna(0)
    df_piv = df_piv.reset_index(drop=True)

    df_piv[df_piv==0]=np.nan

    df_piv = df_piv.sub(df_piv.mean(axis=1),axis=0).divide(df_piv.std(axis=1),axis=0)
    df_piv += -(df_piv.min().min()-1)
    df_piv = df_piv.fillna(0)

    return df_piv

def gen_collab_model(df):
    X = np.array(df_piv.T)

    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model.fit(X)

    dist_matrix = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        distances, indices = model.kneighbors(X[i,:][np.newaxis], n_neighbors = X.shape[0])
        dist_matrix[:,i] = distances[:,np.argsort(indices)]
        dist_matrix[i,i] = np.nan


    sim = 1-sim_matrix
    sim[sim==0]=np.nanmin(sim[sim>0])
    sim_log = np.log10(sim)
    sim_z = (sim_log-np.nanmean(sim_log,axis=1))/np.nanstd(sim_log,axis=1)
    df_sim = pd.DataFrame(sim_z,columns=df_piv.columns)

    df_sim.to_csv('{}/item_collab_filt.csv'.format(file),index=False)

    return df_sim

if __name__ == '__main__':
    gen_collab_model('~/Documents/insight/skinsight')

