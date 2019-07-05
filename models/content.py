import pandas as pd
import numpy as np
import warnings
from app.sksutils.sksutils import *
warnings.filterwarnings('ignore')

def confidence(ups, downs):
	# Wilson confindence interval

    n = ups + downs

    z = 1.96 #1.44 = 85%, 1.96 = 95%
    phat = ups / n
    
    return ((phat + z*z/(2*n) - z * np.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sentiment_agg(avg_info,tmp,name,agg_type='skin type'):
    # get some metrics about sentiment across products

    avg_info = pd.merge(avg_info, 
        tmp.groupby('product_id')['{}_sentiment'.format(name)].mean().reset_index(),on='product_id',how='left')
    avg_info = pd.merge(avg_info,
        tmp.groupby('product_id').apply(lambda x: (x['{}_sentiment'.format(name)]>=.75).sum())
        .reset_index().rename(columns={0:'{}_n_pos'.format(name)}),on='product_id',how='left')
    avg_info = pd.merge(avg_info,
        tmp.groupby('product_id').apply(lambda x: (x['{}_sentiment'.format(name)]<=.25).sum())
        .reset_index().rename(columns={0:'{}_n_neg'.format(name)}),on='product_id',how='left')
    avg_info = pd.merge(avg_info,
        tmp.groupby('product_id')['{}_sentiment'.format(name)].count()
        .reset_index().rename(columns={'{}_sentiment'.format(name):'{}_n_mention'.format(name)}),
        on=['product_id'],how='left')
    
    
    if agg_type == 'all':
        avg_info['{}_wilson'.format(name)] = confidence(avg_info['{}_n_pos'.format(name)],avg_info['num_reviews'])
    else:
        avg_info['{}_wilson'.format(name)] = confidence(avg_info['{}_n_pos'.format(name)],avg_info[name.split('_')[1]])
        
    avg_info['{}_ratio'.format(name)] = avg_info['{}_n_pos'.format(name)] / 
    	(avg_info['{}_n_pos'.format(name)]+avg_info['{}_n_neg'.format(name)])
    
    avg_info['{}_summary'.format(name)] = avg_info['{}_ratio'.format(name)] * 
    	avg_info['{}_wilson'.format(name)] * sigmoid(np.log10(avg_info['{}_n_pos'.format(name)]))
    
    return avg_info


def init_process(file,bert_path):
	# pull in bert sentiment data
	
	df_all = pd.read_csv('{}/skin_relevant_sentences_summary.csv'.format(file))

	D = bashdir2concerns()
	ttype = 'test'

	for aspect,a2 in D.items():
	    
	    name = '{}_sentiment'.format(a2)
	    basedir = '{}/{}/data'.format(bert_path,aspect.replace(' ','_'))
	    results = pd.read_csv('{0}_{1}/{1}_results.tsv'.format(basedir,ttype),sep='\t',header=-1).rename(columns={1:name}).reset_index()
	    test = pd.read_csv('{0}_{1}/{1}.tsv'.format(basedir,ttype),sep='\t')

	    idx = pd.read_csv('{0}_{1}/test_idx.csv'.format(basedir,ttype),header=-1).rename(columns={0:'index',1:'sample_index'})
	    bert = pd.merge(test,pd.merge(idx,results[['index',name]],on='index'),on='index')
	    df_all = pd.merge(df_all,bert[['sample_index',name]],on='sample_index',how='left')

	return df_all

def gen_content_model(file, bert_path):

	df_reviews = pd.read_csv('{}/skin_reviews.csv'.format(file))
	df_reviews['skin_type'] = df_reviews['skin_type'].fillna('none')

	df_sum = pd.read_csv('{}/skin_summary.csv'.format(file))

	df_all = init_process(file,bert_path)
	df_all['skin_type'] = df_all['skin_type'].fillna('none')

	cols = init_cats('concerns')

	avg_info = df_sum[['product_id','num_reviews']]

	tmp = df_reviews.groupby(['product_id','skin_type'])['rating'].count()
		.reset_index().pivot(index='product_id',columns='skin_type',values='rating').reset_index()
	avg_info = pd.merge(avg_info,tmp,on='product_id',how='left')

	for i,c in enumerate(cols):
	    tmp = df_all[df_all['{}_match'.format(c)]].groupby(['product_id','user_name'])['{}_sentiment'.format(c)].mean().reset_index()
	    
	    avg_info = sentiment_agg(avg_info,tmp,c,'all')

	    
	    for sk in df_all['skin_type'].unique():
	        name = '{}_{}'.format(c,sk)
	        tmp = df_all[(df_all['{}_match'.format(c)]) & (df_all['skin_type']==sk)].groupby(['product_id','user_name'])['{}_sentiment'.format(c)].mean().reset_index()
	        tmp = tmp.rename(columns={'{}_sentiment'.format(c): '{}_sentiment'.format(name)})
	        avg_info = sentiment_agg(avg_info,tmp,name)
	
	df_sum = pd.merge(df_sum,avg_info,on=['product_id','num_reviews'])

	df_sum.to_csv('content_model.csv', index=False)
	return df_sum





if __name__ == '__main__':
	gen_content_model('~/Documents/insight/skinsight','~/Documents/insight/skinsight/bert/bert_final/results')

