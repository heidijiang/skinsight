import pandas as pd
import numpy as np
import warnings
from sksutils.sksutils import sigmoid, minmax, init_cats, bashdir2concerns
warnings.filterwarnings('ignore')

def confidence(ups, downs):

	'''
    Generate Wilson binomial confidence interval
    input:
    	ups: # positive
		downs: # negative
    '''
	n = ups + downs

	z = 1.96 #1.44 = 85%, 1.96 = 95%
	phat = ups / n
    
	return ((phat + z*z/(2*n) - z * np.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))


def sentiment_agg(avg_info,agg,name,agg_type='skin type'):

	'''
    Aggregate sentiment information for each product and skin type on a few dimensions
    input:
    	avg_info: df onto which successive aspect sentiment columns will be merged
    	agg:  initial aggregation of reviews by product
    	name: aspect
    	agg_type: whether to use skin type specific information
    '''
    
	avg_info = pd.merge(avg_info, 
	    agg.groupby('product_id')['{}_sentiment'.format(name)].mean().reset_index(),on='product_id',how='left')
	avg_info = pd.merge(avg_info,
	    agg.groupby('product_id').apply(lambda x: (x['{}_sentiment'.format(name)]>=.75).sum()).reset_index(name='{}_n_pos'.format(name)),on='product_id',how='left')
	avg_info = pd.merge(avg_info,
	    agg.groupby('product_id').apply(lambda x: (x['{}_sentiment'.format(name)]<=.25).sum()).reset_index(name='{}_n_neg'.format(name)),on='product_id',how='left')
	avg_info = pd.merge(avg_info,
	    agg.groupby('product_id')['{}_sentiment'.format(name)].count().reset_index('{}_n_mention'.format(name)),
	                    on=['product_id'],how='left')


	if agg_type == 'all':
	    avg_info['{}_wilson'.format(name)] = confidence(avg_info['{}_n_mention'.format(name)],avg_info['num_reviews'])
	else:
	    avg_info['{}_wilson'.format(name)] = confidence(avg_info['{}_n_mention'.format(name)],avg_info[name.split('_')[1]])

	col_sum = '{}_summary'.format(name)
	avg_info['{}_ratio'.format(name)] = avg_info['{}_n_pos'.format(name)] / (avg_info['{}_n_pos'.format(name)]+avg_info['{}_n_neg'.format(name)])
	avg_info[col_sum] = avg_info['{}_ratio'.format(name)] * avg_info['{}_wilson'.format(name)] * sigmoid(avg_info['{}_n_pos'.format(name)])

	avg_info[col_sum] = minmax(np.sqrt(avg_info[col_sum]))
	return avg_info


def init_process(file,bert_path, save=True):
	
	'''
    Pull in sentiment from bert output
    '''
	
	df_all = pd.read_csv('{}/db_aspect_sentences.csv'.format(file))

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

	if save:
		df_all.to_csv('{}/db_aspect_sentences_bert.csv'.format(file), index=False)

	return df_all

def gen_knowledge_model(path, bert_path, save=True):

	'''
    Add sentiment scores, aggregate information by product and skin type for each skin concern
    '''

	df_reviews = pd.read_csv('{}/db_reviews.csv'.format(path))
	df_reviews['skin_type'] = df_reviews['skin_type'].fillna('none')

	df_sum = pd.read_csv('{}/db_summary.csv'.format(path))

	df_all = init_process(path,bert_path)
	df_all['skin_type'] = df_all['skin_type'].fillna('none')

	cols = init_cats('concerns')

	avg_info = df_sum[['product_id','num_reviews']]

	agg = df_reviews.groupby(['product_id','skin_type'])['rating'].count().reset_index().pivot(index='product_id',columns='skin_type',values='rating').reset_index()
	avg_info = pd.merge(avg_info,agg,on='product_id',how='left')

	for i,c in enumerate(cols):
	    agg = df_all[df_all['{}_match'.format(c)]].groupby(['product_id','user_name'])['{}_sentiment'.format(c)].mean().reset_index()
	    
	    avg_info = sentiment_agg(avg_info,agg,c,'all')

	    
	    for sk in df_all['skin_type'].unique():
	        name = '{}_{}'.format(c,sk)
	        agg = df_all[(df_all['{}_match'.format(c)]) & (df_all['skin_type']==sk)].groupby(['product_id','user_name'])['{}_sentiment'.format(c)].mean().reset_index()
	        agg = agg.rename(columns={'{}_sentiment'.format(c): '{}_sentiment'.format(name)})
	        avg_info = sentiment_agg(avg_info,agg,name)
	
	df_sum = pd.merge(df_sum,avg_info,on=['product_id','num_reviews'])

	if save:
		df_sum.to_csv('{}/db_kbm.csv'.format(path), index=False)
	
	print("KB model generated")
	return df_sum

