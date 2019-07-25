import pandas as pd
from models.kbm import gen_knowledge_model
from models.cf import gen_cf


def hybridize(path_in,path_out,bert_path, save=True):

	'''
    combine collab filt and knowledge model
    input:
    	path_in: path to input data from large folder
    	path_out: output for hybrid model, accessible location
    	bert_path: path to bert data
    '''
	df_kbm = gen_knowledge_model(path_in,bert_path, save)
	df_cf = gen_cf(path_in, save)
	df_cf['product_id'] = df_cf.columns
	df_final = pd.merge(df_kbm,df_cf,on='product_id')


	if save:
		df_final.to_csv('{}/kbm_cf.csv'.format(path_out), index=False)

	return df_final
