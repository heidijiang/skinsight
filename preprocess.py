import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import requests
import json

def load_data(file):)
	df = pd.read_csv('{}_cleaners.csv',file

	cols = df.columns.tolist()
	df['cleanser'] = 1

	for cat in ['moisturizers','treatments','eye','sunscreen','masks']:
	    df_tmp = pd.read_csv('{}_{}.csv'.format(cat))
	    if cat[-1]=='s':
	        cat = cat[:-1]
	    df_tmp[cat] = 1
	    df_tmp[cat].fillna(0)
	    df = pd.merge(df,df_tmp,on=cols,how='outer')

	df = df.drop_duplicates()
	ages = {'13to17':'15','18to24':'21','25to34':'30','35to44':'40','45to54':'50','over54':'60'}
	df['age'] = df['age'].replace(ages)
	df['price_num'] = df['price'].map(lambda x: x.lstrip('$').split(" ")[0]).astype('float')
	df['recommended'] = df['recommended'].astype('bool').astype('int64')
	return df



def add_cats(df,s):
    url = str('https://www.sephora.com/api/catalog/categories/'+s[0]+'/products?currentPage=0&pageSize=999999999&content=true&includeRegionsMap=true')
    r = requests.get(url)
    j2 = json.loads(r.content)
    prod_list = [i['productId'] for i in j2['products']]
    cat_str = s[1].replace(' ','_')
    df[cat_str] = 0
    df.loc[df['product_id'].isin(prod_list),cat_str] = 1
    return df
    

def df_cats(df):    
	url = 'https://www.sephora.com/api/catalog/categories/cat150006/products?'
	r = requests.get(url)
	j = json.loads(r.content)

	cat_list = [0,1,3,4,5,6,8,9,11]
	for i in cat_list:
	    sub_dict = j['categories'][0]['subCategories'][i]
	    if 'subCategories' in sub_dict:
	        subcat_list = [(k['categoryId'],k['displayName']) for k in sub_dict['subCategories']]
	    else:
	        subcat_list = [(sub_dict['categoryId'],sub_dict['displayName'])]
	    
	    for s in subcat_list:
	        df = add_cats(df,s)
	        
	exc_idx = df[(df['Body_Sunscreen']==1) | (df['After_Sun_Care']==1) | (df['Value_&_Gift_Sets']==1) | (df['Mini_Size']==1)].index
	df = df.drop(exc_idx)
	skin_type = ['Normal','Oily','Dry','Combination']
	for s in skin_type:
	    df[s] = df['description'].str.contains(str('✔ '+s))



	df = df.drop_duplicates(subset=['product_id', 'user_name'],keep='last').reset_index().drop(['index','Unnamed: 0'],axis=1)

	rm_thresh = [('user_name', 1),('product_id', 10)]
	for i in rm_thresh:
	    tmp = df[i[0]].value_counts().reset_index()
	    multi_list = tmp.loc[tmp[i[0]]>i[1],'index'].tolist()
	    df = df[df[i[0]].isin(multi_list)]

	df['cleanser'] = df[['Makeup_Removers','Face_Wash_&_Cleansers','Face_Wipes','Toners']].astype(bool).any(axis=1)
	df['moisturizer'] = df[['Moisturizers','Night_Creams','Face_Oils','Mists_&_Essences','BB_&_CC_Creams']].astype(bool).any(axis=1)
	df['treatment'] = df[['Exfoliators','Face_Serums','Blemish_&_Acne_Treatments','Facial_Peels']].astype(bool).any(axis=1)
	df['mask'] = df[['Face_Masks','Sheet_Masks']].astype(bool).any(axis=1)
	df['sunscreen'] = df['Face_Sunscreen'].astype(bool)
	df['eye'] = df['eye'].astype(bool) ### MUST FIX THIS

	types = ['Normal','Dry','Oily','Combination']
	for t in types:
	    df[t] = df[t].astype(bool)

	return df

if __name__=='__main__':

	
