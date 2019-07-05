import pandas as pd
import numpy as np
import requests
import json
from sksutils import init_cats

def init_process(file,cats):
    

    df = pd.DataFrame()
    for cat in cats:
        df_tmp = pd.read_csv('{}/sephora_review_db_{}.csv'.format(file,cat))
        if cat[-1]=='s':
            cat = cat[:-1]
        df_tmp[cat] = 1
        df_tmp[cat] = df_tmp[cat].fillna(0)
        df = df.append(df_tmp, ignore_index = True) 

    ages = {'13to17':'15','18to24':'21','25to34':'30','35to44':'40','45to54':'50','over54':'60'}
    df['age'] = df['age'].replace(ages)
    df['price_num'] = df['price'].map(lambda x: x.lstrip('$').split(" ")[0]).astype('float')
    df['recommended'] = df['recommended'].astype('bool').astype('int64')

    df = df.drop_duplicates(subset=['product_id', 'user_name'],keep='last').reset_index().drop(['index'],axis=1)
    
    df = thresh_rm(df,['product_id'],20)
        
    skin_type = ['Normal','Oily','Dry','Combination']
    for s in skin_type:
        df['product_skin_type_{}'.format(s)] = df['description'].str.contains(str('✔ '+s))
    
    C = {'acne':'Acne', 'aging':'Wrinkles', 'blackheads':'Acne', 'darkCircles':'Dark Spots', 'dullness':'Texture/Pores', 
     'pores':'Texture/Pores', 'redness':'Redness', 'sensitivity':'Sensitivity', 'sunDamage':'Dark Spots', 
     'unevenSkinTones':'Texture/Pores'}
    
    df['skin_concerns'] = df['skin_concerns'].replace(C)
        
    return df


def add_cats(df,s,i):
    url = str('https://www.sephora.com/api/catalog/categories/'+s[0]+'/products?currentPage=0&pageSize=999999999&content=true&includeRegionsMap=true')
    r = requests.get(url)
    j2 = json.loads(r.content)
    prod_list = [i['productId'] for i in j2['products']]
    
    cat_str = s[1].replace(' ','_')
    df[cat_str] = 0
    df.loc[df['product_id'].isin(prod_list),cat_str] = 1
    
    return df
    
    
def get_true_cats(df):
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
            df = add_cats(df,s,i)

    exc_idx = df[(df['Body_Sunscreen']==1) | (df['After_Sun_Care']==1) | (df['Value_&_Gift_Sets']==1) | (df['Mini_Size']==1)].index
    df = df.drop(exc_idx).reset_index(drop=True)
    
    C = {'Cleanser': ['Makeup_Removers','Face_Wash_&_Cleansers','Face_Wipes','Toners'],
        'Moisturizer':['Moisturizers','Night_Creams','Face_Oils','Mists_&_Essences','BB_&_CC_Creams'],
        'Treatment': ['Exfoliators','Face_Serums','Blemish_&_Acne_Treatments','Facial_Peels'],
         'Mask': ['Face_Masks','Sheet_Masks'],
         'Sunscreen': ['Face_Sunscreen']
        }
    
    for key,vals in C.items():
        df[key] = df[vals].astype(bool).any(axis=1)
    df['Eye'] = df['eye'].fillna(0).astype(bool)
    
    df = df[['brand', 'name', 'brand_id', 'brand_image_url', 'product_id',
       'product_image_url', 'rating', 'skin_type', 'eye_color',
       'skin_concerns', 'incentivized_review', 'skin_tone', 'age',
       'beauty_insider', 'user_name', 'review_text', 'price','price_num', 'recommended',
       'first_submission_date', 'last_submission_date', 'location',
       'description', 'Cleanser', 'Moisturizer', 'Treatment', 'Eye',
       'Sunscreen', 'Mask','product_skin_type_Normal', 'product_skin_type_Oily','product_skin_type_Dry', 'product_skin_type_Combination']]
    
    return df


def get_summary(df):
    df_means = df.groupby('product_id')['rating','recommended'].mean()
    df_sd = df.groupby('product_id')['rating'].std()
    user_cats = ['user_name','rating','skin_type','eye_color','skin_concerns','incentivized_review','skin_tone','age','review_text','recommended','first_submission_date','last_submission_date','location']
    df_sum = df.groupby('product_id').first().reset_index().drop(user_cats,axis=1)
    df_sum = df_sum.join(df_means,on='product_id')
    df_sum['rating_sd'] = df_sd.reset_index()['rating']
    df_sum['num_reviews'] = df.groupby('product_id').size().reset_index()[0]
    return df_sum



if __name__=='__main__':

	file = '~/Documents/insight/skinsight'
	cats = init_cats('product type')
	df = init_process(file,cats)
    df = get_true_cats(df)
	df_sum = get_summary(df)

	df.to_csv('skin_reviews.csv',index=False)
	df_sum.to_csv('skin_summary.csv',index=False)

	
