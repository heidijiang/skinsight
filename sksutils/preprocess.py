import pandas as pd
import requests
import json
from sksutils.sksutils import init_cats, thresh_rm

def preprocess(path):
    
    print('Initializing data cleaning...')
    df = pd.read_csv('{}/db_reviews_raw.csv'.format(path))
    df_sum = pd.read_csv('{}/db_summary_raw.csv'.format(path))

    df = df.dropna(subset=['review_text']).reset_index(drop=True)
    df_sum['price_num'] = df_sum['price'].map(lambda x: x.lstrip('$').split(" ")[0]).astype('float')
    df_sum['url'] = 'https://www.sephora.com'+df_sum['url']
    df_sum = pd.merge(df_sum,df.groupby('product_id')['rating'].count().reset_index().rename(columns={'rating':'num_reviews'}),on='product_id',how='left')
    df_sum = pd.merge(df_sum,df.groupby('product_id')['rating'].std().reset_index().rename(columns={'rating':'rating_std'}),on='product_id',how='left')

    df = df.drop_duplicates(subset=['product_id', 'user_name']).reset_index().drop(['index'],axis=1)

    skin_type = ['Normal','Oily','Dry','Combination']
    for s in skin_type:
        df['product_skin_type_{}'.format(s)] = df['description'].str.contains(str('✔ '+s))

    C = {'acne':'Acne', 'aging':'Wrinkles', 'blackheads':'Acne', 'darkCircles':'Dark Spots', 'dullness':'Texture/Pores', 
     'pores':'Texture/Pores', 'redness':'Redness', 'sensitivity':'Sensitivity', 'sunDamage':'Dark Spots', 
     'unevenSkinTones':'Texture/Pores'}

    df['skin_concerns'] = df['skin_concerns'].replace(C)

    df = thresh_rm(df,['product_id'],30)
    df_sum = df_sum[df_sum['product_id'].isin(df['product_id'].unique())].reset_index(drop=True)

    df_sum = get_true_cats(df_sum)

    df = pd.merge(df,df_sum[['product_id','Cleanser','Moisturizer','Treatment','Mask','Sunscreen','Eye']],on=['product_id'],how='left')
    
    df.to_csv('{}/db_reviews.csv'.format(path),index=False)
    df_sum.to_csv('{}/db_summary.csv'.format(path), index=False)

    print('Initial data cleaning finished')
    return df, df_sum


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
         'Sunscreen': ['Face_Sunscreen'],
         'Eye': ['Eye_Creams_&_Treatments', 'Eye_Masks']
        }

    for key,vals in C.items():
        df[key] = df[vals].astype(bool).any(axis=1)
        
    exc = [j for i in C.values() for j in i]
    exc.extend(['Value_&_Gift_Sets', 'Mini_Size', 'Acne_&_Blemishes', 'Anti-Aging',
           'Dark_Spots', 'Pores', 'Dryness', 'Fine_Lines_&_Wrinkles', 'Dullness',
           'Decollete_&_Neck_Creams', 'Blotting_Papers', 'Body_Sunscreen',
           'After_Sun_Care'])

    df = df.drop(exc,axis=1).reset_index(drop=True)
    
    return df
