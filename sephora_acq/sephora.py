import json
import csv
import requests
import pandas as pd
import numpy as np

class SephoraAPIFetch:

    
    def __init__(self):
        self.base_url = 'https://www.sephora.com/api/catalog'
        self.product_url = '{}/categories/cat150006/products?currentPage=0&pageSize=1000&content=true&includeRegionsMap=true'.format(self.base_url)


    def product_list(self,s):

        '''
        Gets all the products across categories, and some general summary info. Returns dataframe.
        '''

        url = '{}/categories/{}/products?currentPage=0&pageSize=999999999&content=true&includeRegionsMap=true'.format(self.base_url,s['id'])
        r = requests.get(url)
        content = json.loads(r.content)
        D = dict()
        D['brand'], D['name'], D['product_id'], D['rating'], D['url'], D['price'] = [], [], [], [], [], []
        for i in content['products']:
            D['brand'].append(i['brandName'])
            D['name'].append(i['displayName'])
            D['product_id'].append(i['productId'])
            D['rating'].append(i['rating'])
            D['url'].append(i['targetUrl'])
            D['price'].append(i['currentSku']['listPrice'])
        df = pd.DataFrame(D)
        df['category'] = s['name']
        return df


    def review_url(self,product_id,offset=0):

        '''
        Updates product review API url with offset
        '''
        
        return 'https://api.bazaarvoice.com/data/reviews.json?Filter=ProductId%3A{}&Sort=Helpfulness%3Adesc&Limit=100&Offset={}&Include=Products%2CComments&Stats=Reviews&passkey=rwbw526r2e7spptqd2qzbkp7&apiversion=5.4'.format(product_id,offset)
    
    
    def req(self, url):

        r = requests.get(url)
        return json.loads(r.content)
    
    
    def query_summary(self):

        '''
        Fetch product category
        '''

        cat_list = [4,5,6,8,9,11]
        
        j = self.req(self.product_url)
        
        self.df = pd.DataFrame()
        
        for i in cat_list:
            sub_dict = j['categories'][0]['subCategories'][i]

            D = {'id': sub_dict['categoryId'], 'name': sub_dict['displayName'].lower()}

            self.df = self.df.append(self.product_list(D),ignore_index=True)
            self.df = self.df.drop_duplicates(subset='product_id').reset_index(drop=True)

        print('Summary db fetched')
        
    
    def query_reviews(self):

        '''
        Wrapper for fetching reviews
        '''
        
        self.df_reviews = pd.DataFrame()
        
        for index, row in self.df.iterrows():
            self.get_reviews(row['product_id'])
            print('.',end=" ")

        print('Review db fetched')
            
   
    def get_reviews(self, product_id):

        '''
        Fetch reviews, 100 at a time
        '''
        
        url = self.review_url(product_id)
        init = self.req(url)
        
        try:
            base = init['Includes']['Products'][product_id]
        except:
            tmp = j['Includes']['Products']
            base = next(iter(tmp.values()))
            
        D = dict()
        D['product_id'], D['product_image_url'], D['description'] = [],[],[]
        D['user_name'], D['rating'], D['review_text'], D['skin_type'], D['skin_concerns'] = [], [], [], [], []
        
        for offset in range(0,init['TotalResults'],100):

            try:
            
                data = self.req(self.review_url(product_id, offset))

                for review in data['Results']:
                    
                    D['product_id'].append(product_id)
                    D['product_image_url'].append(base['ImageUrl'])
                    D['description'].append(base['Description'])
                    
                    D['user_name'].append(review['UserNickname'])
                    D['rating'].append(review['Rating'])
                    D['review_text'].append(review['ReviewText'])
                    
                    try:
                        D['skin_type'].append(review['ContextDataValues']['skinType']['Value'])
                    except:
                        D['skin_type'].append(np.nan)
                    try:
                        D['skin_concerns'].append(review['ContextDataValues']['skinConcerns']['Value'])
                    except:
                        D['skin_concerns'].append(np.nan)
            except:
                continue

        self.df_reviews = self.df_reviews.append(pd.DataFrame(D), ignore_index=True)


def init_api(path):
    s = SephoraAPIFetch()
    s.query_summary()
    s.query_reviews()
    s.df.to_csv('{}/db_summary_raw.csv'.format(path), index=False)
    s.df_reviews.to_csv('{}/db_reviews_raw.csv'.format(path),index=False)

if __name__ == '__main__':
    init_api()
