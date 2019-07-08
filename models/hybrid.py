import pandas as pd 
import numpy as np
from sksutils.sksutils import *
from scipy.spatial import distance

class Hybrid:

	def __init__(self):

		self.items = False
		self.num_items = 0
		self.data = pd.read_csv('data/kbm_cf.csv')

		self.w_init = .5
		self.cf_display = 40
		self.rank_display = 10


	def user_input(self,results):

		self.product = results['product type']
		self.skin = results['skin type']
		self.price = results['price sensitivity']
		self.concerns = np.array([int(results[i]) for i in init_cats('concerns')])/100


	def gen_cf_sample(self,product_type):

		cols = ['product_id','brand','name','product_image_url']

		rev_pct = np.percentile(self.data['num_reviews'],50)

		df = self.data.loc[~(self.data[product_type]) & (self.data['num_reviews']>=rev_pct),cols]
		df = df[cols].sample(self.cf_display)

		return df


	def set_item_history(self, items):

		if len(items)>0:
			self.item_history = items
			self.items = True
			self.num_items = len(items)


	def concern_cols(self):
		if self.use_skintype:
			return ['{}_{}_summary'.format(c,self.skin.lower()) for c in init_cats('concerns')]

		return ['{}_summary'.format(c) for c in init_cats('concerns')]


	def KBM(self, norm, use_skintype):
		self.use_skintype = use_skintype
		concern_names = self.concern_cols()

		# add 7th dim (price, not included in radar plot!)
		tmp = self.data[concern_names].copy()
		tmp['price_scaled'] = 1 - minmax(self.data['price_num']) 
		price = get_price(self.price)

		# get similarity btwn content model and user
		self.data['kbm'] = np.array(tmp).dot((np.append(self.concerns,price)**2).T)
		# self.data['kbm'] = tmp.apply(lambda x: distance.euclidean(x,np.append(self.concerns,price)),axis=1)
		# self.data['kbm'] = self.data['kbm'].max() - self.data['kbm']
		if norm == 'minmax':
			self.data['kbm_norm'] = minmax(self.data['kbm'])



	def CF(self, norm):
		# now do collab filt
		
		if self.items:
			
			self.data['cf'] = np.nanmean(self.data[self.item_history],axis=1)

			if norm == 'minmax':
				self.data['cf_norm'] = minmax(self.data['cf'])


	def gen_ranks(self):

		cols = ['name','brand','price','final_rec','product_image_url','url']
		cols.extend(self.concern_cols())

		w = weight_models(self.num_items,self.w_init)

		if self.items:
			self.data['final_rec'] = w*self.data['cf_norm'] + (1-w)*self.data['kbm_norm']

		else:
			self.data['final_rec'] = self.data['kbm_norm']

		self.output = self.data.loc[self.data[self.product],cols].sort_values('final_rec',ascending=False)
		self.output = self.output.reset_index(drop=True).reset_index().iloc[0:self.rank_display]
		self.output['index'] += 1
		print(self.output[self.concern_cols()])
		print(self.concerns)

	def add_radar(self):

		Q = gen_Q()
		df_user = self.output[self.concern_cols()]
		self.output['img_file'] = [radar_plot(Q['concerns'],df_user.iloc[i].to_list(),list(self.concerns)) for i in range(self.rank_display)]



