import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from math import pi
from sklearn.metrics.pairwise import cosine_similarity
from sksutils.sksutils import init_cats

def radar_plot(cat,values,concerns,m):

	N = len(cat)
	x_as = [n / float(N) * 2 * pi for n in range(N)]
	x_as += x_as[:1]

	values += values[:1]
	concerns += concerns[:1]
	
	plt.rc('axes', linewidth=5, edgecolor="#19212d")
	plt.rc('xtick', labelsize=16)
	ax = plt.subplot(polar=True)

	ax.set_theta_offset(pi / 2)
	ax.set_theta_direction(-1)
	ax.set_rlabel_position(0)

	expand = m[1]+(m[1]-m[0])/5

	ax.xaxis.grid(False)
	ax.yaxis.grid(True,color="#19212d", linestyle='solid', linewidth=1)
	plt.xticks(x_as[:-1], ['' for i in cat],color='w')
	plt.yticks(np.linspace(m[0],m[1],4), ['' for i in range(4)])
	ax.plot(x_as, concerns, linewidth=0, zorder=3)
	ax.fill(x_as, concerns, color="#969cba", alpha=1)

	ax.plot(x_as, values, linewidth=2, color='#19212d',linestyle='solid', zorder=3)
	ax.fill(x_as, values, color="#ed9382", alpha=0.75)
	plt.ylim(m[0],m[1])
	ax.set_facecolor(("#efe0ce"))

	for i in range(N):
	    angle_rad = i / float(N) * 2 * pi

	    if angle_rad == 0:
	        ha, distance_ax = "center", expand
	    elif 0 < angle_rad < pi:
	        ha, distance_ax = "left", expand
	    elif angle_rad == pi:
	        ha, distance_ax = "center", expand
	    else:
	        ha, distance_ax = "right", expand
	    ax.text(angle_rad, distance_ax, cat[i], size=16, horizontalalignment=ha, verticalalignment="center", fontname='Andale Mono')

	# Show polar plot

	img = io.BytesIO()
	plt.savefig(img, format='png')
	img.seek(0)
	graph_url = base64.b64encode(img.getvalue()).decode()
	plt.close()
	return 'data:image/png;base64,{}'.format(graph_url)

def minmax(df):
	n_min = np.nanmin(df,axis=0)
	n_max = np.nanmax(df,axis=0)

	df = (df-n_min)/(n_max-n_min)
	return df

def weight_models(l):
	w_init = .5
	w = w_init * (1/(1+np.exp(-l)))
	return w

def get_price(x):

	cats = init_cats('price sensitivity')
	if x == cats[0]:
		y = 0
	elif x == cats[1]:
		y = .5
	else:
		y = 1
	return y

def get_collab_imgs(file,vals,n_disp):
	cols = ['product_id','brand','name','product_image_url']
	df = pd.read_csv(file)
	rev_pct = np.percentile(df['num_reviews'],50)

	df = df.loc[~(df[vals['product'].lower()]) & (df['num_reviews']>=rev_pct),
		cols]
	df = df[cols].sample(n_disp).T.to_dict()
	return df

def get_recs(file_content,file_item,vals,Q,n,iids_user):
	
	df = pd.read_csv(file_content)

	# content
	concerns = np.array([int(i) for i in vals['concerns']])/100
	concern_names = ['{}_{}_summary'.format(c,vals['skin'].lower()) for c in init_cats('concerns')]


	df[concern_names] = minmax(df[concern_names])

	# add 7th dim (price, not included in radar plot!)
	tmp = df[concern_names].copy()
	tmp['price_scaled'] = 1 - minmax(df['price_num']) 
	price = get_price(vals['price'])

	# get similarity btwn content model and user
	df['content_sim'] = np.array(tmp).dot((np.append(concerns,price)).T)
	df['content_sim'] = minmax(df['content_sim'])

	# now do collab filt
	iids_user = list(iids_user.to_dict().keys())
	if len(iids_user)>0:
		sim = np.genfromtxt(file_item)
		user_idx = np.array(df[df['product_id'].isin(iids_user)].index)

		sim[sim==0]=np.nanmin(sim[sim>0])

		sim_log = np.log10(sim)
		sim_z = minmax(sim_log)
		sim_mean = np.nanmean(sim_z[:,user_idx],axis=1)
		sim_mean_z = minmax(sim_mean)

		df['item_sim'] = sim_mean_z

		print(df['item_sim'])
		w = weight_models(len(user_idx))
		df['final_rec'] = w*df['item_sim'] + (1-w)*df['content_sim']
	else:
		df['final_rec'] = df['content_sim']

	prod = vals['product'].lower()
	df = df[df[prod]].reset_index(drop=True)
	
	df_ranked = df.sort_values('final_rec',ascending=False).reset_index(drop=True).iloc[0:n]
	m = [0,1]

	img_files = [radar_plot(Q['concerns'],df_ranked[concern_names].iloc[i].to_list(),list(concerns),m) for i in range(n)]


	df_ranked['img_file'] = img_files
	df_ranked['idx'] = list(range(1,n+1))
	products = df_ranked[['idx','url','product_image_url','name','brand','price','img_file']].T.to_dict()
	products = [j for i,j in products.items()]

	return products

