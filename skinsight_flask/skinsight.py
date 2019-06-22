import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from math import pi
from sklearn.metrics.pairwise import cosine_similarity

def radar_plot(cat,values,m):

	N = len(cat)

	x_as = [n / float(N) * 2 * pi for n in range(N)]
	values += values[:1]
	x_as += x_as[:1]

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
	ax.plot(x_as, values, linewidth=2, color='#19212d',linestyle='solid', zorder=3)
	ax.fill(x_as, values, color="#19212d", alpha=0.3)
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


def weight_models(l):
	w_init = .5
	w = w_init * (1/(1+np.exp(-l)))
	return w


def get_collab_imgs(file,vals,n_disp):
	cols = ['product_id','brand','name','product_image_url']
	df = pd.read_csv(file)
	rev_pct = np.percentile(df['num_reviews'],50)

	df = df.loc[~(df[vals['product'].lower()]) & (df['num_reviews']>=rev_pct),
		cols]
	df = df[cols].sample(n_disp).T.to_dict()
	return df


def get_recs(file,vals,Q,n,iids_user):
	
	df = pd.read_csv(file)
	concerns = np.array([int(i)/100 for i in vals['concerns']])
	concern_names = [i for i in Q['concerns']]
	df[concern_names] = (df[concern_names]-df[concern_names].mean())/df[concern_names].std()
	df['content_sim'] = cosine_similarity(df[concern_names],concerns[:,np.newaxis].T).ravel() 
	#np.array(df[concern_names]).dot((concerns).T)
	# cosine_similarity(df[concern_names],concerns[:,np.newaxis].T).ravel() 
	df['content_sim'] = (df['content_sim']-df['content_sim'].mean())/df['content_sim'].std()

	# THE OLD WAY
	
	# concern_names = ['{}_final'.format(i) for i in Q['concerns']]
	# df[concern_names] = (df[concern_names].mean() - df[concern_names])/df[concern_names].std()
	# df['content_sim'] = np.array(df[concern_names]).dot((concerns).T)
	# df['content_sim'] = (df['content_sim'].mean() - df['content_sim'])/df['content_sim'].std()

	iids_user = list(iids_user.to_dict().keys())
	if len(iids_user)>0:
		sim = np.genfromtxt('skinsight_flask/static/data/item_collab_sim.csv')
		user_idx = np.array(df[df['product_id'].isin(iids_user)].index)

		sim[sim==0]=np.nanmin(sim[sim>0])

		sim_log = np.log10(sim)
		sim_z = (sim_log-np.nanmean(sim_log,axis=1))/np.nanstd(sim_log,axis=1)
		sim_mean = np.nanmean(sim_z[:,user_idx],axis=1)
		sim_mean_z = (sim_mean-sim_mean.mean())/sim_mean.std()
		# item_ranked = np.argsort(sim_mean_z)[::-1]

		print(df.shape)
		print(sim_mean_z.shape)
		df['item_sim'] = sim_mean_z
		w = weight_models(len(user_idx))
		df['final_rec'] = w*df['item_sim'] + (1-w)*df['content_sim']
	else:
		df['final_rec'] = df['content_sim']

	prod = vals['product'].lower()
	# df = df[(df[prod])].reset_index(drop=True)
	df = df[(df[prod]) &  (df[vals['skin type']])].reset_index(drop=True)
	df_ranked = df.sort_values('final_rec',ascending=False).iloc[0:n]
	img_files = []
	m = [df_ranked[concern_names].min().min(),df_ranked[concern_names].max().max()]
	
	for i in range(n):
		img_files.append(radar_plot(Q['concerns'],df_ranked[concern_names].iloc[i].to_list(),m))

	df_ranked['img_file'] = img_files
	df_ranked['idx'] = list(range(1,n+1))
	products = df_ranked[['idx','url','product_image_url','name','brand','price','img_file']].T.to_dict()
	products = [j for i,j in products.items()]

	return products

