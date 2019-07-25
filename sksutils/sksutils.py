import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from math import pi

def thresh_rm(df,cols,thresh):

	'''
    Remove data below a certain percentile of a given frequency distribution
    input:
    	df: review dataframe
    	cols: column on which to generate distribution
    	thresh: threshold for removal
    '''

	rm_thresh = [(i,np.percentile(df[i].value_counts().values,thresh)) for i in cols]
	for i in rm_thresh:
	    tmp = df[i[0]].value_counts().reset_index()
	    multi_list = tmp.loc[tmp[i[0]]>i[1],'index'].tolist()
	    df = df[df[i[0]].isin(multi_list)].reset_index(drop=True)
	return df


def init_cats(cat_type):

	'''
    Returns hard-coded information for website
    input: category of information desired
    '''

	if cat_type == 'broad':
		return ['skin type', 'price sensitivity', 'concerns', 'product type']

	elif cat_type == 'concerns':
		return ['Texture/Pores','Redness','Dark Spots','Sensitivity','Wrinkles','Acne']

	elif cat_type == 'product type':
		return ['Cleanser','Moisturizer','Treatment','Mask','Eye','Sunscreen']

	elif cat_type == 'skin type':
		return ['Normal','Oily','Dry','Combination']

	elif cat_type == 'price sensitivity':
		return ['Low','Medium','High']

	else:
		return -1


def gen_Q():

	'''
    Generate dictionary of all categories for KBM
    '''

	return {c: init_cats(c) for c in init_cats('broad')}


def concern_strmatch():

	'''
    Returns dictionary of concerns and string lookup list for each concern
    '''

	cats = init_cats('concerns')
	strmatches = ['pore|texture|smooth|soft|dull',
	' red |redness',
	'dark|pigment|bright|glow',
	'sensitiv|burn|irritat|gentle|inflame',
	'wrinkle| lines| age | aging',
	'acne|break|broke|pimple|bump|blackhead|whitehead|comedone']

	return {cats[i]:strmatches[i] for i in range(len(cats))}


def stack_lists(df,new_col_name):

	'''
    expand list within dataframe cell into a column
    input:
    	df: review dataframe
    	new_col_name: renamed col
    '''

	df_new = (df.apply(pd.Series)
          .stack()
          .reset_index(level=1, drop=True)
          .to_frame(new_col_name)).reset_index()

	return df_new

def radar_plot(cat,values,concerns):

	'''
    Build radar plot and return byte-encoded tmp image file path
    input:
    	cat: aspect names
    	values: model-generated aspect aggregation
    	concerns: user generated concerns
    '''

	N = len(cat)
	m = [0,1]
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

	'''
    specific minmax scaler that works on single series too
    '''

	n_min = np.nanmin(df,axis=0)
	n_max = np.nanmax(df,axis=0)

	df = (df-n_min)/(n_max-n_min)
	return df

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def weight_models(l,w_init):

	'''
    Determine hyperparameter for weighing KBM vs CF
    input:
    	l: # of items chosen by user
    	w_init: inital weight
    '''

	w = w_init * sigmoid(l)
	return w

def get_price(x):

	'''
    Helper fct for setting price weight
    '''

	cats = init_cats('price sensitivity')
	if x == cats[0]:
		y = 0
	elif x == cats[1]:
		y = .5
	else:
		y = 1
	return y

def bashdir2concerns():

	'''
    Helper fct for converting directories to website names
    '''

	D = dict()
	D['acne'] = 'Acne'
	D['texture'] = 'Texture/Pores'
	D['redness'] = 'Redness'
	D['sensitive'] = 'Sensitivity'
	D['dark spot'] = 'Dark Spots'
	D['wrinkle'] = 'Wrinkles'
	return D

def bashdir():
	return ['acne','texture','redness', 'wrinkle','sensitive','dark spot']