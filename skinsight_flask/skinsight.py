from sksutils.sksutils import *
from models.hybrid import Hybrid


def get_collab_imgs(Model,product_type):

	df = Model.gen_cf_sample(product_type)
	
	return df.T.to_dict()


def get_recs(Model):
	
	Q = gen_Q()

	Model.KBM('minmax',use_skintype=True)
	Model.CF('minmax')
	Model.gen_ranks()
	Model.add_radar()

	products = Model.output.T.to_dict()
	products = [j for i,j in products.items()]

	return products

