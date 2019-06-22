from skinsight_flask import app
from flask import render_template, request, session
from skinsight_flask.skinsight import *


Q = {'age':['15-21','22-29','30-39','40-49','50+'],
	'skin type':['Normal','Oily','Dry','Combination'],
	'concerns':['Acne','Dark Spots','Redness','Texture/Pores','Sensitivity','Wrinkles'],
	'product type':['Cleanser','Moisturizer','Treatment','Mask','Eye','Sunscreen']
}

@app.route('/')
@app.route('/index.html')
def index():
	return render_template('index.html', Q=Q)


@app.route('/quiz', methods=['POST'])
def quiz():
	n_disp = 40
	vals = dict()
	vals['concerns'] = dict()
	concerns = []
	for i in Q['concerns']:
		concerns.append(request.values[i])

	vals['concerns'] = concerns
	vals['age'] = request.values['age']
	vals['skin type'] = request.values['skin type']
	vals['product'] = request.values['product type']

	products = get_collab_imgs('skinsight_flask/static/data/skin_concerns_summary.csv',vals,n_disp)
	session['vals'] = vals
	return render_template('quiz.html', products=products)


@app.route('/results',methods=['POST'])
def results():
	iids_user = request.values
	n = 10
	vals = session['vals']
	products = get_recs('skinsight_flask/static/data/skin_concerns_summary.csv', vals, Q, n, iids_user)

	return render_template('results.html', products=products)


if __name__=='__main__':
	app.run(debug=True)