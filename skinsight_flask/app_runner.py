from skinsight_flask import app
from flask import render_template, request, session
from skinsight_flask.skinsight import *
from sksutils.sksutils import init_cats


Q = gen_Q()

maindir = './skinsight_flask/static/data'

@app.route('/')
@app.route('/index.html')
def index():
	return render_template('index.html', Q=Q)


@app.route('/quiz', methods=['POST'])
def quiz():
	n_disp = 40
	vals = dict()

	for c in Q.keys():
		cnew = c.split(' ')[0]
		if c == 'concerns':
			vals[cnew] = [request.values[i] for i in Q[c]]
		else:
			vals[cnew] = request.values[c]
	

	products = get_collab_imgs('{}/df_skin_concerns_sentiment.csv'.format(maindir),vals,n_disp)
	session['vals'] = vals
	return render_template('quiz.html', products=products)


@app.route('/results',methods=['POST'])
def results():
	iids_user = request.values
	n = 10
	vals = session['vals']

	file_content = '{}/df_skin_concerns_sentiment.csv'.format(maindir)
	file_item = '{}/item_collab_sim.csv'.format(maindir)

	products = get_recs(file_content, file_item,vals, Q, n, iids_user)

	return render_template('results.html', products=products)


if __name__=='__main__':
	app.run(debug=True)