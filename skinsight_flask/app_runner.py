from skinsight_flask import app
from flask import render_template, request, session
from skinsight_flask.skinsight import *
from sksutils.sksutils import init_cats
from models.hybrid import Hybrid

Model = Hybrid()

@app.route('/')
@app.route('/index.html')
def index():
	Q = gen_Q()
	return render_template('index.html', Q=Q)


@app.route('/quiz', methods=['POST'])
def quiz():

	session['user_input'] = request.values
	products = get_collab_imgs(Model,request.values['product type'])
	return render_template('quiz.html', products=products)


@app.route('/results',methods=['POST'])
def results():
	Model.user_input(session['user_input'])
	Model.set_item_history(list(request.values.to_dict().keys()))

	products = get_recs(Model)

	return render_template('results.html', products=products)


if __name__=='__main__':
	app.run(debug=True)