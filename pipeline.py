from models.hybridize import hybridize
from bert import nlp_process
from sksutils import preprocess
from scraping import sephora_scrape
from bert import bert_setup
import argparse

def scrape(path):
	'''
	Wrapper function to scrape sephora API data
	input: path where you want scraping results to be output
	'''

	sephora_scrape.init_scrape(path)


def clean(path):
	'''
	Wrapper function to pre-process raw summary and review data
	input: path where you want cleaned data to be output
	'''

	preprocess.preprocess(path)
	nlp_process.init_process(path)
	nlp_process.bert_setup(path)
	bert_setup.save_test(path)

def model(path_in, path_out, path_bert):
	'''
	Wrapper function to scrape sephora API data
	input: 
		path_in: path where cleaned data are stored
		path_out: path where your final model data are stored
		path_bert: where bert output is stored
	'''

	hybridize(path_in,path_out,path_bert)

def main():

	root = '~/Documents/insight/skinsight/app/'
	lg_path = '{}/data_large'.format(root)
	sm_path = '{}/data'.format(root)
	bert_path = '{}/data_large/bert/bert_final'.format(root)

	parser = argparse.ArgumentParser()
	parser.add_argument('--scrape', default=False)
	parser.add_argument('--clean', default=False)
	parser.add_argument('--model',  default=False)
	args = parser.parse_args()

	if args.scrape == 'True':
		scrape(lg_path)

	if args.clean == 'True':
		clean(lg_path)

	if args.model == 'True':
		model(lg_path, sm_path, bert_path)


	return

if __name__ == '__main__':
	main()







