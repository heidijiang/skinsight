import pandas as pd
import spacy
import re
from sksutils.sksutils import *


def clean_words(text):
    text = re.sub('[^a-zA-Z0-9\'\!\.\?\:\;\,]', ' ',text)
    text = re.sub(r'([^a-zA-Z0-9])\1{1,}', r'\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'(?<=[.,:;?!])(?=[^\s])', r'', text)
    text = re.sub('\s+',' ', text)
    text = re.sub('^\s','',text)
    
    return text.lower()


def init_process(file):

	df = pd.read_csv('{}/skin_reviews.csv'.format(file))

	df['review_clean'] = df['review_text'].apply(clean_words)
	df['review_sents'] = df['review_clean'].str.replace('!','.').str.replace('?','.').str.split('.')

	df_all = stack_lists(df['review_sents'])

	df_all = df_all[df_all['sents'].str.len()>15].reset_index(drop=True)
	df_all = df_all[df_all['sents'].notna()].reset_index(drop=True)
	df_all['sents'] = df_all['sents'].str.replace('^\s','', regex=True)

	df_all = pd.merge(df.reset_index(),df_all,on='index',how='left').drop(['review_text','review_clean','review_sents'],axis=1)

	D = concern_strmatch()

	for key,vals in D.items():
	    name = '{}_match'.format(key)
	    df_all[name] = df_all['sents'].str.contains(vals)

	df_all.to_csv('{}/skin_all_sentences.csv'.format(file), index=False)

	return df_all


def relevant_spacy(file):

	cols = ['{}_match'.format(i) for i in init_cats('concerns')]
	df = pd.read_csv('{}/skin_all_sentences.csv'.format(file))

	df_spacy = df[df[cols].any(axis=1)]

	nlp = spacy.load("en_core_web_sm")
	nlp.Defaults.stop_words -= {"no","n't", "nothing","not",'never','least','less','more',
								'however','few','even','but','have'}

	df_spacy['nlp'] = df_spacy['sents'].apply(nlp)

	df_spacy['all_terms'] = df_spacy['nlp'].apply(lambda x: 
		[token.lemma_ for token in x if not len(token.text)==0 and token.text not in nlp.Defaults.stop_words and not token.is_punct])
	df_spacy['sentiment_terms'] = df_spacy['nlp'].apply(lambda x: 
		[token.lemma_ for token in x if token.text not in nlp.Defaults.stop_words and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB" or token.pos_ == "ADV")])
	df_spacy['aspect_terms'] = df_spacy['nlp'].apply(lambda x: 
		[token.lemma_ for token in x if token.text not in nlp.Defaults.stop_words and not token.is_punct and token.pos_ == 'NOUN'])

	df_spacy['all_terms_nolemma'] = df_spacy['nlp'].apply(lambda x:
		[token for token in x if not len(token)==0 and token not in nlp.Defaults.stop_words])
	df_spacy['sentiment_terms_nolemma'] = df_spacy['nlp'].apply(lambda x: 
		[token.text for token in x if token.text not in nlp.Defaults.stop_words and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB" or token.pos_ == "ADV")])
	df_spacy['aspect_terms_nolemma'] = df_spacy['nlp'].apply(lambda x: 
		[token.text for token in x if token.text not in nlp.Defaults.stop_words and not token.is_punct and token.pos_ == 'NOUN'])


	df_spacy.to_pickle('skin_relevant_sentences_train.pkl')

	return df_spacy
	

if __name__ == '__main__':
	file = '~/Documents/insight/skinsight'
	init_process(file)
	relevant_spacy(file)
