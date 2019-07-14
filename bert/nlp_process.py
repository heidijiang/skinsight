import pandas as pd
import spacy
import re
from sksutils.sksutils import concern_strmatch, stack_lists
import os

def clean_words(text):
    text = re.sub('[^a-zA-Z0-9\'\!\.\?\:\;\,]', ' ',text)
    text = re.sub(r'([^a-zA-Z0-9])\1{1,}', r'\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'(?<=[.,:;?!])(?=[^\s])', r'', text)
    text = re.sub('\s+',' ', text)
    text = re.sub('^\s','',text)
    
    return text.lower()


def init_process(path):
	df = pd.read_csv('{}/db_reviews.csv'.format(path))

	df['review_clean'] = df['review_text'].apply(clean_words)
	df['review_sents'] = df['review_clean'].str.replace('!','.').str.replace('?','.').str.split('.')

	df_all = stack_lists(df['review_sents'],'sent_process')

	df_all = df_all[df_all['sent_process'].str.len()>15].reset_index(drop=True)
	df_all = df_all[df_all['sent_process'].notna()].reset_index(drop=True)
	df_all['sent_process'] = df_all['sent_process'].str.replace('^\s','', regex=True)

	df_all = pd.merge(df.reset_index(),df_all,on='index',how='left').drop(['review_text','review_clean','review_sents'],axis=1)

	D = concern_strmatch()

	for key,vals in D.items():
	    name = '{}_match'.format(key)
	    df_all[name] = df_all['sent_process'].str.contains(vals)

	df_rel = df_all[df_all[['{}_match'.format(i) for i in D.keys()]].any(axis=1)].reset_index().rename(columns={'level_0':'sample_index'})

	rel_file = '{}/db_aspect_sentences_bert.csv'.format(path)

	if os.path.isfile(os.path.expanduser(rel_file)):
	    df_bert = pd.read_csv(rel_file)
	    df_bert['sentiment'] = True
	    df_rel = pd.merge(df_rel,df_bert[['user_name','product_id','sent_process','sentiment']],on=['user_name','product_id','sent_process'],how='left')    
	    df_rel['sentiment'] = df_rel['sentiment'].fillna(False)
	else:
	    df_rel['sentiment'] = False

	# df_all.to_csv('{}/db_all_sentences.csv'.format(path), index=False)
	df_rel.to_csv('{}/db_aspect_sentences.csv'.format(path), index=False)

	df_rel_new = df_rel[~(df_re['sentiment'])].reset_index(drop=True)
	df_rel_new.to_csv('{}/db_aspect_sentences_new.csv'.format(path), index=False)

	print('Sentence processing done')

	return df_all



def relevant_spacy(file):

	cols = ['{}_match'.format(i) for i in init_cats('concerns')]
	df = pd.read_csv('{}/db_aspect_sentences.csv'.format(file))

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


	df_spacy.to_pickle('db_aspect_sentences_train.pkl')

	return df_spacy
	

if __name__ == '__main__':
	file = '~/Documents/insight/skinsight'
	init_process(file)
	relevant_spacy(file)
