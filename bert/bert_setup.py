import os
import pandas as pd
from sksutils.sksutils import bashdir

def save_test(path_in,path_out):
    df = pd.read_csv('{}/db_aspect_sentences_new.csv'.format(path_in))
    aspects = bashdir()

    for aspect in aspects:
        
        basedir = '{}/{}/data'.format(path_out, aspect.replace(' ','_'))
        for d in ['train','dev','test']:
            try:
                os.makedirs('{}_{}'.format(basedir,d))
            except:
                continue
        
        test = df[df[aspect]].reset_index(drop=True)
        df_bert_test = pd.DataFrame({'index':list(range(test.shape[0]))})
        df_bert_test['sentences'] = test['sent_process']

        df_bert_test.to_csv('{}_test/test.tsv'.format(basedir), sep='\t', index=False, header=True)
        test[['user_name','product_id','sample_index']].to_csv('{}_test/test_idx.csv'.format(basedir))

    print('Sentences saved to bert path')
    return
