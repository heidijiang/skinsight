import pandas as pd
from models.kbm import gen_knowledge_model
from models.cf import gen_cf


def hybridize(file_in,file_out,bert_path):

    df_kbm = gen_knowledge_model(file_in,bert_path)
    df_cf = gen_cf(file_in)
    df_cf['product_id'] = df_cf.columns
    df_final = pd.merge(df_kbm,df_cf,on='product_id')

    df_final.to_csv('{}/kbm_cf.csv'.format(file_out), index=False)

    return df_final
