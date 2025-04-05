import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import OrdinalEncoder

def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='diabetic_data.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    parser.add_argument('--categorical_features', type=str, default='insulin, race, A1Cresult, gender, weight, max_glu_serum, metformin, glyburide, glipizide, glimepiride, rosiglitazone, pioglitazone, repaglinide, acarbose, glyburide-metformin, nateglinide, chlorpropamide, miglitol, examide, citoglipton, glimepiride-pioglitazone')

    return parser.parse_known_args()
  


if __name__=="__main__":
    args, _ = _parse_args()
    df = pd.read_csv(os.path.join(args.filepath, args.filename))
    df = df.replace(regex=r'\.', value='_')
    df = df.replace(regex=r'\_$', value='')

    #leer los datos 
    pd.set_option('display.max_columns', 500)     # ver todas las columnas
    pd.set_option('display.max_rows', 20)         # salida 
    df

    #remplazar valores "?" con Nan y eliminarlos si es necesario 
    df.replace('?', pd.NA, inplace=True)

     # Eliminar columnas con muchos valores faltantes o irrelevantes (ejemplo)
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    #mapear la varaible objetivo 
    readmitted_map = {'NO': 0, '>30': 1, '<30': 2}
    df['readmitted'] = df['readmitted'].map(readmitted_map)

    #splitear los datos en dos reparticiones del 70 % y del 70% al 90% y el 10% restante para el test
    int(0.7 * len(df)), int(0.9 * len(df))
    train_df, validation_df, test_df= np.split(df.sample(frac=1, random_state=1729), [int(0.7 * len(df)), int(0.9 * len(df))]) 

    #concatenar
    pd.concat([train_df['readmitted'], train_df.drop(['readmitted', 'readmitted'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
    pd.concat([validation_df['readmitted'], validation_df.drop(['readmitted', 'readmitted'], axis=1)], axis=1).to_csv('validation.csv', index=False, header=False)
    test_df['readmitted'].to_csv(os.path.join(args.outputpath, 'test/test_script_y.csv'), index=False, header=False)
    test_df.drop(['readmitted'], axis=1).to_csv(os.path.join(args.outputpath, 'test/test_script_x.csv'), index=False, header=False)
    print("## Processing completed. Exiting.")




