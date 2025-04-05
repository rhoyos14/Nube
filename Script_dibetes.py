import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import OrdinalEncoder

def _parse_args():
    parser = argparse.ArgumentParser()

    # Argumentos para las rutas
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='diabetic_data.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    parser.add_argument('--categorical_features', type=str, default='insulin, race, A1Cresult, gender, weight, max_glu_serum, metformin, glyburide, glipizide, glimepiride, rosiglitazone, pioglitazone, repaglinide, acarbose, glyburide-metformin, nateglinide, chlorpropamide, miglitol, examide, citoglipton, glimepiride-pioglitazone')

    return parser.parse_known_args()

if __name__ == "__main__":
    args, _ = _parse_args()

    # Leer el archivo CSV de entrada
    df = pd.read_csv(os.path.join(args.filepath, args.filename))

    # Limpieza básica
    df = df.replace(regex=r'\.', value='_')
    df = df.replace(regex=r'\_$', value='')
    df.replace('?', pd.NA, inplace=True)

    # Eliminar columnas irrelevantes
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Mapear variable objetivo
    readmitted_map = {'NO': 0, '>30': 1, '<30': 2}
    df['readmitted'] = df['readmitted'].map(readmitted_map)

    # Separar los datos en train (70%), validation (20%), test (10%)
    train_df, validation_df, test_df = np.split(df.sample(frac=1, random_state=1729), 
                                                 [int(0.7 * len(df)), int(0.9 * len(df))])

    # Asegurar que las carpetas de salida existen
    train_output_path = os.path.join(args.outputpath, 'train')
    validation_output_path = os.path.join(args.outputpath, 'validation')
    test_output_path = os.path.join(args.outputpath, 'test')

    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(validation_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    # Guardar los archivos de salida
    pd.concat([train_df['readmitted'], train_df.drop(['readmitted'], axis=1)], axis=1).to_csv(
        os.path.join(train_output_path, 'train.csv'), index=False, header=False)

    pd.concat([validation_df['readmitted'], validation_df.drop(['readmitted'], axis=1)], axis=1).to_csv(
        os.path.join(validation_output_path, 'validation.csv'), index=False, header=False)

    test_df['readmitted'].to_csv(
        os.path.join(test_output_path, 'test_script_y.csv'), index=False, header=False)

    test_df.drop(['readmitted'], axis=1).to_csv(
        os.path.join(test_output_path, 'test_script_x.csv'), index=False, header=False)

    print("## Processing completed. Exiting.")





