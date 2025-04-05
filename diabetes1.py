import pandas as pd
import numpy as np
import argparse
import os

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='diabetic_data.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    return parser.parse_known_args()

if __name__ == "__main__":
    args, _ = _parse_args()

    input_file = os.path.join(args.filepath, args.filename)
    output_train_path = os.path.join(args.outputpath, 'train')
    output_val_path = os.path.join(args.outputpath, 'validation')
    output_test_path = os.path.join(args.outputpath, 'test')

    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_val_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    # Leer datos
    df = pd.read_csv(input_file)
    print(f"✅ Datos cargados desde: {input_file}, shape: {df.shape}")

    # Limpieza básica
    df.replace(regex=r'\.', value='_', inplace=True)
    df.replace(regex=r'\_$', value='', inplace=True)
    df.replace('?', pd.NA, inplace=True)

    # Eliminar columnas irrelevantes o con muchos nulos
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
    df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # Mapear target
    df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

    # Eliminar filas con valores nulos (opcional pero recomendable)
    df.dropna(inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df)

    # Separar target
    y = df['readmitted']
    X = df.drop('readmitted', axis=1)

    # Unir target + features (target al inicio)
    df_final = pd.concat([y, X], axis=1)

    # Shuffle y split
    train_df, val_df, test_df = np.split(df_final.sample(frac=1, random_state=1729), 
                                         [int(0.7 * len(df_final)), int(0.9 * len(df_final))])

    # Guardar CSVs sin encabezado
    train_df.to_csv(os.path.join(output_train_path, 'train.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(output_val_path, 'validation.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(output_test_path, 'test.csv'), index=False, header=False)

    print("🚀 Archivos generados:")
    print(" - Train:", os.path.join(output_train_path, 'train.csv'))
    print(" - Validation:", os.path.join(output_val_path, 'validation.csv'))
    print(" - Test:", os.path.join(output_test_path, 'test.csv'))
    print("✅ Procesamiento finalizado.")
