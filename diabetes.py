import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import OrdinalEncoder

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='diabetic_data.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    return parser.parse_known_args()

if __name__ == "__main__":
    args, _ = _parse_args()

    # Rutas completas
    input_file = os.path.join(args.filepath, args.filename)
    output_train_path = os.path.join(args.outputpath, 'train')
    output_val_path = os.path.join(args.outputpath, 'validation')
    output_test_path = os.path.join(args.outputpath, 'test')

    # Asegurar carpetas de salida
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_val_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    # Leer datos
    df = pd.read_csv(input_file)
    print(f"✅ Datos cargados desde: {input_file}, shape: {df.shape}")

    # Limpieza básica
    df = df.replace(regex=r'\.', value='_').replace(regex=r'\_$', value='')
    df.replace('?', pd.NA, inplace=True)
    df.drop(columns=['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr'], errors='ignore', inplace=True)

    # Mapear target
    df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

    # Dividir los datos
    train_df, val_df, test_df = np.split(df.sample(frac=1, random_state=1729), [int(0.7 * len(df)), int(0.9 * len(df))])
    print(f"📊 Split: Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}")

    # Guardar outputs
    train_df.to_csv(os.path.join(output_train_path, 'train.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(output_val_path, 'validation.csv'), index=False, header=False)
    test_df['readmitted'].to_csv(os.path.join(output_test_path, 'test_y.csv'), index=False, header=False)
    test_df.drop(['readmitted'], axis=1).to_csv(os.path.join(output_test_path, 'test_x.csv'), index=False, header=False)

    # Confirmaciones
    print("✅ Archivos guardados:")
    print(" -", os.path.join(output_train_path, 'train.csv'))
    print(" -", os.path.join(output_val_path, 'validation.csv'))
    print(" -", os.path.join(output_test_path, 'test_x.csv'))
    print(" -", os.path.join(output_test_path, 'test_y.csv'))

    print("🚀 Script finalizado con éxito.")