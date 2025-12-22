import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib

cols_to_drop = [
    'Культура', 
    'Гаражний номер',
    'Сума_опадів_за_попередні_48_год_мм',
    'Сума_опадів_за_попередні_24_год_мм',
    'Сума_опадів_за_попередні_12_год_мм',
    'Сума_опадів_за_попередні_9_год_мм',
    'Сума_опадів_за_попередні_6_год_мм'
]

rain_cols_to_fill = [
    'Опади_мм',
    'Сума_опадів_за_позавчора_мм',
    'Сума_опадів_за_вчора_мм',
    'Сума_опадів_за_ніч_мм',
    'Сума_опадів_за_попередні_3_год_мм'
]

def clean_model_name(name):
    name = str(name).upper()
    if name.startswith('CR'):
        return 'NEW HOLLAND ' + name
    return name

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    cols_to_drop_actual = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop_actual)
    df[rain_cols_to_fill] = df[rain_cols_to_fill].fillna(0)
    df['Модель'] = df['Модель'].apply(clean_model_name)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['date_time'] = pd.to_datetime(df['date_time'])
    
    df['Година'] = df['date_time'].dt.hour.astype('float')
    return df

def time_series_split(df: pd.DataFrame, 
                      val_start_date='2025-08-01 00:00:00', 
                      test_start_date='2025-08-15 00:00:00'):
    train_df = df[df['date_time'] < val_start_date].copy()
    val_df = df[(df['date_time'] >= val_start_date) & (df['date_time'] < test_start_date)].copy()
    test_df = df[df['date_time'] >= test_start_date].copy()
    
    print(f"--- Розподіл даних ---")
    print(f"Train: {len(train_df)} рядків. Період: {train_df['date_time'].min()} — {train_df['date_time'].max()}")
    print(f"Val:   {len(val_df)} рядків.   Період: {val_df['date_time'].min()} — {val_df['date_time'].max()}")
    print(f"Test:  {len(test_df)} рядків.  Період: {test_df['date_time'].min()} — {test_df['date_time'].max()}")
    
    return train_df, val_df, test_df

def process_features(train_df, val_df, test_df, target_col='Зібрано_га', scale_numeric=False):
    num_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in num_features:
        num_features.remove(target_col)
    
    cat_features = train_df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Числові ознаки: {num_features}")
    print(f"Категорійні: {cat_features}")
    print(f"Цільова змінна: {target_col}")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[cat_features])
    encoded_cols = list(encoder.get_feature_names_out(cat_features))
    
    def transform_subset(subset_df):
        
        encoded_data = encoder.transform(subset_df[cat_features])
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=subset_df.index)
        
        num_df = subset_df[num_features]
        
        X = pd.concat([num_df, encoded_df], axis=1)
        y = subset_df[target_col]
        return X, y

    X_train, y_train = transform_subset(train_df)
    X_val, y_val = transform_subset(val_df)
    X_test, y_test = transform_subset(test_df)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, encoder

def get_processed_data(filepath):
    """Головна функція-обгортка."""
    df = load_and_clean_data(filepath)
    df = feature_engineering(df)
    train_df, val_df, test_df = time_series_split(df)
    
    return process_features(train_df, val_df, test_df)

def preprocess_for_inference(df: pd.DataFrame, encoder_path: str = 'encoder.joblib'):
    """
    Підготовка нових даних для прогнозування.
    1. Очищення та feature engineering.
    2. Кодування категорій (використовуючи збережений encoder).
    """
    cols_to_drop_actual = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop_actual)
    
    existing_rain_cols = [c for c in rain_cols_to_fill if c in df.columns]
    df[existing_rain_cols] = df[existing_rain_cols].fillna(0)
    
    if 'Модель' in df.columns:
        df['Модель'] = df['Модель'].apply(clean_model_name)
    
    df = feature_engineering(df)
    
    try:
        encoder = joblib.load(encoder_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Не знайдено файл {encoder_path}. Спочатку збережіть енкодер з ноутбука.")

    cat_features = encoder.feature_names_in_.tolist()
    
    exclude_cols = cat_features + ['date_time', 'Зібрано_га', 'Прогноз_га']
    num_features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    
    encoded_data = encoder.transform(df[cat_features])
    encoded_cols = list(encoder.get_feature_names_out(cat_features))
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
    
    X = pd.concat([df[num_features], encoded_df], axis=1)
    
    return X, df