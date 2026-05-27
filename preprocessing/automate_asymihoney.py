import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ── Setup Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Fungsi-fungsi Preprocessing ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset dari filepath."""
    logger.info(f"Loading data dari: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data berhasil dimuat. Shape: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris duplikat."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    logger.info(f"Duplikat dihapus: {removed} baris. Shape sekarang: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputasi missing values — median untuk numerik, modus untuk kategorikal."""
    numerical_cols  = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Hanya proses kolom yang benar-benar ada di dataframe
    num_cols_exist = [c for c in numerical_cols  if c in df.columns]
    cat_cols_exist = [c for c in categorical_cols if c in df.columns]

    if num_cols_exist:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols_exist] = num_imputer.fit_transform(df[num_cols_exist])

    if cat_cols_exist:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols_exist] = cat_imputer.fit_transform(df[cat_cols_exist])

    logger.info(f"Missing values setelah imputasi: {df.isnull().sum().sum()}")
    return df


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus outlier menggunakan metode IQR pada fitur numerik."""
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cols_exist     = [c for c in numerical_cols if c in df.columns]

    before = len(df)
    for col in cols_exist:
        Q1    = df[col].quantile(0.25)
        Q3    = df[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df    = df[(df[col] >= lower) & (df[col] <= upper)]

    logger.info(f"Outlier dihapus: {before - len(df)} baris. Shape sekarang: {df.shape}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan kolom kategorikal bertipe integer."""
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    cols_exist       = [c for c in categorical_cols if c in df.columns]

    for col in cols_exist:
        df[col] = df[col].astype(int)

    logger.info("Encoding kategorikal selesai.")
    return df


def split_and_scale(df: pd.DataFrame):
    """Split train/test lalu standarisasi fitur numerik."""
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Split selesai — Train: {X_train.shape}, Test: {X_test.shape}")

    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cols_exist     = [c for c in numerical_cols if c in X_train.columns]

    scaler          = StandardScaler()
    X_train[cols_exist] = scaler.fit_transform(X_train[cols_exist])
    X_test[cols_exist]  = scaler.transform(X_test[cols_exist])
    logger.info("Standarisasi selesai.")

    train_df = X_train.copy(); train_df['target'] = y_train.values
    test_df  = X_test.copy();  test_df['target']  = y_test.values
    return train_df, test_df


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame,
              output_dir: str = 'preprocessing/heart_preprocessing') -> None:
    """Simpan hasil preprocessing ke CSV."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'heart_train.csv')
    test_path  = os.path.join(output_dir, 'heart_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    logger.info(f"Data tersimpan → {train_path} & {test_path}")


# ── Pipeline Utama ────────────────────────────────────────────────────────────

def preprocess_pipeline(input_path:  str = 'heart_raw/heart.csv',
                        output_dir:  str = 'preprocessing/heart_preprocessing') -> tuple:
    """
    Jalankan seluruh pipeline preprocessing dari raw CSV
    hingga train/test yang siap dilatih.

    Returns:
        (train_df, test_df) — DataFrame siap latih
    """
    logger.info("=" * 55)
    logger.info("MEMULAI PIPELINE PREPROCESSING")
    logger.info("=" * 55)

    df = load_data(input_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = remove_outliers_iqr(df)
    df = encode_categorical(df)

    train_df, test_df = split_and_scale(df)
    save_data(train_df, test_df, output_dir)

    logger.info("=" * 55)
    logger.info("PIPELINE SELESAI ✅")
    logger.info(f"  Train shape : {train_df.shape}")
    logger.info(f"  Test shape  : {test_df.shape}")
    logger.info("=" * 55)

    return train_df, test_df


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_df, test_df = preprocess_pipeline(
        input_path='heart_raw/heart.csv',
        output_dir='preprocessing/heart_preprocessing'
    )