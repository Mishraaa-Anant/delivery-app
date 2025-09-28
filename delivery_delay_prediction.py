import argparse
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from xgboost import XGBClassifier


# ---------------------- Config ----------------------
DELIVERY_DELAY_THRESHOLD = 30  # minutes; change if you have a different definition of 'delayed'
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_DIR = "models"
DATA_PATH = "data/data.csv"  # Fixed dataset path
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------- Utility functions ----------------------

def parse_time_strings(date_series, time_series):
    """Parse date and time columns (strings) into pandas datetime and return hours, minutes and timestamps."""
    # combine date and time into a single datetime (if time has seconds or not it will handle common cases)
    combined = pd.to_datetime(date_series.astype(str) + " " + time_series.astype(str), errors='coerce')
    hour = combined.dt.hour.fillna(0).astype(int)
    minute = combined.dt.minute.fillna(0).astype(int)
    return combined, hour, minute


def safe_minutes_diff(later_dt, earlier_dt):
    """Return minutes difference (later - earlier), handle NaT safely."""
    diff = (later_dt - earlier_dt).dt.total_seconds() / 60.0
    return diff.fillna(0.0)


# ---------------------- Preprocessing ----------------------

def preprocess(df, fit=True, ct=None, scaler=None, label_encoders=None):
    """Preprocess DataFrame and return X, y, plus fitted transformers if fit=True.

    Steps:
    - parse date/time
    - create proxy distance feature using latitudes (because Store_Longitude not present)
    - create pickup_delay = minutes between Order and Pickup
    - categorical encodings for Weather, Traffic, Vehicle, Area, Category
    - numeric imputation
    - scaling
    """

    df = df.copy()

    # ----- basic cleaning -----
    # Check which columns are actually present
    print("Available columns:", df.columns.tolist())
    
    # Core required columns (adjust based on actual dataset)
    core_required = ['Order_ID', 'Delivery_Time']
    missing_core = set(core_required) - set(df.columns)
    if missing_core:
        raise ValueError(f"Missing critical columns: {missing_core}")

    # Optional columns with fallbacks
    optional_cols = {
        'Agent_Age': 30,  # default age
        'Agent_Rating': 4.0,  # default rating
        'Store_Latitude': 0.0,
        'Drop_Latitude': 0.0,
        'Drop_Longitude': 0.0,
        'Order_Date': '2023-01-01',
        'Order_Time': '12:00:00',
        'Pickup_Time': '12:30:00',
        'Weather': 'Clear',
        'Traffic': 'Medium',
        'Vehicle': 'Car',
        'Area': 'Urban',
        'Category': 'Food'
    }
    
    # Add missing columns with defaults
    for col, default_val in optional_cols.items():
        if col not in df.columns:
            print(f"Adding missing column '{col}' with default value: {default_val}")
            df[col] = default_val

    # parse datetimes
    try:
        order_dt, order_hour, order_min = parse_time_strings(df['Order_Date'], df['Order_Time'])
        pickup_dt = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Pickup_Time'].astype(str), errors='coerce')
        
        df['order_dt'] = order_dt
        df['order_hour'] = order_hour
        df['order_minute'] = order_min
        df['pickup_dt'] = pickup_dt
        df['pickup_delay_mins'] = safe_minutes_diff(df['pickup_dt'], df['order_dt'])
    except Exception as e:
        print(f"Error parsing dates/times: {e}")
        # Set defaults if parsing fails
        df['order_hour'] = 12
        df['order_minute'] = 0
        df['pickup_delay_mins'] = 30.0

    # target: delayed if Delivery_Time (assumed minutes) > threshold
    df['Delayed'] = (pd.to_numeric(df['Delivery_Time'], errors='coerce').fillna(30) > DELIVERY_DELAY_THRESHOLD).astype(int)
    
    print(f"Target distribution - Delayed: {df['Delayed'].sum()}, Not Delayed: {(df['Delayed'] == 0).sum()}")

    # proxy distance: absolute difference of latitudes (not perfect but usable if longitude missing)
    df['lat_diff'] = (pd.to_numeric(df['Store_Latitude'], errors='coerce').fillna(0) - 
                      pd.to_numeric(df['Drop_Latitude'], errors='coerce').fillna(0)).abs()

    # numeric features
    numeric_features = ['Agent_Age', 'Agent_Rating', 'pickup_delay_mins', 'lat_diff', 'order_hour']

    # categorical features
    categorical_features = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']

    if fit:
        # Column transformer for numeric and categorical
        numeric_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Fixed parameter name
            ]
        )

        ct = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features),
                ('cat', categorical_pipeline, categorical_features)
            ], remainder='drop'  # Changed from passthrough to drop
        )

        X = ct.fit_transform(df)
        # grab scaler from pipeline for potential use with LSTM
        scaler = ct.named_transformers_['num'].named_steps['scaler']

        # Also keep mapping of onehot categories for potential inverse transforms
        fitted_ohe = ct.named_transformers_['cat'].named_steps['onehot']
        metadata = {
            'ct': ct,
            'scaler': scaler,
            'ohe_categories': fitted_ohe.categories_,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }
        y = df['Delayed'].values

        print(f"Feature matrix shape: {X.shape}")
        return X, y, metadata

    else:
        if ct is None:
            raise ValueError("ct (ColumnTransformer) must be provided when fit=False")
        X = ct.transform(df)
        y = df['Delayed'].values
        return X, y


# ---------------------- Modeling & Evaluation ----------------------

def evaluate_model(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = None
    if y_proba is not None:
        try:
            roc = roc_auc_score(y_true, y_proba)
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
            roc = None

    print("Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    if roc is not None:
        print(f"ROC AUC: {roc:.4f}")
    print('\nClassification report:\n')
    print(classification_report(y_true, y_pred, zero_division=0))


# ---------------------- XGBoost training ----------------------

def train_xgb(X_train, y_train, X_val, y_val):
    print("Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    
    # Fixed indentation
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20
    )

    return model


# ---------------------- Simple LSTM (tabular -> single timestep) ----------------------

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ---------------------- Main flow ----------------------

def main(args):
    # Use default path if not provided
    data_path = args.data if args.data else DATA_PATH
    
    print("Loading data:", data_path)
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} does not exist!")
        print("Please make sure your CSV file is in the correct location.")
        return
    
    try:
        df = pd.read_csv(data_path)
        print("Rows:", len(df))
        print("Columns:", len(df.columns))
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Handle empty dataset
    if len(df) == 0:
        print("Error: Dataset is empty!")
        return

    # quick shuffle
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    try:
        # preprocess (fit)
        X_all, y_all, metadata = preprocess(df, fit=True)
        ct = metadata['ct']

        # Check if we have enough samples
        if len(X_all) < 10:
            print("Error: Not enough samples for training!")
            return

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
            stratify=y_all if len(np.unique(y_all)) > 1 else None
        )

        # further split train -> train/val for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, 
            stratify=y_train if len(np.unique(y_train)) > 1 else None
        )

        # ---------- XGBoost ----------
        print(f"\nTraining with {len(X_tr)} samples, validating with {len(X_val)} samples")
        xgb_model = train_xgb(X_tr, y_tr, X_val, y_val)

        # evaluate XGBoost
        preds = xgb_model.predict(X_test)
        if hasattr(xgb_model, 'predict_proba'):
            proba = xgb_model.predict_proba(X_test)[:, 1]
        else:
            proba = None
        print('\nXGBoost Results on Test Set:')
        evaluate_model(y_test, preds, proba)

        # save xgb and column transformer
        joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model.joblib'))
        joblib.dump(ct, os.path.join(MODEL_DIR, 'column_transformer.joblib'))
        print('Saved XGBoost model and preprocessing to', MODEL_DIR)

        # ---------- LSTM ----------
        print("\nTraining LSTM...")
        # LSTM expects 3D input: (samples, timesteps, features). We'll use timesteps=1 for tabular data.
        X_train_l = X_tr.reshape((X_tr.shape[0], 1, X_tr.shape[1]))
        X_val_l = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        X_test_l = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        lstm = build_lstm(input_shape=(1, X_tr.shape[1]))
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        chkpt_path = os.path.join(MODEL_DIR, 'lstm_best.h5')
        mc = ModelCheckpoint(chkpt_path, monitor='val_loss', save_best_only=True)

        history = lstm.fit(
            X_train_l, y_tr,
            validation_data=(X_val_l, y_val),
            epochs=100,
            batch_size=min(64, len(X_tr) // 4),  # Adjust batch size for small datasets
            callbacks=[es, mc],
            verbose=1
        )

        # evaluate LSTM
        lstm_preds_proba = lstm.predict(X_test_l).flatten()
        lstm_preds = (lstm_preds_proba >= 0.5).astype(int)
        print('\nLSTM Results on Test Set:')
        evaluate_model(y_test, lstm_preds, lstm_preds_proba)

        # save LSTM
        lstm.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
        print('Saved LSTM to', MODEL_DIR)

        # plot training history for LSTM
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('LSTM Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('LSTM Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'lstm_training.png'))
        print('Saved LSTM training plots')
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delivery delay prediction pipeline')
    parser.add_argument('--data', type=str, default=DATA_PATH, help='Path to CSV file (default: data/data.csv)')
    args = parser.parse_args()
    main(args)