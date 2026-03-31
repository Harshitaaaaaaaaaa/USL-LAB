import pandas as pd
from sklearn.preprocessing import StandardScaler


# ==============================
# 1. LOAD DATA
# ==============================
def load_data(path):
    df = pd.read_csv(path)

    # Drop ID column safely
    if "CUST_ID" in df.columns:
        df = df.drop("CUST_ID", axis=1)

    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    print("\n✅ Data Loaded Successfully")
    print("Shape:", df.shape)

    return df


# ==============================
# 2. SELECT FEATURES 🔥
# ==============================
def get_features(df):

    # 👉 Improved feature set (VERY IMPORTANT)
    features = [
        'BALANCE',
        'PURCHASES',
        'CREDIT_LIMIT',
        'PAYMENTS',
        'MINIMUM_PAYMENTS',
        'INSTALLMENTS_PURCHASES'
    ]

    # Keep only existing columns (safe)
    features = [col for col in features if col in df.columns]

    print("\n📊 Features Used:", features)

    return df[features]


# ==============================
# 3. SCALING
# ==============================
def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n📏 Data Scaled (StandardScaler Applied)")

    return X_scaled