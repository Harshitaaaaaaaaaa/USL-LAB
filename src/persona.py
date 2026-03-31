def create_persona(df, features):

    # ==============================
    # 1. CLUSTER SUMMARY
    # ==============================
    cluster_summary = df.groupby('cluster')[features].mean()

    # ==============================
    # 2. DYNAMIC THRESHOLDS 🔥
    # ==============================
    balance_high = df['BALANCE'].quantile(0.7)
    balance_low = df['BALANCE'].quantile(0.3)

    payment_high = df['PAYMENTS'].quantile(0.7)
    purchase_high = df['PURCHASES'].quantile(0.7)
    purchase_low = df['PURCHASES'].quantile(0.3)

    # ==============================
    # 3. PERSONA LOGIC (IMPROVED)
    # ==============================
    def assign(cluster):
        row = cluster_summary.loc[cluster]

        # 🟢 Premium users
        if row['BALANCE'] >= balance_high and row['PAYMENTS'] >= payment_high:
            return "Premium Customers"

        # 🔴 Risky users (high balance, low payment)
        elif row['BALANCE'] >= balance_high and row['PAYMENTS'] < payment_high:
            return "Risky Customers"

        # 🟡 High spenders (frequent purchases)
        elif row['PURCHASES'] >= purchase_high:
            return "Frequent Spenders"

        # 🔵 Low usage
        elif row['PURCHASES'] <= purchase_low and row['BALANCE'] <= balance_low:
            return "Low Activity Users"

        # ⚪ Default
        else:
            return "Regular Users"

    # ==============================
    # 4. ASSIGN PERSONA
    # ==============================
    df['persona'] = df['cluster'].apply(assign)

    # ==============================
    # 5. PERSONA SUMMARY
    # ==============================
    persona_summary = df.groupby('persona')[features].mean()

    print("\n📊 Persona Summary:\n", persona_summary)

    return df, cluster_summary