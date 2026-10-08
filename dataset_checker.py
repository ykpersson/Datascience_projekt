import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Läs in träningsfilen
df = pd.read_csv(r"C:\Users\Eier\OneDrive\Dokument\DataScience\Data_science_projektkurs\archive (1)\diabetes_prediction_dataset.csv")
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['gender'].unique())
print(df['smoking_history'].unique())
df_encoded = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=False)
print(df_encoded.head())
print(df_encoded.columns.tolist())
numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 
                'HbA1c_level', 'blood_glucose_level', 'diabetes']

df_numeric = df_encoded[numeric_cols]
print(df_numeric.describe().T)
print(df['diabetes'].value_counts())

Y = df_encoded['diabetes']
X = df_encoded.drop(columns=['diabetes'])
corr_list = []

for col in X.columns:
    corr_list.append([col, X[col].corr(Y)])

corr_df = pd.DataFrame(corr_list, columns=['feature', 'corr_with_Y'])
print(corr_df.sort_values(by='corr_with_Y', ascending=False))

corr_matrix = X.corr()
print(corr_matrix)

stronger_corr_df = corr_df[corr_df['corr_with_Y'].abs() > 0.1]
print(stronger_corr_df)

weaker_corr_df = corr_df[corr_df['corr_with_Y'].abs() < 0.1]
print(weaker_corr_df)

top7 = [
    'age',
    'hypertension',
    'heart_disease',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level',
    'smoking_history_No Info'
]

corr = df_encoded[top7].corr().round(2)
print(corr)

import pandas as pd
import statsmodels.api as sm

# 1. Välj dina features (t.ex. top7)
X = df_encoded[top7].copy()

# 2. Lägg till konstant (intercept)
X_const = sm.add_constant(X)
# 3. Tvinga allt till float (viktigt!) 
X_const = X_const.astype(float)


# 3. Beräkna VIF manuellt via OLS-hjälpregressioner
vif_list = []

for col in X_const.columns:
    # Y = variabeln vi testar
    y = X_const[col]
    
    # X = alla andra variabler
    X_others = X_const.drop(columns=[col])
    
    # OLS-modell
    model = sm.OLS(y, X_others).fit()
    
    # R^2
    R2 = model.rsquared
    
    # VIF-formeln
    vif_value = 1 / (1 - R2)
    
    vif_list.append([col, vif_value])

# 4. Snygg DataFrame
vif_df = pd.DataFrame(vif_list, columns=["feature", "VIF"])
print(vif_df)