# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Läs in träningsfilen
df = pd.read_csv(r"C:\Users\Eier\OneDrive\Dokument\DataScience\Data_science_projektkurs\archive (1)\diabetes_prediction_dataset.csv")

def jupyter_info(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Non-null": df.notnull().sum().values,
        "Nulls": df.isnull().sum().values,
        "Unique": df.nunique().values,
        "Dtype": df.dtypes.astype(str).values
    })
    
print(jupyter_info(df))
# %%
df.describe().T.style.set_table_styles([
    {"selector": "th", "props": "background-color: #f2f2f2; font-weight: bold;"},
    {"selector": "td", "props": "padding: 6px;"}
]).format("{:.2f}")


# %%
df.shape


numerical = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

for col in numerical:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    

# Räkna antal och procent
import matplotlib.pyplot as plt
import seaborn as sns

# Antag att kolumnen heter 'diabetes'
counts = df["diabetes"].value_counts().sort_index()
perc = df["diabetes"].value_counts(normalize=True).sort_index() * 100

plt.figure(figsize=(5, 4))
sns.barplot(x=counts.index, y=counts.values, palette="Greys")

plt.title("Fördelning av målvariabeln Y")
plt.xlabel("Klass (Y)")
plt.ylabel("Antal observationer")

max_count = counts.max()

# antal ovanför, procent under
for i, (count, p) in enumerate(zip(counts.values, perc.values)):
    plt.text(i, count + max_count*0.03, str(count), ha="center")
    plt.text(i, -max_count*0.08, f"{p:.1f}%", ha="center")

plt.ylim(-max_count*0.15, max_count*1.15)
plt.tight_layout()
plt.show()




numerical = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

df[numerical].skew()
