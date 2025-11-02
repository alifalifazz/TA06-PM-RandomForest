import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Membaca Dataset ===
df = pd.read_csv("dataset/diabetes.csv")

# === 2. Menampilkan 5 Baris Pertama ===
print("--- 5 Baris Pertama Data ---")
print(df.head())

# === 3. Visualisasi Distribusi Target (Outcome) ===
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df, hue='Outcome', legend=False, palette='pastel')
plt.title('Distribusi Kelas Target (1 = Diabetes, 0 = Tidak Diabetes)')
plt.xlabel('Kelas Target (Outcome)')
plt.ylabel('Jumlah Data')
plt.show()
