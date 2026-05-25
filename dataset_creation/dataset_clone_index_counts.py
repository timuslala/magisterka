import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytaj dataset
df = pd.read_csv("test_dataset_50_50.csv")

# Przygotowanie danych do wykresu
df_melt = pd.melt(df, 
                  id_vars=['label'], 
                  value_vars=['code1_clone_index', 'code2_clone_index'],
                  var_name='position', 
                  value_name='clone_index')

# Mapowanie label na czytelną nazwę
df_melt['type'] = df_melt['label'].map({1: 'Clone (label=1)', 0: 'Non-Clone (label=0)'})

# Liczebność dla każdego clone_index
plt.figure(figsize=(14, 8))
sns.countplot(data=df_melt, x='clone_index', hue='type', palette=['#ff6b6b', '#4ecdc4'])

plt.title('Liczebność wystąpień clone_index w dataset testowym\n(50 par klonów + 50 par non-klonów)', 
          fontsize=16, pad=20)
plt.xlabel('Clone Index')
plt.ylabel('Liczba wystąpień (code1 + code2)')
plt.legend(title='Typ pary')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Dodanie wartości nad słupkami
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%d', padding=3)

plt.tight_layout()
plt.savefig("clone_index_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# Statystyki
print("=== STATYSTYKI ===")
print(df['code1_clone_index'].value_counts().sort_index().head(10))
print("\nLiczba unikalnych clone_index w teście:", df['code1_clone_index'].nunique())