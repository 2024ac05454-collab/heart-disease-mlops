import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder to save our report images
os.makedirs("outputs", exist_ok=True)

print("Loading dataset...")
# Update this path if your data is stored somewhere else!
# Assuming standard UCI Heart Disease format. 
# If you are fetching it directly from the web in train.py, use that same method here.
try:
    df = pd.read_csv("heart.csv") # Change to your actual data path
except FileNotFoundError:
    print("Could not find local CSV, fetching from UCI directly...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=cols, na_values="?")
    df.dropna(inplace=True)

# 1. Class Balance (Target Variable)
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Class Balance: Heart Disease Presence (0 = No, 1+ = Yes)')
plt.savefig("outputs/class_balance.png", bbox_inches='tight')
plt.close()
print("Saved outputs/class_balance.png")

# 2. Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig("outputs/correlation_heatmap.png", bbox_inches='tight')
plt.close()
print("Saved outputs/correlation_heatmap.png")

# 3. Histograms for Numerical Features
df.hist(figsize=(15, 12), bins=15, color='steelblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features', y=1.02, size=16)
plt.tight_layout()
plt.savefig("outputs/histograms.png", bbox_inches='tight')
plt.close()
print("Saved outputs/histograms.png")

print("EDA complete! Check the 'outputs' folder for your screenshots.")