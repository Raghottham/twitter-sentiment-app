import pandas as pd

# Load full dataset (without column names)
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)

# Add column names manually
df.columns = ['Target', 'ID', 'Date', 'Flag', 'User', 'Text']

# Keep only 0 (negative) and 4 (positive)
df = df[df['Target'].isin([0, 4])]

# Sample 25000 of each class
df_neg = df[df['Target'] == 0].sample(n=25000, random_state=42)
df_pos = df[df['Target'] == 4].sample(n=25000, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=42)

# Save smaller file
df_balanced.to_csv('tiny_dataset.csv', index=False, header=False, encoding='ISO-8859-1')

print("✅ Done! Saved balanced dataset as 'tiny_dataset.csv'")
