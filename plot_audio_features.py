import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data
music_df = pd.read_csv("Scenario 2_ AI Music Composer & Listener Insight platform/Music Info.csv")

# Set up plotting style
sns.set_theme(style='whitegrid')

# Create plots directory if not exists
os.makedirs('plots', exist_ok=True)

# 1. Scatterplot of Energy vs Valence
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=music_df, x='energy', y='valence', alpha=0.6, ax=ax)
ax.set_title('Scatterplot of Energy vs Valence')
ax.set_xlabel('Energy')
ax.set_ylabel('Valence (Happiness)')
plt.tight_layout()
fig.savefig('plots/energy_vs_valence.png')
plt.close(fig)

# 2. Tempo distribution by genre
# Filter to top genres for clarity
top_genres = music_df['genre'].dropna().value_counts().head(10).index
filtered_df = music_df[music_df['genre'].isin(top_genres)]

fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=filtered_df, x='genre', y='tempo', ax=ax)
ax.set_title('Tempo Distribution by Genre')
ax.set_xlabel('Genre')
ax.set_ylabel('Tempo (BPM)')
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig('plots/tempo_by_genre.png')
plt.close(fig)

# 3. Compare loudness across moods/genres
# For moods, use valence to categorize: high valence = happy, low = sad
music_df['mood'] = pd.cut(music_df['valence'], bins=[0, 0.5, 1], labels=['Sad', 'Happy'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=music_df, x='mood', y='loudness', ax=ax)
ax.set_title('Loudness Distribution by Mood')
ax.set_xlabel('Mood')
ax.set_ylabel('Loudness (dB)')
plt.tight_layout()
fig.savefig('plots/loudness_by_mood.png')
plt.close(fig)

# Also, loudness by genre
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=filtered_df, x='genre', y='loudness', ax=ax)
ax.set_title('Loudness Distribution by Genre')
ax.set_xlabel('Genre')
ax.set_ylabel('Loudness (dB)')
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig('plots/loudness_by_genre.png')
plt.close(fig)
