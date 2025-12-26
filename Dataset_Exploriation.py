import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

music=pd.read_csv("Scenario 2_ AI Music Composer & Listener Insight platform/Music Info.csv")
user=pd.read_csv("Scenario 2_ AI Music Composer & Listener Insight platform/User Listening History.csv")

# checking the dataseT

# print(music.head())

# print(user.head())

# print(music.info())

# print(user.info())

# print(music.describe())

# print(user.describe())

# print(music.isnull().sum())
# print(user.isnull().sum())

# --- Genre distribution plotting (Matplotlib / Seaborn) ---



def plot_genre_distribution(df, top_n=10, save_dir='plots', show=True):
    """Plot and save the top-N genre distribution from the provided DataFrame.

    Expects a `genre` column (can contain NaN). Saves a horizontal bar plot to `save_dir`.
    Set `show=False` in headless / automated runs to avoid blocking.
    """
    # Prepare counts (fill missing genres with 'Unknown')
    genre_series = df['genre'].fillna('Unknown').astype(str).str.strip()
    counts = genre_series.value_counts()

    # Top-N genres
    top_counts = counts.head(top_n)

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_counts.values, y=top_counts.index, ax=ax, palette='viridis')
    ax.set_title(f'Top {top_n} Genres')
    ax.set_xlabel('Number of tracks')
    ax.set_ylabel('Genre')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'genre_distribution_top_{top_n}.png')
    fig.savefig(out_path)
    print(f'✅ Saved genre distribution plot to: {out_path}')

    if show:
        try:
            plt.show()
        except Exception:
            print('⚠️ Unable to display plot (headless environment); file saved.')


if __name__ == '__main__':
    # When run directly, save plots but don't block on display in headless contexts
    plot_genre_distribution(music, top_n=10, show=False)

