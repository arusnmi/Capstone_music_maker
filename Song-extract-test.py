import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from glob import glob
import seaborn as sns
import librosa
import librosa.display

audio_files = glob(
    r"C:\Users\warty\OneDrive\Desktop\Python_projects\Capstone_music_maker\Scenario 2_ AI Music Composer & Listener Insight platform\MP3-Example\**\*.mp3"
)

#y=raw song data, sr=sampling rate

y,sr=librosa.load(audio_files[0])


pd.Series(y[13000, 26000]).plot(figsize=(14,5))

plt.show()

