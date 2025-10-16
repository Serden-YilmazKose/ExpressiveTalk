"""Extract audio to mel feature spectograms"""

####### The following resources were used in the research of this code: #######
# https://github.com/rhasspy/wav2mel
# https://www.kaggle.com/code/gaurav41/how-to-convert-audio-to-mel-spectrogram-to-audio
# https://www.youtube.com/watch?v=g8Q452PEXwY
# https://clouddatascience.medium.com/mel-spectrograms-with-python-and-librosa-audio-feature-extraction-4ab18c14797c
# https://www.hackersrealm.net/post/extract-features-from-audio-mfcc-python

# import librosa
# import librosa.display
# import IPython.display as ip
# import matplotlib.pyplot as plt
# import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def extract_mel_features(path):
    """Input file path and output mel spectogram"""
    # Load Audio File
    try:
        # Returns floting point times series (y), and sampling rate (sr)
        y, sr = librosa.load(path)
    # If it doesnt exist, warn the use
    except FileNotFoundError:
        print(f"File {path} not found.")
        return
    # Any other error, warn the user
    except Exception as e:
        print(f"An error happened: {e}")
        return 0

    # Extract features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50).T, axis=0)

    return mfcc
    # Extract Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to Decibels (Log Scale)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spectrogram_db, x_axis="time", y_axis="mel", sr=sr, cmap="viridis"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.show()
    # return


if __name__ == "__main__":
    # Specify path
    audio_path = "./Joe_Biden.ogg"
    # audio_path = "./doesnt_exist.ogg"
    features = extract_mel_features(audio_path)
    print(features)
