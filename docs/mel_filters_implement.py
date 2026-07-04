import numpy as np
import matplotlib.pyplot as plt

def show_spectrogram(audio_file):
    import librosa
    import librosa.display

    n_fft = 400
    hop_length = 200
    n_mels = 80
    y, sr = librosa.load(audio_file, sr=None)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # 幅度谱
    S_magnitude = np.abs(D)

    # 功率谱
    S_power = S_magnitude**2

    S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max)
    S_power_db = librosa.power_to_db(S_power, ref=np.max)

    # 梅尔谱
    S_mel = librosa.feature.melspectrogram(
        S=S_power,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    # 同时显示原始波形、幅度谱和梅尔谱
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('Waveform')
    axes[0].set_ylabel('Amplitude')

    img1 = librosa.display.specshow(
        S_db,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz',
        ax=axes[1],
    )
    axes[1].set_title('Magnitude Spectrogram (dB)')
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    img2 = librosa.display.specshow(
        S_mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=axes[2],
    )
    axes[2].set_title('Mel Spectrogram (dB)')
    fig.colorbar(img2, ax=axes[2], format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def show_mel_filters(n_fft, n_mel, sample_rate):
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mel + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mel, n_fft // 2 + 1))
    for i in range(1, n_mel + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = (
            np.linspace(0, 1, bin_points[i] - bin_points[i - 1])
        )
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = (
            np.linspace(1, 0, bin_points[i + 1] - bin_points[i])
        )

    frequency_bins = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for i in range(n_mel):
        axes[0].plot(frequency_bins, filters[i])
    axes[0].set_title('Mel Filter Bank')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Weight')
    axes[0].set_xlim(0, sample_rate / 2)
    axes[0].grid(True, alpha=0.3)

    img = axes[1].imshow(
        filters,
        origin='lower',
        aspect='auto',
        extent=[0, sample_rate / 2, 0, n_mel],
    )
    axes[1].set_title('Mel Filter Bank Matrix')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Mel filter index')
    fig.colorbar(img, ax=axes[1], label='Weight')

    plt.tight_layout()
    plt.show()

    return filters


def show_mel_filters_by_librosa(n_fft, n_mel, sample_rate, htk=False, norm='slaney'):
    import librosa

    filters = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mel,
        htk=htk,
        norm=norm,
    )
    frequency_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for i in range(n_mel):
        axes[0].plot(frequency_bins, filters[i])
    axes[0].set_title('Mel Filter Bank by librosa')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Weight')
    axes[0].set_xlim(0, sample_rate / 2)
    axes[0].grid(True, alpha=0.3)

    img = axes[1].imshow(
        filters,
        origin='lower',
        aspect='auto',
        extent=[0, sample_rate / 2, 0, n_mel],
    )
    axes[1].set_title('Mel Filter Bank Matrix by librosa')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Mel filter index')
    fig.colorbar(img, ax=axes[1], label='Weight')

    plt.tight_layout()
    plt.show()

    return filters


def plot_hz_to_mel(sample_rate):
    hz = np.linspace(0, sample_rate / 2, 1000)
    mel = hz_to_mel(hz)
    
    plt.plot(hz, mel)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mel')
    plt.title('Frequency to Mel Scale')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # audio_file = '/Volumes/tiger/Workspace/side-projects/2026/01_VoiceAssistant/whisper.mars/samples/jfk.wav'
    # show_spectrogram(audio_file)
    # plot_hz_to_mel(16000)
    show_mel_filters(n_fft=400, n_mel=80, sample_rate=16000)
    show_mel_filters_by_librosa(n_fft=400, n_mel=80, sample_rate=16000)
