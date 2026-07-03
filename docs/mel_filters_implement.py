import numpy as np
import matplotlib.pyplot as plt


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_filters(n_fft, n_mel, sample_rate):
    # Compute the Mel filter bank
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

    return filters

def test_mel_filters():
    n_fft = 1024
    n_mel = 128
    sample_rate = 16000
    mel_filters = mel_filters(n_fft, n_mel, sample_rate)
    print(mel_filters)

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
    # test_mel_filters()
    plot_hz_to_mel(16000)