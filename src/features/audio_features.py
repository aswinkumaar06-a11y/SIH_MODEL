"""
Base Audio Feature Extraction Framework
SIH Problem Statement 26172
"""

import abc
import numpy as np

class BaseAudioFeatureExtractor(abc.ABC):
    """
    Abstract base class for streaming and offline audio feature extractors.
    """
    def __init__(self, sample_rate=16000, win_length=480, hop_length=160, fft_length=512):
        self.sample_rate = sample_rate
        self.win_length = win_length      # 30 ms at 16 kHz
        self.hop_length = hop_length      # 10 ms at 16 kHz
        self.fft_length = fft_length      # 512 point FFT
        self.window = np.hanning(win_length).astype(np.float32)

    @abc.abstractmethod
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts spectral features from a raw 1D audio waveform.
        Returns: 2D feature matrix of shape (time_frames, feature_dim).
        """
        pass

    def frame_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Slices audio into overlapping windowed frames.
        """
        num_samples = len(audio)
        if num_samples < self.win_length:
            audio = np.pad(audio, (0, self.win_length - num_samples), mode='constant')
            num_samples = len(audio)

        num_frames = 1 + (num_samples - self.win_length) // self.hop_length
        shape = (num_frames, self.win_length)
        strides = (audio.strides[0] * self.hop_length, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        return frames * self.window
