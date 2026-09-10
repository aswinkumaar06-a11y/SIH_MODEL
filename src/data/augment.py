"""
Audio Data Augmentation Pipeline
SIH Problem Statement 26172
"""

import numpy as np

class AudioAugmenter:
    """
    Configurable audio data augmenter for TinyML training robustness.
    """
    def __init__(self, sample_rate=16000, seed=42):
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def add_noise(self, audio, snr_db_low=10, snr_db_high=30):
        """Adds white Gaussian noise at random SNR."""
        snr_db = self.rng.uniform(snr_db_low, snr_db_high)
        signal_power = np.mean(audio ** 2)
        if signal_power == 0:
            return audio
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
        return audio + noise

    def time_shift(self, audio, max_shift_ms=100):
        """Randomly time-shifts audio within +/- max_shift_ms."""
        max_shift = int(self.sample_rate * (max_shift_ms / 1000.0))
        shift = self.rng.integers(-max_shift, max_shift)
        return np.roll(audio, shift)

    def apply_gain(self, audio, min_gain_db=-6, max_gain_db=6):
        """Applies random gain variation."""
        gain_db = self.rng.uniform(min_gain_db, max_gain_db)
        factor = 10 ** (gain_db / 20.0)
        return audio * factor

    def augment(self, audio, p_noise=0.5, p_shift=0.5, p_gain=0.5):
        """Applies random chain of augmentations."""
        augmented = audio.copy()
        if self.rng.random() < p_shift:
            augmented = self.time_shift(augmented)
        if self.rng.random() < p_noise:
            augmented = self.add_noise(augmented)
        if self.rng.random() < p_gain:
            augmented = self.apply_gain(augmented)
        return np.clip(augmented, -1.0, 1.0)
