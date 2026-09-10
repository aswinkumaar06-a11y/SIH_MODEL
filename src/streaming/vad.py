"""
Energy Voice Activity Detector (VAD) for Edge Inference Gating
SIH Problem Statement 26172 - Milestone 10

Ultra-lightweight time-domain speech detector:
1. Calculates RMS energy and Zero Crossing Rate (ZCR)
2. Tracks dynamic background noise floor
3. Incorporates hangover smoothing to preserve word endings
4. Skips heavy CNN inference during silence to enforce < 10% idle CPU utilization
"""

import numpy as np

class EnergyVAD:
    def __init__(self, sample_rate=16000, min_energy_threshold=0.005,
                 energy_multiplier=2.5, hangover_frames=3):
        self.sample_rate = sample_rate
        self.min_energy_threshold = float(min_energy_threshold)
        self.energy_multiplier = float(energy_multiplier)
        self.hangover_frames = int(hangover_frames)

        # Dynamic background noise floor tracker
        self.background_energy = self.min_energy_threshold
        self.hangover_counter = 0

    def compute_energy_and_zcr(self, chunk: np.ndarray):
        """Computes RMS energy and Zero-Crossing Rate."""
        if len(chunk) == 0:
            return 0.0, 0.0
        # RMS energy
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        # ZCR
        signs = np.sign(chunk)
        signs[signs == 0] = 1
        zcr = float(np.mean(np.abs(signs[1:] - signs[:-1])) / 2.0) if len(chunk) > 1 else 0.0
        return rms, zcr

    def is_speech(self, chunk: np.ndarray) -> bool:
        """
        Determines whether the incoming chunk contains speech.
        Updates dynamic noise floor during quiet periods.
        """
        rms, zcr = self.compute_energy_and_zcr(chunk)
        threshold = max(self.background_energy * self.energy_multiplier, self.min_energy_threshold)

        if rms >= threshold:
            # Active speech detected
            self.hangover_counter = self.hangover_frames
            return True

        # Silence / ambient noise: adapt background noise floor
        alpha = 0.05
        self.background_energy = (1.0 - alpha) * self.background_energy + alpha * rms

        # Hangover decay
        if self.hangover_counter > 0:
            self.hangover_counter -= 1
            return True

        return False

    def reset(self):
        self.background_energy = self.min_energy_threshold
        self.hangover_counter = 0
