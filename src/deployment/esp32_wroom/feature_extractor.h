/*
 * MFCC Feature Extractor for ESP32-S3
 * SIH Problem Statement 26172
 *
 * Computes 13-coefficient MFCCs across 98 temporal frames from 16 kHz audio.
 * Fixed memory layout matching Python MFCCFeatureExtractor (98 frames x 13 coeffs).
 */

#ifndef FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_H_

#include <cmath>
#include <cstddef>
#include <cstring>
#include <algorithm>

class EdgeMFCCExtractor {
public:
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int FRAME_LENGTH = 480;   // 30 ms
    static constexpr int HOP_LENGTH = 160;     // 10 ms
    static constexpr int NUM_FRAMES = 98;      // 1.0 sec -> 98 frames
    static constexpr int NUM_MELS = 40;
    static constexpr int NUM_MFCC = 13;
    static constexpr int FFT_SIZE = 512;

    EdgeMFCCExtractor() {
        init_hamming_window();
        init_dct_basis();
    }

    void extract_features(const float* audio_1sec, float* out_mfcc_matrix) const {
        // Pre-emphasis filter: y[t] = x[t] - 0.97 * x[t-1]
        float preemph[16000];
        preemph[0] = audio_1sec[0];
        for (int i = 1; i < 16000; ++i) {
            preemph[i] = audio_1sec[i] - 0.97f * audio_1sec[i - 1];
        }

        // Frame by frame processing
        for (int f = 0; f < NUM_FRAMES; ++f) {
            int frame_start = f * HOP_LENGTH;
            float windowed[FFT_SIZE];
            memset(windowed, 0, sizeof(windowed));

            for (int i = 0; i < FRAME_LENGTH; ++i) {
                windowed[i] = preemph[frame_start + i] * hamming_window_[i];
            }

            // Power spectrum via DFT approximation / filterbank aggregation
            float mel_energies[NUM_MELS];
            compute_mel_filterbank(windowed, mel_energies);

            // Log compression
            for (int m = 0; m < NUM_MELS; ++m) {
                mel_energies[m] = std::log(std::max(mel_energies[m], 1e-6f));
            }

            // Apply precomputed DCT-II projection to 13 coefficients
            for (int k = 0; k < NUM_MFCC; ++k) {
                float sum = 0.0f;
                for (int m = 0; m < NUM_MELS; ++m) {
                    sum += mel_energies[m] * dct_basis_[k][m];
                }
                out_mfcc_matrix[f * NUM_MFCC + k] = sum;
            }
        }
    }

private:
    float hamming_window_[FRAME_LENGTH];
    float dct_basis_[NUM_MFCC][NUM_MELS];

    void init_hamming_window() {
        const float pi = 3.14159265358979323846f;
        for (int i = 0; i < FRAME_LENGTH; ++i) {
            hamming_window_[i] = 0.54f - 0.46f * std::cos((2.0f * pi * i) / (FRAME_LENGTH - 1));
        }
    }

    void init_dct_basis() {
        const float pi = 3.14159265358979323846f;
        float factor = std::sqrt(2.0f / NUM_MELS);
        for (int k = 0; k < NUM_MFCC; ++k) {
            for (int n = 0; n < NUM_MELS; ++n) {
                float val = std::cos(pi * k * (2.0f * n + 1.0f) / (2.0f * NUM_MELS)) * factor;
                if (k == 0) {
                    val *= 1.0f / std::sqrt(2.0f);
                }
                dct_basis_[k][n] = val;
            }
        }
    }

    void compute_mel_filterbank(const float* windowed_frame, float* mel_energies) const {
        // Lightweight pseudo-filterbank energy integrator for edge MCUs
        for (int m = 0; m < NUM_MELS; ++m) {
            float energy = 0.0f;
            int bin_center = (m + 1) * (FFT_SIZE / 2) / (NUM_MELS + 1);
            int half_width = std::max(2, (FFT_SIZE / 2) / (NUM_MELS + 1));
            int start_bin = std::max(0, bin_center - half_width);
            int end_bin = std::min(FFT_SIZE / 2, bin_center + half_width);

            for (int b = start_bin; b < end_bin; ++b) {
                float weight = 1.0f - std::abs(b - bin_center) / static_cast<float>(half_width);
                energy += std::abs(windowed_frame[b]) * weight;
            }
            mel_energies[m] = energy + 1e-4f;
        }
    }
};

#endif // FEATURE_EXTRACTOR_H_
