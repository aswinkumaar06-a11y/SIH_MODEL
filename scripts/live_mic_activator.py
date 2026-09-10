"""
Live Laptop Microphone Edge Voice Activator
SIH Problem Statement 26172 (ISRO)

Allows real-time testing of the INT8 Quantized Edge Voice Activator directly through your laptop microphone:
1. Streams continuous audio in 50ms chunks (800 samples @ 16 kHz) via laptop microphone array.
2. Performs real-time Ring Buffer -> VAD Gating -> INT8 TFLite Universal Feature Encoder -> Hysteresis State Machine.
3. Automatically triggers low-latency activation when the enrolled keyword is spoken.
4. Captures 2.0s post-wake speech buffer and dispatches it for ASR handover.
"""

import os
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf

# Suppress TF logging clutter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

sys.path.insert(0, r"D:\SIH_Model")
from src.streaming.demo_pipeline import EndToEndVoiceActivatorDemo


def record_user_samples(keyword: str, output_dir: str, num_samples: int = 3, sr: int = 16000) -> list:
    """Interactively records K audio clips directly from the user's laptop microphone."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    duration = 1.0  # 1 second per utterance

    print("\n" + "=" * 75)
    print(f"PERSONALIZED VOICE ENROLLMENT FOR '{keyword.upper()}'")
    print(f"We will record {num_samples} audio clips (1.0 second each) of you speaking.")
    print("=" * 75)

    for i in range(1, num_samples + 1):
        input(f"\n[{i}/{num_samples}] Get ready, then press [ENTER] and say '{keyword.upper()}' clearly...")
        sys.stdout.write("  >> RECORDING (1 second)... ")
        sys.stdout.flush()
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()
        print("Done!")

        # Normalize audio amplitude
        max_amp = np.max(np.abs(audio))
        if max_amp > 0.01:
            audio = (audio / max_amp) * 0.90

        file_path = os.path.join(output_dir, f"{keyword.lower()}_user_sample_{i}.wav")
        sf.write(file_path, audio, sr)
        saved_paths.append(file_path)

    print(f"\nAll {num_samples} samples successfully recorded and saved to {output_dir}!")
    return saved_paths


def run_live_mic(keyword: str = "ZORA", threshold: float = 0.87, record_user: bool = False):
    kw = keyword.strip().upper()
    print("=" * 80)
    print(f"SIH 26172: LIVE LAPTOP MICROPHONE VOICE ACTIVATOR")
    print(f"Hardware Edge Target: ESP32-S3 Profile (< 100 KB Flash, < 256 KB SRAM)")
    print(f"Active Wake Word:     '{kw}'")
    print("=" * 80)

    # 1. Acquire Enrollment Samples
    kw_dir = os.path.join(r"D:\SIH_Model\data\raw\custom_keywords", kw.lower())
    if record_user:
        user_dir = os.path.join(kw_dir, "user_voice")
        enrollment_paths = record_user_samples(kw, user_dir, num_samples=3)
    else:
        # Check if pre-enrolled samples exist
        if os.path.exists(kw_dir):
            enrollment_paths = [os.path.join(kw_dir, f) for f in os.listdir(kw_dir) if f.endswith(".wav")][:3]
        else:
            enrollment_paths = []

        if not enrollment_paths:
            print(f"\nNo pre-recorded samples found for '{kw}'. Starting quick recording...")
            enrollment_paths = record_user_samples(kw, kw_dir, num_samples=3)

    # 2. Initialize Activator Pipeline
    model_path = r"D:\SIH_Model\models\tflite\voice_activator_int8.tflite"
    activator = EndToEndVoiceActivatorDemo(
        tflite_model_path=model_path,
        sample_rate=16000,
        chunk_size_ms=50,
        tau_high=threshold,
        tau_low=threshold - 0.04,
        persistence=2
    )

    enroll_info = activator.enroll_keyword(enrollment_paths, keyword_name=kw)
    print(f"\n  [ENROLLMENT STATUS]")
    print(f"    - Target Keyword:        '{enroll_info['keyword']}'")
    print(f"    - Enrollment Utterances: {enroll_info['shots_count']} audio clips")
    print(f"    - Intra-Shot Similarity: {enroll_info['intra_similarity']:.4f} (high acoustic coherence)")
    print(f"    - Operating Thresholds:  tau_high = {threshold:.2f}, tau_low = {threshold-0.04:.2f}, N = 2 windows")
    print(f"    - Input Audio Device:    {sd.query_devices(kind='input')['name']}")
    print("-" * 80)
    print("  * Speak normally into your laptop microphone.")
    print("  * Say '" + kw + "' clearly.")
    print("  * Press Ctrl + C in the terminal to stop at any time.")
    print("-" * 80 + "\n")

    chunk_samples = 800  # 50 ms @ 16,000 Hz
    output_test_dir = r"D:\SIH_Model\outputs\live_test"
    os.makedirs(output_test_dir, exist_ok=True)

    start_time = time.time()
    activation_count = 0

    try:
        with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=chunk_samples) as stream:
            while True:
                audio_chunk, overflowed = stream.read(chunk_samples)
                audio_chunk = audio_chunk.flatten()
                current_time_ms = (time.time() - start_time) * 1000.0

                # Compute chunk RMS volume for visualization
                rms = np.sqrt(np.mean(audio_chunk**2) + 1e-9)
                vol_pct = int(min(100, rms * 500))

                res = activator.process_chunk(audio_chunk, current_time_ms)
                ev = res.get("event")

                if ev == "VAD_SKIP":
                    bar = "░" * 10
                    sys.stdout.write(f"\r[MIC IDLE    ] [{bar}] Vol: {vol_pct:2d}% | Sim: 0.000 | State: {res.get('state', 'IDLE'):<10}    ")
                    sys.stdout.flush()
                elif ev == "INFERENCE":
                    sim = res.get("smoothed_sim", 0.0)
                    filled = int(max(0, min(10, sim * 10)))
                    bar = "█" * filled + "░" * (10 - filled)
                    sys.stdout.write(f"\r[MIC SPEECH  ] [{bar}] Vol: {vol_pct:2d}% | Sim: {sim:.3f} | State: {res.get('state'):<10}    ")
                    sys.stdout.flush()
                elif ev == "ACTIVATION_TRIGGERED":
                    activation_count += 1
                    sim = res.get("smoothed_sim", 0.0)
                    sys.stdout.write("\n\n" + "=" * 80 + "\n")
                    sys.stdout.write(f"  >>> [ACTIVATION TRIGGERED #{activation_count}] '{kw}' DETECTED! <<<\n")
                    sys.stdout.write(f"  Timestamp: {current_time_ms/1000.0:.2f}s | Confidence: {sim:.4f} >= {threshold:.2f}\n")
                    sys.stdout.write(f"  Action: Dispatched edge wake signal & capturing 2.0s ASR speech buffer...\n")
                    sys.stdout.write("=" * 80 + "\n\n")
                    sys.stdout.flush()
                elif ev == "ASR_HANDOVER_DISPATCHED":
                    asr_info = res.get("asr_transcription", {})
                    save_wav = os.path.join(output_test_dir, f"wake_command_{activation_count}.wav")
                    if "audio_payload" in res:
                        sf.write(save_wav, res["audio_payload"], 16000)
                    sys.stdout.write(f"  >>> [ASR HANDOVER COMPLETED]: Staged {res.get('buffer_duration_sec', 2.0)}s speech payload to {asr_info.get('asr_engine')}.\n\n")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n\n[TEST ENDED] Live microphone session stopped by user.")
        print(f"Total activations triggered: {activation_count}")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Live Laptop Microphone Voice Activator")
    parser.add_argument("--keyword", default="ZORA", help="Target keyword name (default: ZORA)")
    parser.add_argument("--threshold", type=float, default=0.87, help="Activation similarity threshold (default: 0.87)")
    parser.add_argument("--record-user", action="store_true", help="Record 3 enrollment samples using your own voice")
    args = parser.parse_args()
    run_live_mic(keyword=args.keyword, threshold=args.threshold, record_user=args.record_user)


if __name__ == "__main__":
    main()
