"""
Unified Dynamic Custom Keyword Enrollment Utility
SIH Problem Statement 26172 (ISRO)

Enrolls ANY new unseen custom keyword into the Edge Voice Activator with ZERO retraining:
Supports:
1. Microphone mode: Record 3 samples directly from your laptop mic (--mic)
2. Synthetic mode: Auto-generate 3 samples across multiple voices/rates (--synth)
3. Custom audio folder mode: Provide your own 16 kHz WAV files (--audio-dir)

Outputs:
- 32-D L2-normalized prototype centroid vector
- Updated ESP32-S3 C++ firmware header ('src/deployment/esp32/keyword_prototype.h')
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
import soundfile as sf
import sounddevice as sd
import scipy.signal
import tensorflow as tf

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor


def record_from_mic(keyword: str, output_dir: str, num_samples: int = 3, sr: int = 16000) -> list:
    """Interactively records K audio clips from the laptop microphone."""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    duration = 1.0  # 1.0 second

    print("\n" + "=" * 70)
    print(f"MICROPHONE ENROLLMENT: '{keyword.upper()}'")
    print(f"You will record {num_samples} audio utterances (1 second each).")
    print("=" * 70)

    for i in range(1, num_samples + 1):
        input(f"\n[{i}/{num_samples}] Get ready, press [ENTER] and say '{keyword.upper()}' clearly...")
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

        file_path = os.path.join(output_dir, f"{keyword.lower()}_mic_sample_{i}.wav")
        sf.write(file_path, audio, sr)
        saved_files.append(file_path)

    print(f"\nRecorded {len(saved_files)} samples successfully into {output_dir}!")
    return saved_files


def synthesize_samples(keyword: str, output_dir: str, num_samples: int = 3, target_sr: int = 16000) -> list:
    """Auto-synthesizes K audio clips using Windows SAPI TTS across voices and rates."""
    os.makedirs(output_dir, exist_ok=True)
    voices = ["Microsoft David Desktop", "Microsoft Zira Desktop"]
    rates = [-1, 0, 1]
    
    generated_files = []
    count = 0

    print(f"\nAuto-synthesizing {num_samples} samples for '{keyword.upper()}' via Windows SAPI...")
    for voice in voices:
        v_tag = "david" if "David" in voice else "zira"
        for rate in rates:
            if count >= num_samples:
                break
            raw_wav = os.path.join(output_dir, f"temp_{v_tag}_r{rate}_{count}.wav").replace("\\", "/")
            final_wav = os.path.join(output_dir, f"{keyword.lower()}_{v_tag}_r{rate}_{count}.wav")

            ps_cmd = (
                f"Add-Type -AssemblyName System.Speech; "
                f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.SelectVoice('{voice}'); "
                f"$s.Rate = {rate}; "
                f"$s.SetOutputToWaveFile('{raw_wav}'); "
                f"$s.Speak('{keyword}'); "
                f"$s.Dispose()"
            )
            subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)

            if os.path.exists(raw_wav):
                audio, orig_sr = sf.read(raw_wav, dtype="float32")
                if audio.ndim > 1:
                    audio = audio[:, 0]
                
                num_target_samples = int(len(audio) * (target_sr / orig_sr))
                audio_16k = scipy.signal.resample(audio, num_target_samples)

                if len(audio_16k) < target_sr:
                    pad = target_sr - len(audio_16k)
                    audio_16k = np.pad(audio_16k, (0, pad), mode="constant")
                else:
                    audio_16k = audio_16k[:target_sr]

                max_val = np.max(np.abs(audio_16k))
                if max_val > 0:
                    audio_16k = (audio_16k / max_val) * 0.95

                sf.write(final_wav, audio_16k.astype(np.float32), target_sr, subtype="PCM_16")
                os.remove(raw_wav)
                generated_files.append(final_wav)
                count += 1

    print(f"Synthesized {len(generated_files)} audio samples into {output_dir}!")
    return generated_files


def enroll_keyword(
    keyword_name: str,
    audio_files: list,
    model_path: str = r"D:\SIH_Model\models\tflite\voice_activator_int8.tflite",
    output_header: str = r"D:\SIH_Model\src\deployment\esp32\keyword_prototype.h"
) -> dict:
    """Passes audio through the frozen INT8 model and exports the prototype header."""
    feature_extractor = MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]
    in_dtype = interpreter.get_input_details()[0]["dtype"]

    embeddings = []
    for fpath in audio_files:
        audio, sr = sf.read(fpath, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
        else:
            audio = audio[:16000]

        mfcc = feature_extractor.extract(audio)  # (98, 13)
        tensor = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1).astype(in_dtype)

        interpreter.set_tensor(in_idx, tensor)
        interpreter.invoke()
        emb = interpreter.get_tensor(out_idx)[0]

        norm = np.linalg.norm(emb)
        emb_norm = emb / (norm + 1e-9)
        embeddings.append(emb_norm)

    embeddings = np.array(embeddings)
    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    # Intra-shot similarity
    pair_sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            pair_sims.append(float(np.dot(embeddings[i], embeddings[j])))
    intra_sim = float(np.mean(pair_sims)) if pair_sims else 1.0

    # Write C header
    proto_c_str = ", ".join([f"{v:.7f}f" for v in centroid])
    header_content = f"""/*
 * Auto-generated Keyword Prototype Header for ESP32-S3
 * SIH Problem Statement 26172
 * Keyword: {keyword_name.upper()}
 * Embedding Dimension: {len(centroid)}
 * Hypersphere Normalized: True (||p||_2 = 1.0)
 */

#ifndef KEYWORD_PROTOTYPE_H_
#define KEYWORD_PROTOTYPE_H_

#define ENROLLED_KEYWORD_NAME "{keyword_name.upper()}"
#define KEYWORD_PROTOTYPE_DIM {len(centroid)}

// L2-normalized prototype vector centroid on 32-D unit hypersphere
static const float KEYWORD_PROTOTYPE[KEYWORD_PROTOTYPE_DIM] = {{
    {proto_c_str}
}};

#endif // KEYWORD_PROTOTYPE_H_
"""
    with open(output_header, "w", encoding="utf-8") as f:
        f.write(header_content)

    return {
        "keyword": keyword_name.upper(),
        "shots_count": len(audio_files),
        "intra_similarity": round(intra_sim, 4),
        "prototype": centroid,
        "header_path": output_header
    }


def main():
    parser = argparse.ArgumentParser(description="Enroll any new custom keyword into the Edge Voice Activator")
    parser.add_argument("--keyword", required=True, help="Custom keyword name (e.g., HELIOS, APOLLO, JARVIS)")
    parser.add_argument("--mic", action="store_true", help="Record 3 samples interactively using laptop microphone")
    parser.add_argument("--synth", action="store_true", help="Auto-synthesize 3 samples using TTS (no mic needed)")
    parser.add_argument("--audio-dir", default=None, help="Directory containing your own custom WAV files")
    parser.add_argument("--shots", type=int, default=3, help="Number of enrollment shots (default: 3)")

    args = parser.parse_args()
    kw = args.keyword.strip().upper()
    kw_dir = os.path.join(r"D:\SIH_Model\data\raw\custom_keywords", kw.lower())

    print("=" * 80)
    print(f"DYNAMIC KEYWORD ENROLLMENT: '{kw}'")
    print("Edge Target: ESP32-S3 | Architecture: Universal Speech Embedding Space")
    print("Zero Retraining | Zero Weight Modifications | Zero Flash Re-baking")
    print("=" * 80)

    # Acquire audio clips
    if args.audio_dir and os.path.exists(args.audio_dir):
        audio_files = [os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir) if f.endswith(".wav")][:args.shots]
        print(f"Loaded {len(audio_files)} user audio files from {args.audio_dir}")
    elif args.mic:
        audio_files = record_from_mic(kw, kw_dir, num_samples=args.shots)
    elif args.synth:
        audio_files = synthesize_samples(kw, kw_dir, num_samples=args.shots)
    else:
        # Default: if existing files exist, use them; otherwise record or synth
        if os.path.exists(kw_dir) and len([f for f in os.listdir(kw_dir) if f.endswith(".wav")]) >= args.shots:
            audio_files = [os.path.join(kw_dir, f) for f in os.listdir(kw_dir) if f.endswith(".wav")][:args.shots]
            print(f"Using {len(audio_files)} existing audio samples from {kw_dir}")
        else:
            print("No audio input mode specified. Defaulting to auto-synthesis (--synth)...")
            audio_files = synthesize_samples(kw, kw_dir, num_samples=args.shots)

    # Pass through INT8 model and export C header
    print(f"\nPassing {len(audio_files)} samples through frozen INT8 TFLite encoder...")
    result = enroll_keyword(kw, audio_files)

    print("-" * 80)
    print(f"ENROLLMENT SUCCESSFUL!")
    print(f"  Target Keyword:        '{result['keyword']}'")
    print(f"  Enrolled Shots:        {result['shots_count']} audio clips")
    print(f"  Intra-Shot Similarity: {result['intra_similarity']:.4f}")
    print(f"  Centroid Prototype:    32-D unit hypersphere (||p|| = 1.000)")
    print(f"  Firmware Header:       {result['header_path']}")
    print("=" * 80)
    print(f"\nHOW TO TEST '{kw}' NOW ON YOUR LAPTOP:")
    print(f"  1. Live Laptop Mic:     & \"D:\\SIH_Model\\.venv\\Scripts\\python.exe\" scripts/live_mic_activator.py --keyword {kw}")
    print(f"  2. Speaker Simulation:  & \"D:\\SIH_Model\\.venv\\Scripts\\python.exe\" scripts/demo_voice_activator.py --keyword {kw} --play")
    print("=" * 80)


if __name__ == "__main__":
    main()
