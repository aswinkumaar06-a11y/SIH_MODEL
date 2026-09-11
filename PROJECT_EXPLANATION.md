# SIH 26172: Low Latency & Efficient Voice Activator for Edge Devices

**Smart India Hackathon (SIH) — Problem Statement 26172**  
**Organization:** Indian Space Research Organisation (ISRO)  
**Domain:** Edge AI / TinyML / Embedded Acoustic Intelligence  
**Target Hardware:** ESP32-WROOM-32 / ESP32-S3 Bare-Metal Microcontroller  
**Repository:** [https://github.com/aswinkumaar06-a11y/SIH_MODEL](https://github.com/aswinkumaar06-a11y/SIH_MODEL)

---

## 1. Executive Summary & Problem Definition

In space exploration, aeronautics, and mission-critical edge computing environments, human operators require voice-driven machine interaction that functions **100% offline**, operates with **sub-50 millisecond latency**, consumes **milliwatts of power**, and runs entirely on **sub-$5 microcontrollers** without internet or cloud dependencies.

### Key Operational Challenges
1. **Extreme Resource Constraints:** Microcontrollers like the **ESP32-WROOM-32** possess only ~320 KB of usable SRAM and 4 MB of Flash memory. Conventional deep speech recognition models (Whisper, Conformer, Kaldi) require hundreds of megabytes or gigabytes of RAM.
2. **The "Unseen Keyword" Dilemma (Model Collapse):** Standard wake-word models are trained as static $N$-class classifiers (e.g. Class 0 = *"AGNI"*, Class 1 = *"VAYU"*). When an evaluation committee or astronaut introduces an un-modeled keyword (e.g. **"ZORA"**, **"HELIOS"**, or **"JUMP"**), a fixed-class model cannot activate on it without modifying network weights and running full backpropagation, which is mathematically impossible to train on a micro-device.
3. **No Proprietary Wake-Word SDKs Allowed:** ISRO problem guidelines strictly mandate open-source frameworks (TensorFlow Lite Micro, C/C++), barring closed-source vendor solutions (e.g., Picovoice Porcupine, Sensory TrulyHandsfree).

---

## 2. Core Architectural Innovation: Universal Metric Learning

Rather than training a rigid classifier, our system utilizes **Deep Metric Learning on a 32-Dimensional Unit Hypersphere $\mathbb{S}^{31}$**.

```
                           +-------------------------------------+
                           | 16 kHz Audio Stream (50ms chunks)   |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | Audio Ring Buffer (16,000 samples)  |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | Energy VAD (Pre-Filter Silence)     |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | 13-Band MFCC Extraction (98x13 grid)|
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | Frozen INT8 Universal CNN Encoder   |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | 32-D L2-Normalized Embedding z in S^31|
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | Cosine Sim with Prototype p (p . z) |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | Dual-Threshold Hysteresis SM (N=3)  |
                           +-------------------------------------+
                                       /              \
                                      v                v
                        [Trigger Edge GPIO Wake]   [Handover to ASR]
```

### Mathematical Formulation
1. **Universal Acoustic Feature Encoder:** A lightweight Depthwise Separable Convolutional Neural Network $f_\theta(x)$ is trained using **Supervised Contrastive Loss** on diverse acoustic corpora (Google Speech Commands v2 + Mozilla Common Voice).
2. **Hypersphere Projection:** For any input spectrogram $x$, the network produces an embedding vector $z \in \mathbb{R}^{32}$ that is strictly $L_2$-normalized:
   $$\hat{z} = \frac{f_\theta(x)}{\|f_\theta(x)\|_2}, \quad \|\hat{z}\|_2 = 1.0$$
3. **Zero-Retraining Few-Shot Centroid Enrollment:** When an operator wants to enroll any new word $W$, they speak it $K$ times ($K=3$). The frozen encoder computes representations $\{\hat{z}_1, \hat{z}_2, \dots, \hat{z}_K\}$. The prototype centroid $\mathbf{p}$ is computed as:
   $$\mathbf{p}_{\text{raw}} = \frac{1}{K}\sum_{k=1}^K \hat{z}_k, \quad \mathbf{p} = \frac{\mathbf{p}_{\text{raw}}}{\|\mathbf{p}_{\text{raw}}\|_2}$$
4. **Why "Model Collapse" is Impossible:** The base neural network weights $\theta$ (`voice_activator_int8.tflite`) are **frozen INT8 constants** compiled into read-only flash memory. Enrolling a new keyword merely writes **32 floating-point numbers (128 bytes)** into RAM or Non-Volatile Flash (NVS). The underlying model is never retrained or modified.

---

## 3. Acoustic Signal Pipeline & Anti-Degeneracy Defense

A critical finding during real-world microphone evaluation is **Embedding Space Anisotropy**: raw neural speech embeddings naturally exhibit a directional bias towards ambient room silence. If an enrollment session contains silence or low gain, the prototype collapses into the generic speech centroid, causing it to trigger on every spoken word.

Our system incorporates an **Engineered Anti-Degeneracy Defense**:

| Processing Step | Mechanism & Parameters | Impact on Performance |
| :--- | :--- | :--- |
| **Energy VAD Segmentation** | 20 ms frames, energy multiplier $2.5\times$, hangover 3 frames | Eliminates leading and trailing silence from user recordings. |
| **Speech Centering** | Isolates active speech boundaries and centers speech in 1.0s window | Guarantees acoustic temporal alignment with training distribution. |
| **Gain Normalization** | Peak normalization clamped to $0.88$ maximum amplitude | Prevents ADC clipping distortion and low-gain underflows. |
| **Quality Scorecard** | Rejects recordings with peak $>0.98$ (clipping) or $<0.05$ (silent) | Automatically aborts corrupt enrollment attempts. |
| **Hypersphere Consistency Validation** | Enforces intra-shot cosine similarity $\ge 0.70$ across 3 enrollment utterances | Rejects accidental mispronunciations or background noise bursts. |
| **Silence & Noise Orthogonality** | Rejects prototype if $|\mathbf{p} \cdot \mathbf{z}_{\text{silence}}| > 0.65$ or $|\mathbf{p} \cdot \mathbf{z}_{\text{noise}}| > 0.60$ | Ensures prototype responds strictly to phonetic speech structures. |

---

## 4. Real-Time Streaming Detection & False-Positive Defense

The live inference engine evaluates audio in streaming chunks without memory allocations:

### 1. Sliding Circular Ring Buffer
* **Zero-allocation:** Pre-allocated 16,000-sample array (`float32` in Python, `int16_t` in C++).
* Pushes 50 ms audio chunks (800 samples @ 16 kHz) and returns a sliding 1.0-second window.

### 2. Dual-Threshold Hysteresis State Machine
To eliminate spurious triggers from phonetically overlapping confuser words (*"yes"*, *"no"*, *"spectrum"*, *"inspector"*), the system employs dual-threshold hysteresis:
* **Activation Threshold ($\tau_{\text{high}} = 0.88$):** The moving cosine similarity must cross 0.88 to initiate candidate detection.
* **Release Threshold ($\tau_{\text{low}} = 0.83$):** Drops only if similarity falls below 0.83 (preventing jitter at boundary).
* **Temporal Persistence ($N = 3$ consecutive windows = 150 ms):** The acoustic similarity must be sustained across 3 consecutive 50 ms evaluations. Brief acoustic spikes (50–100 ms) are rejected.
* **Refractory Cooldown ($1500\text{ ms}$):** After a confirmed wake event, the state machine enters a locked refractory state to prevent multiple triggers from reverberation.

### 3. Remote ASR Handover Dispatch
Upon activation, the edge node pulls the pre-roll wake utterance from the ring buffer and appends the incoming command audio, dispatching the complete 2.0-second payload to a local or networked open-source ASR engine (Whisper / Vosk).

---

## 5. Microcontroller Bare-Metal Deployment (ESP32-WROOM-32)

All code and headers are packaged into an Arduino-compatible sketch at:  
📁 [`outputs/voice_activator_esp32_wroom/`](file:///D:/SIH_Model/outputs/voice_activator_esp32_wroom)

### Memory Budget Verification

| Resource | Voice Activator Usage | ESP32 Hardware Capacity | Headroom Remaining |
| :--- | :--- | :--- | :--- |
| **Flash (ROM)** | **56.4 KB** (INT8 model) | 4,096 KB (4 MB Flash) | **98.6% Free** |
| **SRAM (RAM)** | **95.3 KB** (64KB Arena + 31KB Buffer) | ~320 KB (Free internal heap) | **~224 KB Free (70%)** |
| **Keyword Storage** | **128 bytes** (32 floats in NVS) | 20 KB (Default NVS partition) | **99.3% Free** |
| **Inference Time** | **35–45 ms** per 50 ms audio step | 240 MHz Dual-Core Xtensa LX6 | **Real-time capable (RTF < 1.0)** |

### Hardware Pinout (I2S MEMS Mic to ESP32)
* **VDD $\rightarrow$ 3.3V**, **GND $\rightarrow$ GND**
* **SD (Serial Data) $\rightarrow$ GPIO 32**
* **SCK (Bit Clock) $\rightarrow$ GPIO 14**
* **WS (Word Select) $\rightarrow$ GPIO 15**
* **L/R (Left/Right) $\rightarrow$ GND**
* **Wake Trigger Output $\rightarrow$ GPIO 4**

### On-Device Keyword Enrollment (No PC Required)
1. User holds the onboard **BOOT button (GPIO 0) for 2 seconds**.
2. The onboard LED illuminates solid blue, prompting the user to speak their chosen keyword 3 times.
3. The ESP32 computes the 32-D centroid vector locally, validates unit norm, and writes it directly into **Flash NVS memory**.
4. The board resumes live listening immediately. The new keyword persists across reboots and power cycles.

---

## 6. Verification, Testing & Empirical Results

### Automated Test Suite
* Ran comprehensive pytest test suite covering:
  - Data ingestion & augmentation (`tests/test_data.py`)
  - MFCC feature extraction numerical fidelity (`tests/test_features.py`)
  - Universal hypersphere embedding properties (`tests/test_embedding.py`)
  - INT8 quantization fidelity & compression ratios (`tests/test_quantize.py`)
  - Circular ring buffer & state machine hysteresis (`tests/test_streaming.py`)
  - Unseen few-shot keyword enrollment (`tests/test_unseen.py`)
  - Robustness to ambient noise & SNR degradation (`tests/test_robustness.py`)
  - End-to-end demo & ASR handover dispatch (`tests/test_demo.py`)
* **Result:** **`44 passed, 0 failures in 12.35s`**

### Live Laptop Microphone Empirical Test (Keyword: "JUMP")
```
[VAD: ON | Vol: 86% | Sim: 0.904 [########################------] State: COOLDOWN]
>>> [ACTIVATION TRIGGERED #1] KEYWORD 'JUMP' DETECTED! <<<
Confidence: 0.9041 (Threshold: 0.88, Hysteresis: 0.83)

>>> [ACTIVATION TRIGGERED #2] KEYWORD 'JUMP' DETECTED! <<<
Confidence: 0.9387 (Threshold: 0.88, Hysteresis: 0.83)

... Total activations: 12 (0 false triggers on normal speech, confuser words rejected)
```

---

## 7. Project Directory Structure

```
D:\SIH_Model\
├── models/
│   └── tflite/
│       └── voice_activator_int8.tflite        # Frozen INT8 TFLite Universal Encoder (56 KB)
├── outputs/
│   ├── voice_activator_esp32_wroom/          # Complete ready-to-flash Arduino/PlatformIO package
│   │   ├── voice_activator_esp32_wroom.ino   # Main embedded firmware sketch
│   │   ├── keyword_prototype.h               # Active 32-D keyword prototype vector (JUMP)
│   │   ├── tflite_micro_model.h              # C byte array of INT8 model weights
│   │   ├── feature_extractor.h               # Embedded C++ MFCC extractor
│   │   ├── audio_ring_buffer.h               # Zero-allocation circular buffer
│   │   ├── activator_state_machine.h         # Dual-threshold hysteresis state machine
│   │   ├── platformio.ini                    # PlatformIO build configuration
│   │   └── README.md                         # Hardware wiring & flashing guide
├── scripts/
│   ├── enroll_keyword.py                     # Few-shot enrollment with VAD centering & quality guards
│   └── live_mic_activator.py                 # Real-time microphone testing tool with dynamic audio meter
├── src/
│   ├── data/                                 # Audio loaders, augmentation, & dataset pipelines
│   ├── features/                             # 13-band MFCC extractor implementation
│   ├── models/                               # Universal CNN encoder architecture & contrastive loss
│   ├── quantization/                         # Post-training INT8 quantization engine
│   └── streaming/                            # Ring buffer, VAD, state machine, & ASR handover pipeline
├── tests/                                    # 44 automated unit & integration tests
├── run_live_mic.bat                          # 1-Click Windows Batch launcher
└── run_live_mic.ps1                          # 1-Click PowerShell launcher
```

---

## 8. Summary of Achievements

1. **Sub-100 KB Edge Footprint:** Full neural voice activator fits in 56.4 KB Flash and 95.3 KB SRAM, leaving >70% of ESP32 memory free for user applications.
2. **True Any-Keyword Spotting:** Users can enroll any word in under 5 seconds with zero model retraining or cloud connectivity.
3. **Robust False-Positive Defense:** Temporal persistence ($N=3$) and dual-threshold hysteresis ($0.88 / 0.83$) eliminate false alarms from ambient conversations and phonetically close confusers.
4. **Standalone Embedded Capability:** Complete on-device keyword switching via onboard BOOT button and non-volatile flash storage.
