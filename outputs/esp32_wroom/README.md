# ESP32-WROOM Low-Latency Voice Activator Deployment Guide
**SIH Problem Statement 26172 (ISRO)**

Complete guide and source files to run the trained INT8 Voice Activator on an **ESP32-WROOM-32** (ESP32 DevKit v1 / NodeMCU-32S) using standard **Arduino IDE** or **PlatformIO**.

---

## 1. Hardware Specifications & Pinout

### Required Hardware
1. **ESP32-WROOM-32** Dev Module (Xtensa Dual-Core LX6 @ 240 MHz, 520 KB SRAM, 4MB Flash)
2. **INMP441** I2S Digital MEMS Microphone (or ICS-43434 / SPH0645)
3. Micro-USB cable & breadboard jumper wires

### Wiring Diagram: INMP441 to ESP32-WROOM-32

| INMP441 Microphone Pin | ESP32-WROOM Pin | Description |
| :--- | :--- | :--- |
| **VDD** | **3V3** | 3.3V Power |
| **GND** | **GND** | Ground |
| **SD** (Serial Data) | **GPIO 32** | I2S Data In (`I2S_SD_PIN`) |
| **SCK** (Serial Clock) | **GPIO 14** | I2S Bit Clock (`I2S_SCK_PIN`) |
| **WS** (Word Select) | **GPIO 15** | I2S Word Select / LR Clock (`I2S_WS_PIN`) |
| **L/R** (Channel) | **GND** | Connect to GND for Left Channel Mono |

### Indicators & Trigger Pins
- **GPIO 2**: Onboard Blue LED (Lights up for 1.5s when keyword is detected).
- **GPIO 4**: High-level wake pulse (Connect to host MCU or remote system to trigger ASR handover).

---

## 2. Memory Footprint on ESP32-WROOM

| Memory Type | Required by Voice Activator | Total Available on ESP32-WROOM | Headroom Remaining |
| :--- | :--- | :--- | :--- |
| **Flash** | **56.4 KB** (TFLite INT8 model) | 4,096 KB (4 MB) | **98.6% free** (Plenty of room for WiFi/BLE) |
| **Internal SRAM** | **63.0 KB** (Tensor Arena) + **31.3 KB** (Ring Buffer) = **94.3 KB** | ~320 KB (Free internal heap) | **~225 KB free** |
| **Inference Time** | **~35 - 45 ms** | 50 ms budget | **Real-time 1.0x capable** |

---

## 3. Deployment via Arduino IDE (Simplest Method)

### Step 1: Install Arduino Board & Library
1. Open **Arduino IDE** (v2.x recommended).
2. Go to **File -> Preferences**, and paste into *Additional Board Manager URLs*:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Go to **Tools -> Board -> Boards Manager**, search for `esp32` by **Espressif Systems**, and install it.
4. Go to **Sketch -> Include Library -> Manage Libraries**, search for `TensorFlowLite_ESP32` by **Tanaka Masayuki**, and click **Install**.

### Step 2: Open the Sketch Folder
Create a folder named `voice_activator_esp32_wroom` and place these files inside it:
- `voice_activator_esp32_wroom.ino`
- `tflite_micro_model.h`
- `keyword_prototype.h`
- `feature_extractor.h`
- `audio_ring_buffer.h`
- `activator_state_machine.h`

### Step 3: Configure Board Settings
In the Arduino IDE **Tools** menu:
- **Board**: `ESP32 Dev Module`
- **CPU Frequency**: `240MHz (WiFi/BT)`
- **Flash Frequency**: `80MHz`
- **Flash Mode**: `QIO`
- **Partition Scheme**: `Default 4MB with spiffs (1.2MB APP / 1.5MB SPIFFS)`
- **Port**: Select your ESP32 COM port (e.g. `COM3` / `COM4`)

### Step 4: Compile & Upload
1. Click **Upload** (Arrow icon).
2. Open **Tools -> Serial Monitor** at **115200 baud**.
3. Speak your enrolled keyword (e.g., **"ASWIN"**) into the INMP441 microphone.
4. The onboard blue LED will illuminate and the serial monitor will output:
   ```
   >>> ========================================================
   >>> [ACTIVATION EVENT] Target Keyword 'ASWIN' Detected!
   >>> Cosine Similarity: 0.9142 | Inference Latency: 38 ms
   >>> Triggering Wake GPIO & Dispatching ASR Handover...
   >>> ========================================================
   ```

---

## 4. Deployment via PlatformIO (VS Code)

If using VS Code with PlatformIO:
1. Open this folder in PlatformIO.
2. The included `platformio.ini` automatically pulls `TensorFlowLite_ESP32`.
3. Run:
   ```bash
   pio run --target upload && pio device monitor
   ```

---

## 5. Changing the Custom Keyword
To switch the enrolled keyword to a different word:
1. Run on your PC:
   ```bash
   python scripts/enroll_keyword.py --keyword <NEW_WORD> --mic
   ```
2. Replace `keyword_prototype.h` in the ESP32 project folder with the newly generated `src/deployment/esp32/keyword_prototype.h`.
3. Re-flash the ESP32. **No model retraining or weights change needed!**
