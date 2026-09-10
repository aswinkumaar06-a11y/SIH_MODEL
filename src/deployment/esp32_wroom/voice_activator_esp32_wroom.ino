/*
 * ESP32-WROOM Firmware Entry Point: Low Latency Voice Activator
 * Target: ESP32-WROOM-32 / ESP32 DevKit v1 (Xtensa LX6 dual-core)
 * SIH Problem Statement 26172
 *
 * Microcontroller Specifications:
 * - MCU: ESP32-WROOM-32 (240 MHz, 520 KB SRAM, 4MB Flash)
 * - Microphone: INMP441 (I2S Digital MEMS Microphone)
 * - Model: INT8 Quantized Metric CNN (56.4 KB Flash, 63.0 KB SRAM Arena)
 * - Real-time processing: 50 ms streaming chunks @ 16 kHz
 */

#include <Arduino.h>
#include "driver/i2s.h"

// TFLite Micro Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Local Component Headers
#include "tflite_micro_model.h"
#include "keyword_prototype.h"
#include "audio_ring_buffer.h"
#include "feature_extractor.h"
#include "activator_state_machine.h"

// ==============================================================================
// Hardware Pin Definitions (Configured for ESP32-WROOM / ESP32-DevKit v1)
// ==============================================================================
#define I2S_MIC_PORT        I2S_NUM_0
#define I2S_SCK_PIN         14    // BCLK (Serial Clock) -> INMP441 SCK
#define I2S_WS_PIN          15    // WS / LRCL (Word Select) -> INMP441 WS
#define I2S_SD_PIN          32    // SD / DOUT (Serial Data) -> INMP441 SD
#define WAKE_LED_PIN         2    // On-board Blue LED (GPIO 2 on DevKit v1)
#define WAKE_TRIGGER_PIN     4    // External GPIO pulse to wake host MCU / remote ASR

// ==============================================================================
// Audio & ML Configuration
// ==============================================================================
constexpr size_t SAMPLE_RATE = 16000;
constexpr size_t CHUNK_SIZE = 800;              // 50 ms @ 16 kHz (800 samples)
constexpr float VAD_ENERGY_THRESHOLD = 0.012f;  // RMS Gate to skip CNN on silence
constexpr size_t TENSOR_ARENA_SIZE = 64 * 1024; // 64 KB Static Arena in SRAM

// Static Memory Allocations in SRAM (Zero dynamic heap allocations during loop)
static alignas(16) uint8_t g_tensor_arena[TENSOR_ARENA_SIZE];
static AudioRingBuffer g_ring_buffer;
static EdgeMFCCExtractor g_mfcc_extractor;
static ActivatorStateMachine g_state_machine(0.82f, 0.78f, 3, 4, 1500);

// Global TFLite Micro Pointers
static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input_tensor = nullptr;
static TfLiteTensor* g_output_tensor = nullptr;

// Activation State Tracking
static bool g_led_active = false;
static uint32_t g_activation_time_ms = 0;

// ==============================================================================
// Helper Functions
// ==============================================================================

// Compute RMS Energy for Voice Activity Detection (VAD) gating
static float compute_rms_energy(const float* samples, size_t count) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum_sq += samples[i] * samples[i];
    }
    return std::sqrt(sum_sq / static_cast<float>(count));
}

// Compute Cosine Similarity between 32-D Embedding and Enrolled Prototype
static float compute_cosine_similarity(const float* embedding, const float* prototype, int dim) {
    float dot = 0.0f;
    float norm_e = 0.0f;
    float norm_p = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += embedding[i] * prototype[i];
        norm_e += embedding[i] * embedding[i];
        norm_p += prototype[i] * prototype[i];
    }
    float denom = std::sqrt(norm_e) * std::sqrt(norm_p);
    return (denom > 1e-9f) ? (dot / denom) : 0.0f;
}

// ==============================================================================
// I2S Microphone Initialization
// ==============================================================================
void setup_i2s_microphone() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = CHUNK_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD_PIN
    };

    esp_err_t err = i2s_driver_install(I2S_MIC_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[ERROR] Failed to install I2S driver: %d\n", err);
        return;
    }

    err = i2s_set_pin(I2S_MIC_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[ERROR] Failed to set I2S pins: %d\n", err);
        return;
    }

    Serial.println("[INFO] I2S INMP441 Microphone Initialized (16 kHz, 16-bit Mono)");
}

// ==============================================================================
// TensorFlow Lite Micro Initialization
// ==============================================================================
void setup_tflite_micro() {
    Serial.println("================================================================");
    Serial.println("SIH 26172: ESP32-WROOM VOICE ACTIVATOR INITIALIZATION");
    Serial.printf("Target Keyword: %s\n", ENROLLED_KEYWORD_NAME);
    Serial.printf("Model Size:     %u bytes (%.2f KB)\n", g_voice_activator_model_data_len, g_voice_activator_model_data_len / 1024.0f);
    Serial.printf("Tensor Arena:   %u bytes (%.2f KB)\n", TENSOR_ARENA_SIZE, TENSOR_ARENA_SIZE / 1024.0f);
    Serial.println("================================================================");

    // 1. Map model flatbuffer
    g_model = tflite::GetModel(g_voice_activator_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[ERROR] Model schema mismatch! Expected %d, got %ld\n", TFLITE_SCHEMA_VERSION, g_model->version());
        return;
    }

    // 2. Register Required Quantized Operations (Compatible with standard Xtensa LX6)
    static tflite::MicroMutableOpResolver<7> op_resolver;
    op_resolver.AddConv2D();
    op_resolver.AddRelu();
    op_resolver.AddAveragePool2D();
    op_resolver.AddFullyConnected();
    op_resolver.AddReshape();
    op_resolver.AddQuantize();
    op_resolver.AddDequantize();

    // 3. Instantiate MicroInterpreter in static SRAM arena
    static tflite::MicroInterpreter static_interpreter(
        g_model,
        op_resolver,
        g_tensor_arena,
        TENSOR_ARENA_SIZE
    );
    g_interpreter = &static_interpreter;

    // 4. Allocate Tensors
    TfLiteStatus allocate_status = g_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("[ERROR] AllocateTensors() failed! Increase TENSOR_ARENA_SIZE");
        return;
    }

    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);

    Serial.printf("[INFO] TFLite Micro ready on ESP32-WROOM. Listening for keyword '%s'...\n", ENROLLED_KEYWORD_NAME);
}

// ==============================================================================
// Audio Chunk Processing Loop (50 ms Step)
// ==============================================================================
void process_audio_chunk(const int16_t* pcm_chunk, size_t chunk_len, uint32_t current_time_ms) {
    // 1. Push 50 ms audio chunk into 1.0s Ring Buffer
    g_ring_buffer.push(pcm_chunk, chunk_len);

    if (!g_ring_buffer.is_full()) {
        return; // Fill initial 1.0s buffer window
    }

    // 2. Read full 1.0s audio window (16,000 samples)
    static float s_audio_window[16000];
    g_ring_buffer.read_window(s_audio_window, 16000);

    // 3. Short-Time RMS Energy VAD Gating (Prevents unnecessary CNN inference during silence)
    float rms = compute_rms_energy(s_audio_window + (16000 - chunk_len), chunk_len);
    if (rms < VAD_ENERGY_THRESHOLD) {
        return; // Ambient silence / low noise: skip inference to keep idle CPU < 10%
    }

    uint32_t t_start = millis();

    // 4. Extract MFCC Feature Matrix (98 frames x 13 coefficients)
    static float s_mfcc_matrix[98 * 13];
    g_mfcc_extractor.extract_features(s_audio_window, s_mfcc_matrix);

    // 5. Populate Input Tensor (handles INT8 quantization if required)
    if (g_input_tensor->type == kTfLiteInt8) {
        float scale = g_input_tensor->params.scale;
        int zero_point = g_input_tensor->params.zero_point;
        int8_t* in_data = g_input_tensor->data.int8;
        for (int i = 0; i < 98 * 13; ++i) {
            int q = static_cast<int>(std::round(s_mfcc_matrix[i] / scale)) + zero_point;
            in_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
        }
    } else {
        memcpy(g_input_tensor->data.f, s_mfcc_matrix, sizeof(s_mfcc_matrix));
    }

    // 6. Invoke TFLite Micro Neural Network
    if (g_interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[WARN] TFLite invoke failed");
        return;
    }

    // 7. Read Output Embedding (32-D)
    float current_embedding[KEYWORD_PROTOTYPE_DIM];
    if (g_output_tensor->type == kTfLiteInt8) {
        float scale = g_output_tensor->params.scale;
        int zero_point = g_output_tensor->params.zero_point;
        int8_t* out_data = g_output_tensor->data.int8;
        for (int i = 0; i < KEYWORD_PROTOTYPE_DIM; ++i) {
            current_embedding[i] = (out_data[i] - zero_point) * scale;
        }
    } else {
        memcpy(current_embedding, g_output_tensor->data.f, sizeof(current_embedding));
    }

    // 8. Compute Cosine Similarity against Enrolled Keyword Prototype
    float raw_similarity = compute_cosine_similarity(
        current_embedding,
        KEYWORD_PROTOTYPE,
        KEYWORD_PROTOTYPE_DIM
    );

    uint32_t latency_ms = millis() - t_start;

    // 9. Update Hysteresis State Machine
    bool activated = g_state_machine.update(raw_similarity, current_time_ms);

    if (activated) {
        Serial.printf("\n>>> ========================================================\n");
        Serial.printf(">>> [ACTIVATION EVENT] Target Keyword '%s' Detected!\n", ENROLLED_KEYWORD_NAME);
        Serial.printf(">>> Cosine Similarity: %.4f | Inference Latency: %u ms\n", raw_similarity, latency_ms);
        Serial.printf(">>> Triggering Wake GPIO & Disptaching ASR Handover...\n");
        Serial.printf(">>> ========================================================\n\n");

        // Turn on Onboard LED & Trigger Pin
        digitalWrite(WAKE_LED_PIN, HIGH);
        digitalWrite(WAKE_TRIGGER_PIN, HIGH);
        g_led_active = true;
        g_activation_time_ms = current_time_ms;
    }
}

// ==============================================================================
// Arduino Main Functions
// ==============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);

    // Setup GPIO outputs
    pinMode(WAKE_LED_PIN, OUTPUT);
    pinMode(WAKE_TRIGGER_PIN, OUTPUT);
    digitalWrite(WAKE_LED_PIN, LOW);
    digitalWrite(WAKE_TRIGGER_PIN, LOW);

    // Setup I2S Mic & Neural Network
    setup_i2s_microphone();
    setup_tflite_micro();
}

void loop() {
    // 1. Read 50 ms audio chunk (800 samples) from I2S DMA
    int16_t pcm_chunk[CHUNK_SIZE];
    size_t bytes_read = 0;

    esp_err_t err = i2s_read(I2S_MIC_PORT, pcm_chunk, sizeof(pcm_chunk), &bytes_read, portMAX_DELAY);
    if (err == ESP_OK && bytes_read == sizeof(pcm_chunk)) {
        uint32_t current_time_ms = millis();
        process_audio_chunk(pcm_chunk, CHUNK_SIZE, current_time_ms);
    }

    // 2. Turn off LED and Wake Pin after 1.5s Cooldown
    if (g_led_active && (millis() - g_activation_time_ms >= 1500)) {
        digitalWrite(WAKE_LED_PIN, LOW);
        digitalWrite(WAKE_TRIGGER_PIN, LOW);
        g_led_active = false;
    }
}
