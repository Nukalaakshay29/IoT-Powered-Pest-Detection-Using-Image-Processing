/*
  ============================================================
  ESP32-CAM Plant Pest Detection Device
  Hardware : ESP32-CAM (OV3660)
             DHT22 humidity sensor
             FC-37 leaf wetness sensor (analog — direct connect)
             Red LED, Green LED
             5V Battery

  Flow:
    1. Power on  → connects WiFi → starts live stream
    2. Open dashboard on laptop browser
    3. Live feed shows on dashboard (aim camera at leaf)
    4. Click "Capture & Analyze" button on dashboard
    5. Stream stops → photo taken → sensors read
    6. Data sent to laptop Flask server as JSON
    7. Server runs CNN model → sends result back
    8. RED LED  = Disease detected
       GREEN LED = Healthy
    9. Stream resumes → ready for next scan
  ============================================================
*/

#include "WiFi.h"
#include "esp_camera.h"
#include "HTTPClient.h"
#include "ArduinoJson.h"
#include "DHT.h"
#include "esp_http_server.h"
#include "base64.h"

// ─────────────────────────────────────────────────────────────
// SECTION 1 — CONFIG  (Fill these 3 before uploading)
// ─────────────────────────────────────────────────────────────

const char* ssid      = "iQOO Neo 7pro";                   // PLACEHOLDER: your WiFi name
const char* password  = "Akshay2791";               // PLACEHOLDER: your WiFi password
const char* serverURL = "http://10.22.71.133:5000/predict"; // Flask server IP

// ─────────────────────────────────────────────────────────────
// SECTION 2 — PIN DEFINITIONS
//
// FC-37 Wiring (direct — no extra module needed):
//   FC-37 VCC  → ESP32-CAM 3.3V
//   FC-37 GND  → ESP32-CAM GND
//   FC-37 AO   → ESP32-CAM GPIO 34   (analog output)
//   FC-37 DO   → Not connected
//
// DHT22 Wiring:
//   DHT22 PIN 1 (VCC)  → ESP32-CAM 3.3V
//   DHT22 PIN 2 (DATA) → ESP32-CAM GPIO 13
//                        + 10kΩ resistor between PIN2 and 3.3V
//   DHT22 PIN 3        → Not connected
//   DHT22 PIN 4 (GND)  → ESP32-CAM GND
//
// LED Wiring:
//   GPIO 4  → 220Ω resistor → RED LED   → GND
//   GPIO 12 → 220Ω resistor → GREEN LED → GND
// ─────────────────────────────────────────────────────────────

#define DHTPIN    13    // DHT22 DATA pin  → GPIO 13
#define DHTTYPE   DHT22 // DHT22 sensor
#define FC37_PIN  34    // FC-37 analog out → GPIO 34
#define RED_LED   4     // Red LED  → GPIO 4
#define GREEN_LED 12    // Green LED → GPIO 12

// ─────────────────────────────────────────────────────────────
// SECTION 3 — GLOBALS
// ─────────────────────────────────────────────────────────────

DHT dht(DHTPIN, DHTTYPE);

httpd_handle_t stream_server = NULL;  // port 80 → /stream
httpd_handle_t ctrl_server   = NULL;  // port 81 → /trigger

volatile bool captureMode = false;    // false = streaming | true = capture + send

// ─────────────────────────────────────────────────────────────
// SECTION 4 — LIVE MJPEG STREAM HANDLER
// Runs on port 80 → /stream
// Dashboard shows this as live feed
// Stops automatically when captureMode = true
// ─────────────────────────────────────────────────────────────

#define STREAM_CONTENT_TYPE "multipart/x-mixed-replace;boundary=frame"
#define STREAM_BOUNDARY     "\r\n--frame\r\n"
#define STREAM_PART         "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n"

esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    char part[64];

    httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    Serial.println("📹 Stream started");

    while (!captureMode) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("⚠️ Frame grab failed - retrying");
            delay(100);
            continue;
        }

        httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));

        size_t hlen = snprintf(part, sizeof(part), STREAM_PART, fb->len);
        httpd_resp_send_chunk(req, part, hlen);

        httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);

        esp_camera_fb_return(fb);
        fb = NULL;

        delay(80);  // ~12 fps
    }

    httpd_resp_send_chunk(req, NULL, 0);
    Serial.println("📹 Stream paused — capture in progress");
    return ESP_OK;
}

// ─────────────────────────────────────────────────────────────
// SECTION 5 — TRIGGER HANDLER
// Called when user clicks Capture button on dashboard
// FIX 1: HTTP_OK replaced with 200 — HTTP_OK not available
//         in esp_http_server.h, use raw integer 200 instead
// ─────────────────────────────────────────────────────────────

esp_err_t trigger_handler(httpd_req_t *req) {
    Serial.println("📸 Capture triggered from dashboard!");
    captureMode = true;
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_status(req, "200 OK");                  // FIX 1
    httpd_resp_send(req, "Capture triggered", HTTPD_RESP_USE_STRLEN); // FIX 1
    return ESP_OK;
}

// ─────────────────────────────────────────────────────────────
// SECTION 6 — FC-37 LEAF WETNESS READING
//
// FC-37 behavior:
//   Dry leaf  → high resistance → HIGH ADC (~4095) → 0% wet
//   Wet leaf  → low resistance  → LOW ADC  (~0)    → 100% wet
// ─────────────────────────────────────────────────────────────

float readLeafWetness() {
    float total = 0;
    for (int i = 0; i < 10; i++) {
        total += analogRead(FC37_PIN);
        delay(10);
    }
    float avg        = total / 10.0;
    float percentage = map(avg, 4095, 0, 0, 100);
    percentage       = constrain(percentage, 0, 100);
    Serial.printf("🍃 FC-37 raw ADC: %.0f → Wetness: %.1f%%\n", avg, percentage);
    return percentage;
}

// ─────────────────────────────────────────────────────────────
// SECTION 7 — START HTTP SERVERS
// port 80 → /stream   (MJPEG live feed)
// port 81 → /trigger  (capture command from dashboard)
// ─────────────────────────────────────────────────────────────

void startServers() {
    // Stream server on port 80
    httpd_config_t stream_cfg = HTTPD_DEFAULT_CONFIG();
    stream_cfg.server_port    = 80;
    stream_cfg.ctrl_port      = 32768;

    httpd_uri_t stream_uri = {
        .uri      = "/stream",
        .method   = HTTP_GET,
        .handler  = stream_handler,
        .user_ctx = NULL
    };

    if (httpd_start(&stream_server, &stream_cfg) == ESP_OK) {
        httpd_register_uri_handler(stream_server, &stream_uri);
        Serial.println("✅ Stream server ready  → port 80  /stream");
    } else {
        Serial.println("❌ Stream server failed to start");
    }

    // Control server on port 81
    httpd_config_t ctrl_cfg = HTTPD_DEFAULT_CONFIG();
    ctrl_cfg.server_port    = 81;
    ctrl_cfg.ctrl_port      = 32769;

    httpd_uri_t trigger_uri = {
        .uri      = "/trigger",
        .method   = HTTP_GET,
        .handler  = trigger_handler,
        .user_ctx = NULL
    };

    if (httpd_start(&ctrl_server, &ctrl_cfg) == ESP_OK) {
        httpd_register_uri_handler(ctrl_server, &trigger_uri);
        Serial.println("✅ Control server ready → port 81  /trigger");
    } else {
        Serial.println("❌ Control server failed to start");
    }
}

// ─────────────────────────────────────────────────────────────
// SECTION 8 — SETUP
// Runs once on power on
// ─────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    Serial.println("\n========================================");
    Serial.println("  ESP32-CAM Pest Detection Device");
    Serial.println("  Sensors: DHT22 + FC-37 Leaf Wetness");
    Serial.println("========================================");

    // LED pins
    pinMode(RED_LED,   OUTPUT);
    pinMode(GREEN_LED, OUTPUT);
    digitalWrite(RED_LED,   LOW);
    digitalWrite(GREEN_LED, LOW);

    // Start DHT22
    dht.begin();
    Serial.println("✅ DHT22 started");
    Serial.println("✅ FC-37 on GPIO 34");

    // Connect WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println();
    Serial.println("✅ WiFi connected!");
    Serial.println("   ESP32 IP : " + WiFi.localIP().toString());

    // Camera config — ESP32-CAM AI-Thinker with OV3660
    camera_config_t config;
    config.ledc_channel  = LEDC_CHANNEL_0;
    config.ledc_timer    = LEDC_TIMER_0;
    config.pin_d0        = 5;
    config.pin_d1        = 18;
    config.pin_d2        = 19;
    config.pin_d3        = 21;
    config.pin_d4        = 36;
    config.pin_d5        = 39;
    config.pin_d6        = 35;
    config.pin_d7        = 14;
    config.pin_xclk      = 0;
    config.pin_pclk      = 22;
    config.pin_vsync     = 25;
    config.pin_href      = 23;
    config.pin_sscb_sda  = 26;
    config.pin_sscb_scl  = 27;
    config.pin_pwdn      = 32;
    config.pin_reset     = -1;
    config.xclk_freq_hz  = 20000000;
    config.pixel_format  = PIXFORMAT_JPEG;
    config.frame_size    = FRAMESIZE_VGA;
    config.jpeg_quality  = 12;
    config.fb_count      = 2;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("❌ Camera init failed: 0x%x\n", err);
        for (int i = 0; i < 10; i++) {
            digitalWrite(RED_LED, HIGH); delay(200);
            digitalWrite(RED_LED, LOW);  delay(200);
        }
        return;
    }
    Serial.println("✅ Camera started (OV3660, VGA 640x480)");

    startServers();

    Serial.println();
    Serial.println("========================================");
    Serial.println("  DEVICE READY");
    Serial.println("========================================");
    Serial.println("  📹 Live Stream : http://" + WiFi.localIP().toString() + "/stream");
    Serial.println("  🔗 Trigger     : http://" + WiFi.localIP().toString() + ":81/trigger");
    Serial.println("  📊 Dashboard   : http://YOUR_PC_IP:5000/dashboard"); // PLACEHOLDER
    Serial.println("========================================\n");

    // Blink green twice = device ready
    for (int i = 0; i < 2; i++) {
        digitalWrite(GREEN_LED, HIGH); delay(300);
        digitalWrite(GREEN_LED, LOW);  delay(300);
    }
}

// ─────────────────────────────────────────────────────────────
// SECTION 9 — MAIN LOOP
// ─────────────────────────────────────────────────────────────

void loop() {

    if (!captureMode) {
        delay(100);
        return;
    }

    Serial.println("\n--- Capture started ---");
    delay(300);

    // Step 1: Read DHT22
    float humidity    = dht.readHumidity();
    float temperature = dht.readTemperature();

    if (isnan(humidity) || isnan(temperature)) {
        Serial.println("⚠️ DHT22 read failed — using safe defaults");
        humidity    = 50.0;
        temperature = 25.0;
    }
    Serial.printf("🌡️  Temperature   : %.1f C\n",  temperature);
    Serial.printf("💧  Humidity      : %.1f %%\n",  humidity);

    // Step 2: Read FC-37
    float leafMoisture = readLeafWetness();

    // Step 3: Capture photo
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("❌ Camera capture failed");
        digitalWrite(RED_LED, HIGH); delay(500); digitalWrite(RED_LED, LOW);
        captureMode = false;
        return;
    }
    Serial.printf("📷 Photo size : %u bytes\n", fb->len);

    // Step 4: Encode photo as base64
    // FIX 2: base64::encode() on ESP32 core 3.x returns a String
    //        Old 3-argument version no longer exists
    String imageBase64 = base64::encode(fb->buf, fb->len); // FIX 2

    Serial.printf("📢 Base64 size: %u characters\n", imageBase64.length());

    // Step 5: Build JSON
    // FIX 2 continued: use imageBase64.c_str() to put String into JSON
    StaticJsonDocument<35000> doc;
    doc["image"]         = imageBase64.c_str(); // FIX 2
    doc["humidity"]      = humidity;
    doc["leaf_moisture"] = leafMoisture;

    String requestBody;
    serializeJson(doc, requestBody);
    Serial.println("📦 Sending to Flask server...");

    // Step 6: POST to Flask server
    HTTPClient http;
    http.begin(serverURL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(15000);

    int httpCode = http.POST(requestBody);

    // Step 7: Read response → control LEDs
    if (httpCode > 0) {
        String response = http.getString();
        Serial.println("📡 Server response: " + response);

        if (response.indexOf("\"led\":\"RED\"") >= 0) {
            digitalWrite(RED_LED,   HIGH);
            digitalWrite(GREEN_LED, LOW);
            Serial.println("🚨 DISEASE DETECTED — RED LED ON");
        } else {
            digitalWrite(RED_LED,   LOW);
            digitalWrite(GREEN_LED, HIGH);
            Serial.println("✅ HEALTHY — GREEN LED ON");
        }
        Serial.printf("HTTP status: %d\n", httpCode);

    } else {
        Serial.printf("❌ Server not reachable — HTTP error: %d\n", httpCode);
        Serial.println("   Check: Flask server running on laptop?");
        Serial.println("   Check: YOUR_PC_IP correct in serverURL?");
        for (int i = 0; i < 5; i++) {
            digitalWrite(RED_LED, HIGH); delay(150);
            digitalWrite(RED_LED, LOW);  delay(150);
        }
    }

    // Step 8: Clean up
    http.end();
    esp_camera_fb_return(fb);

    // Step 9: Reset
    captureMode = false;
    Serial.println("📹 Ready — reload dashboard to start stream again");
    Serial.println("--- Capture complete ---\n");
}
