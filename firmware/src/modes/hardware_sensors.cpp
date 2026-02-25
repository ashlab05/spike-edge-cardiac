// hardware_sensors.cpp
// Real sensor reads for MODE_HARDWARE.
// Per-sensor availability is controlled by flags in config.h:
//   HW_TEMP_ENABLED      — DS18B20 connected on GPIO 5 (1-Wire)
//   HW_MAX30100_ENABLED  — MAX30100 connected on GPIO 8/9 (I2C)
//   HW_AD8232_ENABLED    — AD8232 connected on GPIO 4 (ADC)
// Unconnected sensors fall back to simulated baseline values.

#include "../../config.h"
#include "hardware_sensors.h"
#include <Arduino.h>

#ifdef MODE_HARDWARE

#include <OneWire.h>
#include <DallasTemperature.h>

#ifdef HW_TEMP_ENABLED
static OneWire oneWire(PIN_TEMP_SENSOR);
static DallasTemperature tempSensor(&oneWire);
#endif

#ifdef HW_MAX30100_ENABLED
#include <MAX30100_PulseOximeter.h>
static PulseOximeter pox;
#endif

void hardware_sensors_init() {
#ifdef HW_TEMP_ENABLED
    tempSensor.begin();
#endif

#ifdef HW_MAX30100_ENABLED
    Wire.begin(PIN_PPG_SDA, PIN_PPG_SCL);
    pox.begin();
#endif
}

void hardware_sensors_read(float *hr, float *spo2, float *temp) {
    // ── Temperature ───────────────────────────────────────────────────────────
#ifdef HW_TEMP_ENABLED
    tempSensor.requestTemperatures();
    *temp = tempSensor.getTempCByIndex(0);
#else
    // Fallback: DS18B20 not connected
    *temp = 36.8f + ((float)random(-100, 100) / 1000.0f);
#endif

    // ── HR & SpO2 ─────────────────────────────────────────────────────────────
#ifdef HW_MAX30100_ENABLED
    pox.update();
    *hr   = pox.getHeartRate();
    *spo2 = pox.getSpO2();
#else
    // Fallback: MAX30100 not yet soldered — use simulated baseline with noise
    *hr   = HW_FALLBACK_HR   + ((float)random(-200, 200) / 100.0f);
    *spo2 = HW_FALLBACK_SPO2 + ((float)random(-50,  50)  / 100.0f);
#endif
}

#endif  // MODE_HARDWARE
