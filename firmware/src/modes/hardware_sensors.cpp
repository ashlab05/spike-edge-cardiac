// hardware_sensors.cpp
// Placeholder for real sensor integration (AD8232 + MAX30100 + DS18B20).
// Replace simulated_signals.cpp with this file after soldering.

#include "../../config.h"
#include "hardware_sensors.h"

#ifdef MODE_HARDWARE

#include <OneWire.h>
#include <DallasTemperature.h>
// #include <MAX30100_PulseOximeter.h>  // Add MAX30100 library when ready

OneWire oneWire(PIN_TEMP_SENSOR);
DallasTemperature tempSensor(&oneWire);
// PulseOximeter pox;  // Uncomment when library available

void hardware_sensors_init() {
    tempSensor.begin();
    // Wire.begin(PIN_PPG_SDA, PIN_PPG_SCL);
    // pox.begin();
}

void hardware_sensors_read(float *hr, float *spo2, float *temp) {
    // Real DS18B20 temperature
    tempSensor.requestTemperatures();
    *temp = tempSensor.getTempCByIndex(0);

    // TODO: Replace with real MAX30100 readings
    *hr   = 0.0f;
    *spo2 = 0.0f;
}

#endif  // MODE_HARDWARE
