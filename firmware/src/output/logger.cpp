// logger.cpp — Serial output in dashboard-compatible CSV format
#include "logger.h"
#include <Arduino.h>

void logger_init() {
    Serial.begin(115200);
    // CSV header
    Serial.println("Time,HR,SpO2,Temp,Anomaly");
}

void logger_print(float time_s, float hr, float spo2, float temp, int anomaly) {
    Serial.print(time_s, 1);
    Serial.print(",");
    Serial.print(hr, 1);
    Serial.print(",");
    Serial.print(spo2, 1);
    Serial.print(",");
    Serial.print(temp, 2);
    Serial.print(",");
    Serial.println(anomaly);
}
