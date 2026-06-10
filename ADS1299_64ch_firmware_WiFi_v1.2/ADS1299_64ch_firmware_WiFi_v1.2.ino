#include <SPI.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"

const char* ssid     = "CyberVisor_64ch_v1.2";
const char* password = "12345678";
const char* udpAddress = "192.168.4.255";
const int   udpPort    = 2323;

IPAddress clientIP;          
bool clientConnected = false;


volatile bool is_streaming = false;

const uint8_t CHIP_NUM = 4; 
const uint8_t TOTAL_CHANNELS = CHIP_NUM * 8;
const uint8_t CS[CHIP_NUM] = {20, 11, 3, 2};

#define MOSI  5
#define RST   4
#define SCLK  6
#define MISO  7
#define DRDY  18
#define STR   19

#define SDATAC  0x11
#define RDATAC  0x10
#define RREG    0x20
#define WREG    0x40

#define SAMPLES_PER_PACKET 10 

struct DataPacket 
{
  struct Bundle
  {
    uint32_t timestamp;
    uint8_t samples[CHIP_NUM * 24];
  }
  bundles[SAMPLES_PER_PACKET];
  
};


WiFiUDP udp;

QueueHandle_t dataQueue;

volatile bool data_ready = false;
volatile uint8_t spi_buffer[CHIP_NUM][27]; 
uint8_t bundle_idx = 0;

DataPacket outboundPacket;



void Cmd(uint8_t cmd) 
{
  SPI.transfer(cmd);
  delayMicroseconds(2);
}

void wReg(uint8_t reg, uint8_t value) 
{
  SPI.transfer(WREG | reg);
  SPI.transfer(0x00);
  SPI.transfer(value);
  delayMicroseconds(2);
}

uint8_t rReg(uint8_t reg) 
{
  uint8_t val;
  delayMicroseconds(5);
  SPI.transfer(RREG | reg);
  SPI.transfer(0x00);
  val = SPI.transfer(0x00);
  delayMicroseconds(5);
  return val;
}

void Reset() 
{
  digitalWrite(RST, LOW);
  delay(200);
  digitalWrite(RST, HIGH);
  delay(500); 
}


void chip_config(uint8_t chip_idx, bool is_master) 
{
  digitalWrite(CS[chip_idx], LOW);
  
  Cmd(SDATAC);
  delay(20);

  uint8_t id = rReg(0x00);
  Serial.printf("Chip %d (%s) ID: 0x%X\n", chip_idx + 1, is_master ? "Master" : "Slave", id);

  wReg(0x01, is_master ? 0xF4 : 0xD4);
  wReg(0x02, 0xC0);       
  wReg(0x03, is_master ? 0xEC : 0xE8);  
  wReg(0x17, 0x00);
  wReg(0x04, 0x00);
  wReg(0x0D, is_master ? 0xFF : 0x00);
  wReg(0x0E, 0x00);
  wReg(0x0F, 0x00);
  wReg(0x10, 0x00);
  wReg(0x11, 0x00);
  wReg(0x15, 0x20); 

  for (uint8_t ch = 0; ch < 8; ch++)
  {
    wReg(0x05 + ch, 0x60); 
  }

  if (id == 0x3E) wReg(0x14, 0x80);
  else            wReg(0x14, 0x20);

  Cmd(RDATAC);
  delay(10);
  
  digitalWrite(CS[chip_idx], HIGH);
}

void IRAM_ATTR onDRDY()
{
  data_ready = true;
}


void wifiTask(void * parameter) 
{
  DataPacket txPacket;
  char packetBuffer[255];

  udp.begin(udpPort);
  
  while(true) 
  {
    int packetSize = udp.parsePacket();
    if (packetSize) 
    {
      int len = udp.read(packetBuffer, 255);
      if (len > 0) packetBuffer[len] = 0;

      if (strstr(packetBuffer, "STR") != NULL) 
      {
        bundle_idx = 0;
        is_streaming = true;
        clientIP = udp.remoteIP();
        clientConnected = true;
        Serial.println("Command Received: START");
      }
      else if (strstr(packetBuffer, "STOP") != NULL) 
      {
        is_streaming = false;
        clientConnected = false;
        Serial.println("Command Received: STOP");
      }
    }

    if (xQueueReceive(dataQueue, &txPacket, pdMS_TO_TICKS(100)) == pdTRUE) 
    {
      if (clientConnected)
      {
        udp.beginPacket(clientIP, udpPort);
        udp.write((uint8_t*)&txPacket, sizeof(DataPacket));
        udp.endPacket();
      }
    }
    else 
    {
      vTaskDelay(1);
    }
  }
}



void setup() 
{
  Serial.begin(921600);
  
  Serial.print("Starting AP ...");
  WiFi.mode(WIFI_AP);
  WiFi.setTxPower(WIFI_POWER_15dBm);
  esp_wifi_set_ps(WIFI_PS_NONE);

  if (!WiFi.softAP(ssid, password, 6)) 
  {
    Serial.println("AP creation failed.");
    while(1);
  }
  
  Serial.print("AP Created!");
  Serial.println(ssid);
  Serial.print("CyberVisor IP address: ");
  Serial.println(WiFi.softAPIP());

  dataQueue = xQueueCreate(250, sizeof(DataPacket));

  xTaskCreatePinnedToCore(wifiTask, "WiFiTask", 10000, NULL, 1, NULL, 0);

  SPI.begin(SCLK, MISO, MOSI);
  SPI.setFrequency(8000000); 
  SPI.setDataMode(SPI_MODE1);
  SPI.setBitOrder(MSBFIRST);

  pinMode(DRDY, INPUT);
  pinMode(RST, OUTPUT);
  pinMode(STR, OUTPUT);

  for (uint8_t i = 0; i < CHIP_NUM; i++) 
  {
    pinMode(CS[i], OUTPUT);
    digitalWrite(CS[i], HIGH);
  }

  digitalWrite(STR, LOW); 
  delay(100);
  Reset();
  delay(100);

  for (uint8_t i = 0; i < CHIP_NUM; i++) 
  {
    digitalWrite(CS[i], LOW);
    for(int j=0; j<3; j++) { Cmd(SDATAC); delay(50); }
    digitalWrite(CS[i], HIGH);
  }

  delay(100);

  for (uint8_t i = 0; i < CHIP_NUM; i++) 
  {
    chip_config(i, (i == 0));
    if (i == 0) delay(200);
  }

  delay(100);
  digitalWrite(STR, HIGH);
  delay(100); 

  attachInterrupt(digitalPinToInterrupt(DRDY), onDRDY, FALLING);
}



void loop() 
{
  if (!data_ready) return;

  uint32_t cur_time = millis();

  for (uint8_t chip = 0; chip < CHIP_NUM; chip++) 
  {
    digitalWrite(CS[chip], LOW);
    SPI.transferBytes(NULL, (uint8_t*)spi_buffer[chip], 27);
    digitalWrite(CS[chip], HIGH);
  }

  data_ready = false;
  
  if (is_streaming) 
  {
    if ((spi_buffer[0][0] & 0xF0) != 0xC0) return;

    auto& bundle = outboundPacket.bundles[bundle_idx];

    bundle.timestamp = cur_time;

    for (uint8_t chip = 0; chip < CHIP_NUM; chip++) 
    {
      memcpy(bundle.samples + chip * 24, (const uint8_t*)&spi_buffer[chip][3], 24);
    }

    bundle_idx++;
    if (bundle_idx >= SAMPLES_PER_PACKET) 
    {
      xQueueSend(dataQueue, &outboundPacket, 0);
      bundle_idx = 0;
    }
  }
}