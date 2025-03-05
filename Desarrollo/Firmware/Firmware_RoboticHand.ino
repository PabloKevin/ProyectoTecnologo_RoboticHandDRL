//#include <Arduino.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include <PubSubClient.h>

#define MOTION_TIME 600
#define msg_len 100
// Definir pines para los servos
#define pulgarPin 12
#define indicePin 14
#define medioPin 26
#define anularPin 33
#define meniquePin 32
#define led_Connected LED_BUILTIN //se enciende el led mientras la mano está conectada al wifi y broker y está escuchando (connected && ESTADO=1)
#define ledMoving LED_BUILTIN //se apaga el led mientras la mano se mueve (ESTADO=0)

// Estado global compartido entre núcleos
volatile int ESTADO = 1;
int Action[5]; //grados de cierre para cada dedo
bool OpenAgain = 0; //flag indicadora de si luego de cerrar dedos en un movimiento, se deben volver a abrir o no

// Credenciales de red Wi-Fi
const char* ssid = "Ykay-2.4GHz"; // Tu red Wi-Fi
const char* password = "Kevin210608"; // Contraseña de tu red Wi-Fi

// Dirección del broker MQTT
const char* mqtt_server = "192.168.1.11"; // IP del broker (tu PC)

// Configuración de MQTT
WiFiClient espClient;
PubSubClient client(espClient);
long lastMsg = 0;
char msg[msg_len];

Servo pulgar, indice, medio, anular, menique;

TaskHandle_t TaskCommunication;
TaskHandle_t TaskControl;

typedef struct {
  Servo pulgar;
  Servo indice;
  Servo medio;
  Servo anular;
  Servo menique;
  bool CO[5] = {0, 0, 0, 0, 0}; //mano cerrada o abierta
  struct fd {
    int dPulgar;
    int dIndice;
    int dMedio;
    int dAnular;
    int dMenique;
    int *p0 = &dPulgar;
    int *p1 = &dIndice;
    int *p2 = &dMedio;
    int *p3 = &dAnular;
    int *p4 = &dMenique;
  } fingersDegrees;
} hand;

hand Hand_1;
hand ActualHand; //para manejar interrupciones

void OpenAllFingers(hand *mano);
float* msg2f_array(String msg);


void setup() {
  Serial.begin(115200);
  delay(5000);
  Serial.println("Empezando");

  // Allocate one hardware timer (timer 0) for all PWM operations.
  // All servos will share timer 0.
  //ESP32PWM::allocateTimer(0);
  
  Hand_1.pulgar.setPeriodHertz(50);
  Hand_1.pulgar.attach(pulgarPin, 500, 2400);

  Hand_1.indice.setPeriodHertz(50);
  Hand_1.indice.attach(indicePin, 500, 2400);

  Hand_1.medio.setPeriodHertz(50);
  Hand_1.medio.attach(medioPin, 500, 2400);

  Hand_1.anular.setPeriodHertz(50);
  Hand_1.anular.attach(anularPin, 500, 2400);

  Hand_1.menique.setPeriodHertz(50);
  Hand_1.menique.attach(meniquePin, 500, 2400);

  pinMode(led_Connected, OUTPUT);
  digitalWrite(led_Connected, LOW);
  
  OpenAllFingers(&Hand_1);

  Serial.println("Por mandar las funciones a los cores");
  delay(1000);
  // Crear la tarea para la comunicación en Core 0
  xTaskCreatePinnedToCore(
    TaskCommunication_func,  // Función de la tarea
    "TaskComunicacion", // Nombre de la tarea
    4096,             // Tamaño de la pila
    NULL,              // Parámetro de la tarea
    1,                 // Prioridad de la tarea
    &TaskCommunication,              // Handle de la tarea
    0);                // Core 0
  delay(500);
  // Crear la tarea para el control de actuadores en Core 1
  xTaskCreatePinnedToCore(
    TaskControl_func,       // Función de la tarea
    "TaskControl",     // Nombre de la tarea
    4096,             // Tamaño de la pila
    NULL,              // Parámetro de la tarea
    1,                 // Prioridad de la tarea
    &TaskControl,              // Handle de la tarea
    1);                // Core 1
  delay(500);
}

// Función que se ejecutará en el Core 0 (Comunicación)
void TaskCommunication_func(void *pvParameters) {
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  String received = "";
  //Serial.println("Aquí en Core 0");
  for(;;) {
    if (ESTADO == 1){
      if (!client.connected()) {
          digitalWrite(led_Connected, LOW);
          reconnect();
      }
      digitalWrite(led_Connected, HIGH);
      client.loop();
      vTaskDelay(10 / portTICK_PERIOD_MS); // Evitar consumir la CPU innecesariamente, revisar si es necesario
    }
  }
}

// Función que se ejecutará en el Core 1 (Control de actuadores)
void TaskControl_func(void *pvParameters) {
  Serial.println("Aquí en Core 1");
  for(;;) {
    if (ESTADO == 0) {
      Serial.println("Core 1:");
      
      writeDegrees(&Hand_1, Action[0], Action[1], Action[2], Action[3], Action[4]);
      moveFingers(&Hand_1, 0, 0.25); // void moveFingers(hand *h, bool openAgain, float TimeInSeconds)
      ESTADO = 1; // Una vez completado el movimiento, volver a permitir nuevas acciones
    }
    vTaskDelay(10 / portTICK_PERIOD_MS); // Evitar consumir la CPU innecesariamente
  }
}

void loop() {
  // El loop queda vacío porque las tareas están siendo manejadas por los dos núcleos
}

// Función para configurar la conexión Wi-Fi
void setup_wifi() {
    delay(100);
    Serial.println();
    Serial.print("Conectando a Wi-Fi: ");
    Serial.println(ssid);

    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("Wi-Fi conectado.");
    Serial.print("Dirección IP: ");
    Serial.println(WiFi.localIP());
}

// Callback para manejar mensajes recibidos por MQTT
void callback(char* topic, byte* message, unsigned int length) {
    Serial.print("Mensaje recibido en el tópico: ");
    Serial.println(topic);

    String mensaje;
    for (int i = 0; i < length; i++) {
        mensaje += (char)message[i];
    }
    Serial.print("Mensaje: ");
    Serial.println(mensaje);
    // Imprimir los valores para verificar
    Serial.print("Action: [");
    for (int i = 0; i < 5; i++) {
      Serial.print(Action[i], 10);
      Serial.print(", ");
    }
    Serial.print("]");

    float* combination_output = msg2f_array(mensaje);
    int* temp = agentout2degrees(combination_output);
    for (int i = 0; i < 5; i++) {
      Action[i] = temp[i];
    }

    ESTADO = 0;
}

int* agentout2degrees(float* combination_output){
  //float* temp = combination_output;
  static int action[5];
  for (int i = 0; i < 5; i++) {
      if (combination_output[i] < 0.45){
        action[i] = 0;
      } else if (combination_output[i] < 1.3){
        action[i] = 135;
      } else{
        action[i] = 180;
      }
    }
  return action;
}

float* msg2f_array(String msg) {
  // Remove the first bracket if it exists (e.g. '[')
  msg.remove(0, 1); // remove first character
  // Optionally remove trailing ']' if present
  msg.remove(msg.length() - 1);

  // We'll look for the positions of commas.
  // Since we expect exactly 5 values, there should be 4 commas.
  int indexes[4];
  int j = 0;
  for (int i = 0; i < msg.length(); i++) {
    if (msg[i] == ',') {
      indexes[j] = i;
      j++;
      // Once we have 4 commas, stop
      if (j == 4) {
        break;
      }
    }
  }

  // We'll store the floats in a static array.
  // "static" allows us to return its address safely (with limitations).
  static float parsed[5];

  // First value: from start of string to first comma
  parsed[0] = msg.substring(0, indexes[0]).toFloat();

  // Middle three values
  for (int i = 1; i < 4; i++) {
    parsed[i] = msg.substring(indexes[i - 1] + 1, indexes[i]).toFloat();
  }

  // Last value: from last comma to the end of the string
  parsed[4] = msg.substring(indexes[3] + 1).toFloat();

  return parsed;
}


// Reconectar al broker MQTT si la conexión se pierde
void reconnect() {
    while (!client.connected()) {
        Serial.print("Intentando conectar al broker MQTT...");
        if (client.connect("ESP8266Client")) {
            Serial.println("Conectado.");
            client.subscribe("test"); // Subscribirse a mensajes de configuración (opcional)
        } else {
            Serial.print("Error de conexión, estado: ");
            Serial.println(client.state());
            delay(5000);
        }
    }
}

void writeDegrees(hand *h, int d0, int d1, int d2, int d3, int d4) {
  //Cada dx correscponde a los grados de un dedo, el pulgar es el 0 y el menique el 4
  *(h->fingersDegrees.p0) = d0;
  *(h->fingersDegrees.p1) = d1;
  *(h->fingersDegrees.p2) = d2;
  *(h->fingersDegrees.p3) = d3;
  *(h->fingersDegrees.p4) = d4;
}

void OpenAllFingers(hand *mano) {
  digitalWrite(ledMoving, LOW);
  mano->pulgar.write(0);
  mano->indice.write(0);
  mano->medio.write(0);
  mano->anular.write(0);
  mano->menique.write(0);
  vTaskDelay(MOTION_TIME / portTICK_PERIOD_MS);
}

void CloseAllFingers(hand *mano) {
  digitalWrite(ledMoving, LOW);
  mano->pulgar.write(180);
  mano->indice.write(180);
  mano->medio.write(180);
  mano->anular.write(180);
  mano->menique.write(180);
  vTaskDelay(MOTION_TIME / portTICK_PERIOD_MS);
}

void moveOneFinger(Servo finger, int fingerDegrees) {
  digitalWrite(ledMoving, LOW);
  finger.write(fingerDegrees);
  vTaskDelay(MOTION_TIME / portTICK_PERIOD_MS);
}

void moveWnum(Servo finger, int fingerDegrees, bool *CO) { //CHECK funcionalidad de esta función
  if (*CO == 1) {
    moveOneFinger(finger, 0);
    *CO = 0;
  } else {
    moveOneFinger(finger, fingerDegrees);
    *CO = 1; //check
  }
}

void moveFingers(hand *h, bool openAgain, float TimeInSeconds) {
  Serial.println("moviendo dedos\n");
  digitalWrite(ledMoving, LOW);
  h->pulgar.write(h->fingersDegrees.dPulgar);
  h->indice.write(h->fingersDegrees.dIndice);
  h->medio.write(h->fingersDegrees.dMedio);
  h->anular.write(h->fingersDegrees.dAnular);
  h->menique.write(h->fingersDegrees.dMenique);

  if (OpenAgain == 1) {
    OpenAgain = 0;
    OpenAllFingers(h);
  }
  vTaskDelay(TimeInSeconds*1000 / portTICK_PERIOD_MS);
}

void Wave(hand *h, float timeInSeconds, int repetir) {
  int nums[6] = {180, 0, 0, 0, 0, 0};
  for (int j = 0; j < repetir; j++) {
    for (int i = 0; i < 5; i++) {
      writeDegrees(h, nums[0], nums[1], nums[2], nums[3], nums[4]);
      moveFingers(h, 1, timeInSeconds);
      nums[i] = 0;
      nums[i + 1] = 180;
      delay(250);
    }
  }
}

void CloseOpen(hand *h, float timeInSeconds, bool openAgain) {
  writeDegrees(h, 180, 180, 180, 180, 180);
  moveFingers(h, openAgain, timeInSeconds);
}

void RockAndRoll01(hand *h, float timeInSeconds, bool openAgain) {
  writeDegrees(h, 0, 0, 180, 180, 0);
  moveFingers(h, openAgain, timeInSeconds);
}

void RockAndRoll02(hand *h, float timeInSeconds, bool openAgain) {
  writeDegrees(h, 180, 0, 180, 180, 0);
  moveFingers(h, openAgain, timeInSeconds);
}

void Chill(hand *h, float timeInSeconds, bool openAgain) {
  writeDegrees(h, 0, 180, 180, 180, 0);
  moveFingers(h, openAgain, timeInSeconds);
}

void FuckYou(hand *h, float timeInSeconds, bool openAgain) {
  writeDegrees(h, 180, 180, 0, 180, 180);
  moveFingers(h, openAgain, timeInSeconds);
}

void Great(hand *h, float timeInSeconds, bool openAgain) {
  writeDegrees(h, 180, 150, 0, 0, 0);
  moveFingers(h, openAgain, timeInSeconds);
}

void Ok(hand *h, float timeInSeconds, bool openAgain) {
  writeDegrees(h, 0, 180, 180, 180, 180);
  moveFingers(h, openAgain, timeInSeconds);
}

void Gun(hand *h, float timeInSeconds, bool openAgain, int shots) {
  writeDegrees(h, 0, 0, 0, 180, 180);
  moveFingers(h, 0, timeInSeconds);
  for (int i = 0; i < shots; i++) {
    moveOneFinger(h->pulgar, 170);
  }
  if (openAgain == 1) {
    writeDegrees(h, 0, 0, 0, 0, 0);
    moveFingers(h, 1, timeInSeconds);
  }
}

void Coreo01(hand *h, float timeInSeconds) {
  //Wave(h, timeInSeconds, 2);
  CloseOpen(h, timeInSeconds, 1);
  RockAndRoll01(h, timeInSeconds, 1);
  RockAndRoll02(h, timeInSeconds, 1);
  FuckYou(h, timeInSeconds, 1);
  Chill(h, timeInSeconds, 1);
  Gun(h, timeInSeconds, 1, 2);
  Ok(h, timeInSeconds, 1);
  Great(h, timeInSeconds, 1);
}

void randomRoutine(hand *h) {
  uint8_t numeroAleatorio = random(0, 7);
  switch (numeroAleatorio) {
    case 0:
      Great(h, 3, 1);  // Pasar el puntero directamente
      break;
    case 1:
      RockAndRoll01(h, 3, 1);  // Pasar el puntero directamente
      break;
    case 2:
      RockAndRoll02(h, 3, 1);  // Pasar el puntero directamente
      break;
    case 3:
      Chill(h, 3, 1);  // Pasar el puntero directamente
      break;
    case 4:
      Ok(h, 3, 1);  // Pasar el puntero directamente
      break;
    case 5:
      CloseOpen(h, 3, 1);  // Pasar el puntero directamente
      break;
    case 6:
      Great(h, 3, 1);  // Repetir para aumentar la probabilidad
      break;
  }
}

void CoreoRight01(hand *h, float timeInSeconds) {
  vTaskDelay(timeInSeconds*1000 + 500 / portTICK_PERIOD_MS);
  Chill(h, timeInSeconds, 0);
  vTaskDelay(timeInSeconds*1000 + 500 / portTICK_PERIOD_MS);
  RockAndRoll01(h, timeInSeconds, 0);
  vTaskDelay(timeInSeconds*1000 + 500 / portTICK_PERIOD_MS);
  CloseOpen(h, timeInSeconds, 1);
}

void CoreoLeft01(hand *h, float timeInSeconds) {
  Great(h, timeInSeconds, 0);
  vTaskDelay(timeInSeconds*1000 + 500 / portTICK_PERIOD_MS);
  Ok(h, timeInSeconds, 0);
  vTaskDelay(timeInSeconds*1000 + 500 / portTICK_PERIOD_MS);
  CloseOpen(h, timeInSeconds, 1);
  RockAndRoll02(h, timeInSeconds, 1);
}