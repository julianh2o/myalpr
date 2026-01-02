import paho.mqtt.client as mqtt
import json
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "homeassistant.local")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# MQTT topics
DISCOVERY_PREFIX = "homeassistant"
DEVICE_ID = "driveway_alpr"
DEVICE_NAME = "Driveway ALPR"


class MQTTPublisher:
    """Publishes car detection events to Home Assistant via MQTT."""

    def __init__(self):
        self.client = None
        self.connected = False

    def connect(self):
        """Connect to MQTT broker and setup auto-discovery."""
        try:
            self.client = mqtt.Client()

            if MQTT_USER and MQTT_PASSWORD:
                self.client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()

            print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")

        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            self.connected = False

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            self.connected = True
            print("✓ Connected to MQTT broker successfully")
            self._setup_discovery()
        else:
            self.connected = False
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorized (check MQTT_USER/MQTT_PASSWORD in .env)"
            }
            error_msg = error_messages.get(rc, f"Unknown error code {rc}")
            print(f"✗ Failed to connect to MQTT broker: {error_msg}")
            print("  MQTT publishing will be disabled. The system will continue without MQTT.")

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker."""
        self.connected = False
        print("Disconnected from MQTT broker")

    def _setup_discovery(self):
        """Setup Home Assistant auto-discovery for sensors."""
        device_info = {
            "identifiers": [DEVICE_ID],
            "name": DEVICE_NAME,
            "model": "License Plate Recognition",
            "manufacturer": "Custom"
        }

        # Sensor: Last Plate Number
        plate_config = {
            "name": "Last License Plate",
            "unique_id": f"{DEVICE_ID}_last_plate",
            "state_topic": f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/plate/state",
            "json_attributes_topic": f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/plate/attributes",
            "icon": "mdi:car",
            "device": device_info
        }
        self.client.publish(
            f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_plate/config",
            json.dumps(plate_config),
            retain=True
        )

        # Sensor: Last Direction
        direction_config = {
            "name": "Last Car Direction",
            "unique_id": f"{DEVICE_ID}_direction",
            "state_topic": f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/direction/state",
            "icon": "mdi:arrow-left-right",
            "device": device_info
        }
        self.client.publish(
            f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_direction/config",
            json.dumps(direction_config),
            retain=True
        )

        # Sensor: Last Detection Time
        time_config = {
            "name": "Last Car Detection",
            "unique_id": f"{DEVICE_ID}_last_detection",
            "state_topic": f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/timestamp/state",
            "device_class": "timestamp",
            "icon": "mdi:clock-outline",
            "device": device_info
        }
        self.client.publish(
            f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_timestamp/config",
            json.dumps(time_config),
            retain=True
        )

        print("MQTT auto-discovery configured")

    def publish_detection(self, plate_number, direction, timestamp=None):
        """
        Publish car detection to Home Assistant.

        Args:
            plate_number: License plate number (string or None)
            direction: "arriving" or "departing"
            timestamp: Optional timestamp (defaults to now)
        """
        if not self.connected:
            return  # Silently skip if not connected

        if timestamp is None:
            timestamp = datetime.now()

        # Format timestamp as ISO 8601 for Home Assistant
        timestamp_iso = timestamp.isoformat()

        # Publish plate number
        plate_value = plate_number if plate_number else "Unknown"
        self.client.publish(
            f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/plate/state",
            plate_value
        )

        # Publish attributes for plate sensor
        attributes = {
            "direction": direction,
            "detected_at": timestamp_iso,
            "plate_read": plate_number is not None
        }
        self.client.publish(
            f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/plate/attributes",
            json.dumps(attributes)
        )

        # Publish direction
        self.client.publish(
            f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/direction/state",
            direction
        )

        # Publish timestamp
        self.client.publish(
            f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/timestamp/state",
            timestamp_iso
        )

        print(f"Published to MQTT: {plate_value} ({direction}) at {timestamp_iso}")

    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            print("Disconnected from MQTT broker")


# Global MQTT publisher instance
_mqtt_publisher = None


def get_mqtt_publisher():
    """Get or create the global MQTT publisher instance."""
    global _mqtt_publisher
    if _mqtt_publisher is None:
        _mqtt_publisher = MQTTPublisher()
        _mqtt_publisher.connect()
    return _mqtt_publisher
