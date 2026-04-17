#!/usr/bin/env python3
"""Generate PlatformIO training Q&A pairs.

Covers CLI commands, platformio.ini config, board definitions, frameworks,
library management, debugging, CI/CD, and common errors.

Output: JSONL to stdout.
"""
from __future__ import annotations

import json
import random
import sys

random.seed(42)

DOMAIN = "platformio"

# ---------------------------------------------------------------------------
# Data banks
# ---------------------------------------------------------------------------

BOARDS = {
    "esp32dev": {"mcu": "ESP32", "framework": ["arduino", "espidf"], "f_cpu": "240MHz", "flash": "4MB", "ram": "520KB"},
    "esp32-s3-devkitc-1": {"mcu": "ESP32-S3", "framework": ["arduino", "espidf"], "f_cpu": "240MHz", "flash": "8MB", "ram": "512KB"},
    "esp32-c3-devkitm-1": {"mcu": "ESP32-C3", "framework": ["arduino", "espidf"], "f_cpu": "160MHz", "flash": "4MB", "ram": "400KB"},
    "esp32-c6-devkitc-1": {"mcu": "ESP32-C6", "framework": ["arduino", "espidf"], "f_cpu": "160MHz", "flash": "8MB", "ram": "512KB"},
    "esp32-h2-devkitm-1": {"mcu": "ESP32-H2", "framework": ["espidf"], "f_cpu": "96MHz", "flash": "4MB", "ram": "320KB"},
    "nanoatmega328": {"mcu": "ATmega328P", "framework": ["arduino"], "f_cpu": "16MHz", "flash": "32KB", "ram": "2KB"},
    "uno": {"mcu": "ATmega328P", "framework": ["arduino"], "f_cpu": "16MHz", "flash": "32KB", "ram": "2KB"},
    "megaatmega2560": {"mcu": "ATmega2560", "framework": ["arduino"], "f_cpu": "16MHz", "flash": "256KB", "ram": "8KB"},
    "due": {"mcu": "AT91SAM3X8E", "framework": ["arduino"], "f_cpu": "84MHz", "flash": "512KB", "ram": "96KB"},
    "nucleo_f103rb": {"mcu": "STM32F103RBT6", "framework": ["arduino", "stm32cube", "zephyr"], "f_cpu": "72MHz", "flash": "128KB", "ram": "20KB"},
    "nucleo_f401re": {"mcu": "STM32F401RET6", "framework": ["arduino", "stm32cube", "zephyr"], "f_cpu": "84MHz", "flash": "512KB", "ram": "96KB"},
    "nucleo_f446re": {"mcu": "STM32F446RET6", "framework": ["arduino", "stm32cube", "zephyr"], "f_cpu": "180MHz", "flash": "512KB", "ram": "128KB"},
    "nucleo_h743zi": {"mcu": "STM32H743ZIT6", "framework": ["stm32cube", "zephyr"], "f_cpu": "480MHz", "flash": "2MB", "ram": "1MB"},
    "nucleo_l476rg": {"mcu": "STM32L476RGT6", "framework": ["arduino", "stm32cube", "zephyr"], "f_cpu": "80MHz", "flash": "1MB", "ram": "128KB"},
    "bluepill_f103c8": {"mcu": "STM32F103C8T6", "framework": ["arduino", "stm32cube"], "f_cpu": "72MHz", "flash": "64KB", "ram": "20KB"},
    "blackpill_f411ce": {"mcu": "STM32F411CEU6", "framework": ["arduino", "stm32cube"], "f_cpu": "100MHz", "flash": "512KB", "ram": "128KB"},
    "teensy41": {"mcu": "IMXRT1062", "framework": ["arduino"], "f_cpu": "600MHz", "flash": "8MB", "ram": "1024KB"},
    "teensy40": {"mcu": "IMXRT1062", "framework": ["arduino"], "f_cpu": "600MHz", "flash": "2MB", "ram": "1024KB"},
    "teensy36": {"mcu": "MK66FX1M0", "framework": ["arduino"], "f_cpu": "180MHz", "flash": "1MB", "ram": "256KB"},
    "pico": {"mcu": "RP2040", "framework": ["arduino"], "f_cpu": "133MHz", "flash": "2MB", "ram": "264KB"},
    "rpipicow": {"mcu": "RP2040", "framework": ["arduino"], "f_cpu": "133MHz", "flash": "2MB", "ram": "264KB"},
    "adafruit_feather_m0": {"mcu": "ATSAMD21G18A", "framework": ["arduino"], "f_cpu": "48MHz", "flash": "256KB", "ram": "32KB"},
    "seeed_xiao_esp32s3": {"mcu": "ESP32-S3", "framework": ["arduino", "espidf"], "f_cpu": "240MHz", "flash": "8MB", "ram": "512KB"},
    "d1_mini": {"mcu": "ESP8266", "framework": ["arduino"], "f_cpu": "80MHz", "flash": "4MB", "ram": "80KB"},
    "nodemcuv2": {"mcu": "ESP8266", "framework": ["arduino"], "f_cpu": "80MHz", "flash": "4MB", "ram": "80KB"},
}

LIBRARIES = [
    ("bblanchon/ArduinoJson", "7.3.0", "JSON serialization/deserialization"),
    ("knolleary/PubSubClient", "2.8", "MQTT client"),
    ("adafruit/Adafruit BME280 Library", "2.2.4", "BME280 sensor driver"),
    ("adafruit/Adafruit NeoPixel", "1.12.3", "WS2812B LED driver"),
    ("me-no-dev/ESPAsyncWebServer", "1.2.7", "Async HTTP server for ESP"),
    ("me-no-dev/AsyncTCP", "1.1.4", "Async TCP for ESP32"),
    ("Wire", None, "I2C built-in"),
    ("SPI", None, "SPI built-in"),
    ("sparkfun/SparkFun BME280", "2.0.9", "BME280 alternative"),
    ("pololu/VL53L0X", "1.3.1", "ToF distance sensor"),
    ("adafruit/Adafruit SSD1306", "2.5.12", "OLED display driver"),
    ("adafruit/Adafruit GFX Library", "1.11.11", "Graphics primitives"),
    ("sandeepmistry/arduino-CAN", "0.3.1", "CAN bus"),
    ("miwagner/ESP32-Arduino-CAN", "2.0.0", "ESP32 native CAN"),
    ("fastled/FastLED", "3.7.8", "High-perf LED strip driver"),
    ("h2zero/NimBLE-Arduino", "2.1.2", "BLE stack"),
    ("thingpulse/ESP8266 and ESP32 OLED driver for SSD1306 displays", "4.6.1", "SSD1306 OLED"),
    ("tzapu/WiFiManager", "2.0.17", "WiFi provisioning portal"),
    ("paulstoffregen/OneWire", "2.3.8", "1-Wire protocol"),
    ("milesburton/DallasTemperature", "4.0.4", "DS18B20 temperature"),
    ("bolderflight/sbus", "8.1.4", "SBUS RC protocol"),
    ("jrowberg/i2cdevlib", None, "MPU6050 driver"),
    ("lvgl/lvgl", "9.2.2", "Graphics UI framework"),
    ("espressif/arduino-esp32", None, "ESP32 Arduino core"),
    ("stm32duino/STM32duino FreeRTOS", "10.3.2", "FreeRTOS for STM32"),
]

UPLOAD_SPEEDS = [9600, 19200, 57600, 115200, 230400, 460800, 500000, 921600, 1500000, 2000000]
MONITOR_SPEEDS = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 500000, 921600]
BUILD_FLAGS = [
    "-DCORE_DEBUG_LEVEL=5",
    "-DCONFIG_ASYNC_TCP_RUNNING_CORE=1",
    "-DBOARD_HAS_PSRAM",
    "-mfix-esp32-psram-cache-issue",
    "-DARDUINO_USB_MODE=1",
    "-DARDUINO_USB_CDC_ON_BOOT=1",
    "-DCONFIG_BT_NIMBLE_ROLE_CENTRAL_DISABLED",
    "-DCONFIG_BT_NIMBLE_ROLE_OBSERVER_DISABLED",
    "-DCONFIG_SPIRAM_CACHE_WORKAROUND",
    "-Os",
    "-O2",
    "-O3",
    "-Wall -Wextra",
    "-std=gnu++17",
    "-fno-rtti",
    "-fno-exceptions",
    '-DFIRMWARE_VERSION=\\"1.0.0\\"',
    "-DCONFIG_FREERTOS_HZ=1000",
    "-DCONFIG_ESP32_WIFI_TX_BUFFER_TYPE=0",
]

DEBUG_TOOLS = ["esp-prog", "jlink", "stlink", "cmsis-dap", "esp-builtin", "olimex-arm-usb-ocd-h", "atmel-ice", "blackmagic"]
DEBUG_INTERFACES = ["jtag", "swd"]

PARTITION_TABLES = {
    "default": "Default (1.2MB app, 1.5MB SPIFFS)",
    "no_ota": "No OTA (2MB app, 2MB SPIFFS)",
    "min_spiffs": "Minimal SPIFFS (1.9MB app, 128KB SPIFFS)",
    "huge_app": "Huge APP (3MB app, no OTA, no SPIFFS)",
    "default_16MB": "16MB flash (6.5MB app, 6.5MB SPIFFS)",
    "custom": "Custom CSV partition table",
}

PLATFORMS = [
    ("espressif32", ["esp32dev", "esp32-s3-devkitc-1", "esp32-c3-devkitm-1"]),
    ("espressif8266", ["d1_mini", "nodemcuv2"]),
    ("atmelavr", ["nanoatmega328", "uno", "megaatmega2560"]),
    ("atmelsam", ["due", "adafruit_feather_m0"]),
    ("ststm32", ["nucleo_f103rb", "nucleo_f401re", "nucleo_h743zi", "bluepill_f103c8"]),
    ("teensy", ["teensy41", "teensy40", "teensy36"]),
    ("raspberrypi", ["pico", "rpipicow"]),
]

COMMON_ERRORS = [
    {
        "error": "Error: Could not find the package with 'SomeLib' requirements for your system",
        "cause": "Library name misspelled or not available for the current platform.",
        "fix": "Check the correct library name with `pio pkg search SomeLib`. Use the full owner/name format in lib_deps.",
    },
    {
        "error": "Error: Please specify `upload_port` for environment or use global `--upload-port` option.",
        "cause": "PlatformIO cannot auto-detect the serial port.",
        "fix": "List available ports with `pio device list`, then set `upload_port = /dev/ttyUSB0` (Linux) or `upload_port = COM3` (Windows) in platformio.ini.",
    },
    {
        "error": "A fatal error occurred: Failed to connect to ESP32: No serial data received.",
        "cause": "The ESP32 is not in bootloader mode, or the USB cable is charge-only.",
        "fix": "Hold the BOOT button while pressing RST, then release. Use a data-capable USB cable. On ESP32-S3, add `upload_speed = 460800`.",
    },
    {
        "error": "error: 'Serial' was not declared in this scope",
        "cause": "On ESP32-S3 with USB CDC, the default Serial object is not available.",
        "fix": "Add `-DARDUINO_USB_CDC_ON_BOOT=1` to build_flags, or use `USBSerial` instead of `Serial`.",
    },
    {
        "error": "region `iram0_0_seg' overflowed by N bytes",
        "cause": "IRAM section full, usually from too many IRAM_ATTR functions or large interrupt handlers.",
        "fix": "Remove unnecessary IRAM_ATTR, reduce ISR size, or use `board_build.partitions = min_spiffs.csv` to free flash for code.",
    },
    {
        "error": "Sketch too big; see https://...",
        "cause": "Compiled binary exceeds the partition size.",
        "fix": "Switch to a larger partition scheme: `board_build.partitions = huge_app.csv` or `no_ota.csv`. Alternatively, optimize code size with `-Os` build flag.",
    },
    {
        "error": "fatal error: WiFi.h: No such file or directory",
        "cause": "Framework mismatch — Arduino framework not installed or wrong platform.",
        "fix": "Ensure `framework = arduino` is set and the platform matches the board. For ESP32, use `platform = espressif32`.",
    },
    {
        "error": "Error: Traceback ... pkg_resources.DistributionNotFound",
        "cause": "Python package conflict in PlatformIO's virtual environment.",
        "fix": "Run `pio upgrade --dev` and `pip install -U platformio`. If persistent, delete `~/.platformio` and reinstall.",
    },
    {
        "error": "TimeoutError: Could not get response from device",
        "cause": "Debug probe cannot connect to the target MCU.",
        "fix": "Check wiring (SWDIO, SWCLK, GND, VCC). Reduce debug speed: `debug_speed = 1000`. Ensure target is powered.",
    },
    {
        "error": "ld: symbol multiply defined",
        "cause": "Multiple definitions of the same symbol, often from including .cpp files or conflicting libraries.",
        "fix": "Use header guards, don't include .cpp files. Check for duplicate library entries in lib_deps. Use `lib_compat_mode = strict`.",
    },
    {
        "error": "Error: Unknown board ID 'board_name'",
        "cause": "Board not found in PlatformIO registry.",
        "fix": "Search available boards with `pio boards | grep keyword`. Install the correct platform: `pio platform install platformname`.",
    },
    {
        "error": "PSRAM not found or PSRAM ID read error",
        "cause": "PSRAM enabled in config but not present on the board, or wrong GPIO configuration.",
        "fix": "Remove `-DBOARD_HAS_PSRAM` from build_flags if the board has no PSRAM. For boards with PSRAM, ensure `board_build.arduino.memory_type = qio_opi` is correct.",
    },
]

ESPIDF_COMPONENTS = [
    "esp_wifi", "esp_netif", "esp_http_server", "esp_https_ota",
    "esp_event", "nvs_flash", "esp_partition", "esp_timer",
    "driver", "spi_flash", "esp_adc", "esp_lcd",
    "freertos", "mbedtls", "esp_tls", "mqtt",
    "esp_websocket_client", "esp_http_client", "protocomm",
    "wifi_provisioning", "esp_local_ctrl", "esp_diagnostics",
    "bt", "esp_ble_mesh", "esp_coex",
]

CI_RUNNERS = ["ubuntu-latest", "macos-latest", "windows-latest"]

# ---------------------------------------------------------------------------
# Template generators
# ---------------------------------------------------------------------------

def emit(user: str, assistant: str) -> None:
    rec = {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "domain": DOMAIN,
        "source": "synthetic",
    }
    print(json.dumps(rec, ensure_ascii=False))


def gen_basic_ini():
    """Basic platformio.ini configuration questions."""
    for board_id, info in BOARDS.items():
        fw = random.choice(info["framework"])
        speed = random.choice(UPLOAD_SPEEDS)
        mon = random.choice(MONITOR_SPEEDS)

        emit(
            f"Write a platformio.ini for {info['mcu']} ({board_id}) with {fw} framework, {speed} baud upload, and {mon} baud serial monitor.",
            f"""Here is a platformio.ini configuration for the {info['mcu']}:

```ini
[env:{board_id}]
platform = {_platform_for(board_id)}
board = {board_id}
framework = {fw}
upload_speed = {speed}
monitor_speed = {mon}
```

Key settings:
- **board**: `{board_id}` targets the {info['mcu']} with {info['flash']} flash and {info['ram']} RAM
- **framework**: `{fw}` provides the {"Arduino API abstraction" if fw == "arduino" else "native " + fw + " SDK"}
- **upload_speed**: {speed} baud for flashing
- **monitor_speed**: {mon} baud for serial output"""
        )


def gen_multi_env():
    """Multi-environment configurations."""
    board_pairs = [
        ("esp32dev", "nucleo_f103rb"),
        ("esp32-s3-devkitc-1", "teensy41"),
        ("pico", "nanoatmega328"),
        ("d1_mini", "esp32-c3-devkitm-1"),
        ("bluepill_f103c8", "esp32dev"),
        ("nucleo_f401re", "nucleo_h743zi"),
        ("teensy40", "adafruit_feather_m0"),
        ("esp32-c6-devkitc-1", "seeed_xiao_esp32s3"),
    ]
    for b1, b2 in board_pairs:
        i1, i2 = BOARDS[b1], BOARDS[b2]
        fw1 = random.choice(i1["framework"])
        fw2 = random.choice(i2["framework"])
        emit(
            f"Create a platformio.ini with two environments: one for {i1['mcu']} ({b1}) and one for {i2['mcu']} ({b2}). Share common settings in [env].",
            f"""```ini
[platformio]
default_envs = {b1}

[env]
monitor_speed = 115200
build_flags = -Wall

[env:{b1}]
platform = {_platform_for(b1)}
board = {b1}
framework = {fw1}
upload_speed = {random.choice(UPLOAD_SPEEDS)}

[env:{b2}]
platform = {_platform_for(b2)}
board = {b2}
framework = {fw2}
upload_speed = {random.choice(UPLOAD_SPEEDS)}
```

The `[env]` section defines shared settings inherited by all environments. Use `pio run -e {b1}` or `pio run -e {b2}` to build a specific environment. The `default_envs` directive sets which environment builds when you run `pio run` without `-e`."""
        )


def gen_lib_deps():
    """Library dependency questions."""
    for _ in range(80):
        libs = random.sample(LIBRARIES, k=random.randint(2, 5))
        board_id = random.choice(list(BOARDS.keys()))
        info = BOARDS[board_id]
        fw = random.choice(info["framework"])

        lib_lines = []
        lib_descs = []
        for name, ver, desc in libs:
            if ver:
                lib_lines.append(f"    {name}@^{ver}")
            else:
                lib_lines.append(f"    {name}")
            lib_descs.append(f"- **{name}**{' @ ' + ver if ver else ''}: {desc}")

        lib_block = "\n".join(lib_lines)
        desc_block = "\n".join(lib_descs)

        emit(
            f"How do I add {', '.join(n for n, _, _ in libs)} to my PlatformIO project for {info['mcu']}?",
            f"""Add them to `lib_deps` in your `platformio.ini`:

```ini
[env:{board_id}]
platform = {_platform_for(board_id)}
board = {board_id}
framework = {fw}
lib_deps =
{lib_block}
```

Libraries included:
{desc_block}

PlatformIO will automatically download and install these on the next build. The `^` prefix allows compatible minor/patch updates. You can also install individually:

```bash
pio pkg install --library "{libs[0][0]}"
```"""
        )


def gen_cli_commands():
    """CLI command reference questions."""
    commands = [
        {
            "cmd": "pio run",
            "q": "How do I build my PlatformIO project from the command line?",
            "a": """Use `pio run` to compile your project:

```bash
# Build all environments
pio run

# Build specific environment
pio run -e esp32dev

# Build with verbose output
pio run -v

# Clean build artifacts first
pio run -t clean

# Build and show memory usage
pio run -t size
```

The compiled firmware is placed in `.pio/build/<env>/firmware.bin`. Use `pio run -t upload` to build and flash in one step."""
        },
        {
            "cmd": "pio test",
            "q": "How do I run unit tests with PlatformIO?",
            "a": """PlatformIO supports native and embedded unit testing:

```bash
# Run all tests
pio test

# Run tests for a specific environment
pio test -e native

# Run a specific test suite
pio test -f test_sensors

# Run with verbose output
pio test -v

# Run only on native (desktop) target
pio test -e native --without-uploading
```

Test files go in the `test/` directory:
```
test/
├── test_sensors/
│   └── test_main.cpp
├── test_comms/
│   └── test_main.cpp
└── test_native/
    └── test_main.cpp
```

Example test file using Unity framework:
```cpp
#include <unity.h>

void test_sensor_init(void) {
    TEST_ASSERT_EQUAL(0, sensor_init());
}

void test_sensor_read(void) {
    float val = sensor_read();
    TEST_ASSERT_FLOAT_WITHIN(0.5, 25.0, val);
}

int main(int argc, char **argv) {
    UNITY_BEGIN();
    RUN_TEST(test_sensor_init);
    RUN_TEST(test_sensor_read);
    return UNITY_END();
}
```

Add a native test environment to platformio.ini:
```ini
[env:native]
platform = native
test_framework = unity
```"""
        },
        {
            "cmd": "pio device",
            "q": "How do I list connected devices and open serial monitor in PlatformIO?",
            "a": """Use `pio device` subcommands:

```bash
# List all connected serial devices
pio device list

# Open serial monitor (default 9600 baud)
pio device monitor

# Specify baud rate
pio device monitor -b 115200

# Specify port
pio device monitor -p /dev/ttyUSB0

# Add filters (timestamp, colorize, esp32_exception_decoder)
pio device monitor -f esp32_exception_decoder -f time

# Monitor with log to file
pio device monitor --log-file output.log
```

You can also set defaults in platformio.ini:
```ini
[env:esp32dev]
monitor_speed = 115200
monitor_filters = esp32_exception_decoder, time
monitor_port = /dev/ttyUSB0
```"""
        },
        {
            "cmd": "pio pkg",
            "q": "How do I manage libraries and packages with PlatformIO CLI?",
            "a": """Use `pio pkg` for library and package management:

```bash
# Search for a library
pio pkg search "json"

# Install a library
pio pkg install --library "bblanchon/ArduinoJson@^7.3.0"

# Install a library globally
pio pkg install -g --library "bblanchon/ArduinoJson"

# List installed libraries
pio pkg list

# Update all libraries
pio pkg update

# Uninstall a library
pio pkg uninstall --library "ArduinoJson"

# Install a specific platform
pio pkg install --platform "espressif32"

# Install a tool
pio pkg install --tool "tool-openocd-esp32"
```

Libraries are cached in `~/.platformio/packages/`. Project-specific libraries go in `.pio/libdeps/`."""
        },
        {
            "cmd": "pio boards",
            "q": "How do I find available boards in PlatformIO?",
            "a": """Use `pio boards` to search:

```bash
# List all boards
pio boards

# Filter by keyword
pio boards "esp32"
pio boards "nucleo"
pio boards "teensy"

# Get details about a specific board
pio boards --id-only | grep -i "stm32"

# JSON output for scripting
pio boards --json-output
```

Output includes board ID, MCU, frequency, flash, and RAM. Use the board ID in your platformio.ini `board = ` setting.

Common board IDs:
- ESP32: `esp32dev`, `esp32-s3-devkitc-1`, `esp32-c3-devkitm-1`
- STM32: `nucleo_f103rb`, `nucleo_f401re`, `bluepill_f103c8`
- AVR: `uno`, `nanoatmega328`, `megaatmega2560`
- Teensy: `teensy41`, `teensy40`
- RP2040: `pico`, `rpipicow`"""
        },
        {
            "cmd": "pio remote",
            "q": "How do I flash and monitor a remote device over the network with PlatformIO?",
            "a": """PlatformIO Remote allows over-the-network flashing:

```bash
# Start the remote agent on the machine with the device
pio remote agent start

# List remote agents from your dev machine
pio remote agent list

# Build and upload remotely
pio remote run -e esp32dev -t upload

# Open remote serial monitor
pio remote device monitor -b 115200

# List devices on remote agent
pio remote device list
```

Requirements:
1. PlatformIO account (free tier works)
2. `pio remote agent start` running on the target machine
3. Both machines logged in with `pio account login`

This is useful for CI/CD hardware-in-the-loop testing and remote development."""
        },
        {
            "cmd": "pio check",
            "q": "How do I run static analysis on my PlatformIO project?",
            "a": """PlatformIO integrates cppcheck and other static analyzers:

```bash
# Run default static analysis (cppcheck)
pio check

# Specify environment
pio check -e esp32dev

# Use specific tool
pio check --tool cppcheck
pio check --tool pvs-studio

# Set severity filter
pio check --severity high
pio check --severity medium

# Check specific directories
pio check --src-filters "+<src/*>" --src-filters "-<test/*>"

# JSON output for CI
pio check --json-output
```

Configure in platformio.ini:
```ini
[env:esp32dev]
check_tool = cppcheck
check_flags =
    cppcheck: --enable=all --std=c++17
check_severity = medium, high
```"""
        },
        {
            "cmd": "pio project init",
            "q": "How do I create a new PlatformIO project from the command line?",
            "a": """Use `pio project init` to scaffold a new project:

```bash
# Create project in current directory
pio project init --board esp32dev

# Create in specific directory
pio project init -d my_project --board esp32dev

# With specific framework
pio project init --board esp32dev --project-option "framework=arduino"

# Multiple boards
pio project init --board esp32dev --board nucleo_f103rb

# With IDE integration
pio project init --ide vscode --board esp32dev
```

This creates the standard project structure:
```
my_project/
├── include/        # Header files
│   └── README
├── lib/            # Project-specific libraries
│   └── README
├── src/            # Source files
│   └── main.cpp
├── test/           # Unit tests
│   └── README
└── platformio.ini  # Configuration
```"""
        },
        {
            "cmd": "pio upgrade",
            "q": "How do I update PlatformIO Core and packages?",
            "a": """Keep PlatformIO up to date:

```bash
# Upgrade PlatformIO Core
pio upgrade

# Upgrade to development version
pio upgrade --dev

# Update all installed platforms
pio pkg update -g

# Update a specific platform
pio pkg update -g -p espressif32

# Update all project dependencies
pio pkg update

# Check current version
pio --version

# Self-diagnostics
pio system info
```

If you encounter issues after upgrading:
```bash
# Clear cache
pio system prune

# Rebuild project
pio run -t clean && pio run
```"""
        },
    ]
    for c in commands:
        emit(c["q"], c["a"])

    # Additional CLI variations
    for board_id in random.sample(list(BOARDS.keys()), 15):
        info = BOARDS[board_id]
        emit(
            f"How do I upload firmware to {info['mcu']} using PlatformIO CLI?",
            f"""Upload to {info['mcu']} ({board_id}):

```bash
# Build and upload
pio run -e {board_id} -t upload

# Upload to specific port
pio run -e {board_id} -t upload --upload-port /dev/ttyUSB0

# Upload with verbose output
pio run -e {board_id} -t upload -v
```

If upload fails, check:
1. Correct USB cable (data, not charge-only)
2. Board is in bootloader mode (hold BOOT, press RST)
3. Port permissions: `sudo usermod -a -G dialout $USER` (Linux)
4. `upload_speed` in platformio.ini matches board capability"""
        )


def gen_debugging():
    """Debugging configuration questions."""
    debug_configs = [
        ("esp32dev", "esp-prog", "jtag", "esp32", "openocd"),
        ("esp32-s3-devkitc-1", "esp-builtin", "jtag", "esp32s3", "openocd"),
        ("nucleo_f103rb", "stlink", "swd", "stm32f1x", "openocd"),
        ("nucleo_f401re", "stlink", "swd", "stm32f4x", "openocd"),
        ("nucleo_h743zi", "stlink", "swd", "stm32h7x", "openocd"),
        ("bluepill_f103c8", "stlink", "swd", "stm32f1x", "openocd"),
        ("blackpill_f411ce", "cmsis-dap", "swd", "stm32f4x", "openocd"),
        ("teensy41", "jlink", "swd", "mimxrt1060", "jlink"),
    ]
    for board_id, tool, iface, target, server in debug_configs:
        info = BOARDS[board_id]
        emit(
            f"How do I set up debugging for {info['mcu']} ({board_id}) with {tool} in PlatformIO?",
            f"""Configure debugging in platformio.ini:

```ini
[env:{board_id}]
platform = {_platform_for(board_id)}
board = {board_id}
framework = {random.choice(info['framework'])}
debug_tool = {tool}
debug_init_break = tbreak setup
debug_speed = 5000
build_type = debug
```

Start debugging:
```bash
# Start GDB debug session
pio debug -e {board_id}

# Start and break at setup()
pio debug -e {board_id} --interface=gdb -- -ex "break setup" -ex "continue"
```

In VS Code, press F5 to start debugging with the PlatformIO debugger.

Hardware connections for {tool}:
- {"JTAG: TDI, TDO, TMS, TCK, GND" if iface == "jtag" else "SWD: SWDIO, SWCLK, GND (optional: SWO for trace)"}
- Ensure target voltage matches probe voltage (3.3V)
- {"ESP32-S3 has built-in USB JTAG — no external probe needed" if tool == "esp-builtin" else f"Connect {tool} probe to the {iface.upper()} pins"}"""
        )


def gen_build_flags():
    """Build flags and advanced configuration."""
    for _ in range(60):
        board_id = random.choice(list(BOARDS.keys()))
        info = BOARDS[board_id]
        flags = random.sample(BUILD_FLAGS, k=random.randint(2, 4))
        fw = random.choice(info["framework"])

        flags_str = "\n    ".join(flags)
        emit(
            f"What do these PlatformIO build_flags do for {info['mcu']}: {', '.join(flags[:2])}?",
            f"""These build flags modify compilation for {info['mcu']}:

```ini
[env:{board_id}]
platform = {_platform_for(board_id)}
board = {board_id}
framework = {fw}
build_flags =
    {flags_str}
```

Explanation:
{_explain_flags(flags)}

Build flags are passed directly to GCC. Use `build_unflags` to remove default flags:
```ini
build_unflags = -Os
build_flags = -O2
```"""
        )


def gen_partition_tables():
    """ESP32 partition table configurations."""
    esp_boards = [b for b, i in BOARDS.items() if "ESP32" in i["mcu"]]
    for board_id in esp_boards:
        info = BOARDS[board_id]
        for pt_name, pt_desc in PARTITION_TABLES.items():
            if pt_name == "custom":
                emit(
                    f"How do I create a custom partition table for {info['mcu']} in PlatformIO?",
                    f"""Create a custom CSV partition table:

1. Create `partitions_custom.csv` in your project root:
```csv
# Name,   Type, SubType, Offset,  Size,    Flags
nvs,      data, nvs,     0x9000,  0x5000,
otadata,  data, ota,     0xe000,  0x2000,
app0,     app,  ota_0,   0x10000, 0x1E0000,
app1,     app,  ota_1,   0x1F0000,0x1E0000,
spiffs,   data, spiffs,  0x3D0000,0x30000,
```

2. Reference it in platformio.ini:
```ini
[env:{board_id}]
platform = espressif32
board = {board_id}
framework = arduino
board_build.partitions = partitions_custom.csv
```

Partition types: `app` (firmware), `data` (NVS, SPIFFS, FAT, OTA data).
Total size must not exceed flash size ({info['flash']})."""
                )
            else:
                emit(
                    f"How do I use the {pt_name} partition table for {info['mcu']} in PlatformIO?",
                    f"""Set the partition table in platformio.ini:

```ini
[env:{board_id}]
platform = espressif32
board = {board_id}
framework = {random.choice(info['framework'])}
board_build.partitions = {pt_name}.csv
```

The `{pt_name}` partition scheme: {pt_desc}.

{"Use this when your firmware exceeds the default 1.2MB app partition." if "huge" in pt_name or "no_ota" in pt_name else "This is suitable for most projects."}

List all built-in partition tables:
```bash
ls ~/.platformio/packages/framework-arduinoespressif32/tools/partitions/
```"""
                )


def gen_esp_idf_components():
    """ESP-IDF component and sdkconfig questions."""
    for _ in range(50):
        comps = random.sample(ESPIDF_COMPONENTS, k=random.randint(2, 4))
        board_id = random.choice([b for b, i in BOARDS.items() if "espidf" in i["framework"]])
        info = BOARDS[board_id]

        emit(
            f"How do I configure ESP-IDF components ({', '.join(comps[:3])}) in a PlatformIO project for {info['mcu']}?",
            f"""For ESP-IDF framework in PlatformIO, configure components via sdkconfig or build_flags:

```ini
[env:{board_id}]
platform = espressif32
board = {board_id}
framework = espidf
board_build.cmake_extra_args =
    -DEXTRA_COMPONENT_DIRS=components
```

Create `sdkconfig.defaults` in your project root for component configuration:
```
# {comps[0]} configuration
CONFIG_{comps[0].upper().replace('-', '_')}_ENABLED=y

# {comps[1]} configuration
CONFIG_{comps[1].upper().replace('-', '_')}_ENABLED=y
```

Or use menuconfig:
```bash
pio run -e {board_id} -t menuconfig
```

Add custom components in `components/` directory with their own `CMakeLists.txt`:
```cmake
idf_component_register(
    SRCS "my_component.c"
    INCLUDE_DIRS "include"
    REQUIRES {' '.join(comps[:3])}
)
```"""
        )


def gen_ci_cd():
    """GitHub Actions CI/CD integration."""
    ci_templates = [
        {
            "q": "How do I set up GitHub Actions CI for a PlatformIO project?",
            "boards": ["esp32dev", "nucleo_f103rb"],
        },
        {
            "q": "Create a GitHub Actions workflow that builds PlatformIO for multiple boards.",
            "boards": ["esp32dev", "esp32-s3-devkitc-1", "nanoatmega328"],
        },
        {
            "q": "How do I run PlatformIO unit tests in GitHub Actions?",
            "boards": ["native"],
        },
        {
            "q": "Set up CI/CD with PlatformIO that runs static analysis and tests.",
            "boards": ["esp32dev"],
        },
    ]

    for tpl in ci_templates:
        boards_str = ", ".join(tpl["boards"])
        board_matrix = "\n".join(f'            - {b}' for b in tpl["boards"])
        emit(
            tpl["q"],
            f"""Create `.github/workflows/platformio.yml`:

```yaml
name: PlatformIO CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment:
{board_matrix}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install PlatformIO
        run: |
          pip install platformio
          pio pkg update

      - name: Cache PlatformIO packages
        uses: actions/cache@v4
        with:
          path: ~/.platformio
          key: pio-${{{{ hashFiles('platformio.ini') }}}}

      - name: Build
        run: pio run -e ${{{{ matrix.environment }}}}

      - name: Run Tests
        if: matrix.environment == 'native'
        run: pio test -e native
```

This workflow:
1. Triggers on push/PR to main and develop
2. Builds for environments: {boards_str}
3. Caches PlatformIO packages for faster builds
4. Runs unit tests on the native environment"""
        )

    # More CI variations
    for runner in CI_RUNNERS:
        for board_id in random.sample(list(BOARDS.keys()), 3):
            info = BOARDS[board_id]
            emit(
                f"Create a minimal GitHub Actions workflow for PlatformIO {info['mcu']} on {runner}.",
                f"""```yaml
name: Build {info['mcu']}

on: [push, pull_request]

jobs:
  build:
    runs-on: {runner}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install platformio
      - run: pio run -e {board_id}
```

This is the minimal setup. For production, add caching, test steps, and artifact uploads:
```yaml
      - name: Upload firmware
        uses: actions/upload-artifact@v4
        with:
          name: firmware-{board_id}
          path: .pio/build/{board_id}/firmware.*
```"""
            )


def gen_error_fixes():
    """Common error messages and fixes."""
    for err in COMMON_ERRORS:
        emit(
            f"I'm getting this PlatformIO error: `{err['error']}`\nHow do I fix it?",
            f"""**Cause**: {err['cause']}

**Fix**: {err['fix']}

If the issue persists:
1. Update PlatformIO: `pio upgrade`
2. Clean build: `pio run -t clean`
3. Delete `.pio/` directory and rebuild
4. Check the PlatformIO Community forum for similar issues"""
        )

    # Variant questions about the same errors
    for err in COMMON_ERRORS:
        board_id = random.choice(list(BOARDS.keys()))
        info = BOARDS[board_id]
        emit(
            f"My {info['mcu']} project fails with: {err['error']}",
            f"""This error occurs on {info['mcu']} ({board_id}) because: {err['cause']}

**Solution**: {err['fix']}

For {info['mcu']} specifically, also check:
- Platform version: `pio pkg update -g -p {_platform_for(board_id)}`
- Board definition: `pio boards {board_id}`
- Framework compatibility with your code"""
        )


def gen_advanced_configs():
    """Advanced PlatformIO configurations."""
    configs = [
        {
            "q": "How do I use PlatformIO with FreeRTOS on ESP32?",
            "a": """FreeRTOS is included by default with ESP32 Arduino and ESP-IDF:

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
build_flags =
    -DCONFIG_FREERTOS_HZ=1000
    -DCONFIG_FREERTOS_TIMER_TASK_STACK_DEPTH=4096
```

Example FreeRTOS task:
```cpp
#include <Arduino.h>

void sensorTask(void *parameter) {
    for (;;) {
        float temp = readSensor();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void setup() {
    Serial.begin(115200);
    xTaskCreatePinnedToCore(
        sensorTask,    // Task function
        "SensorTask",  // Name
        4096,          // Stack size
        NULL,          // Parameters
        1,             // Priority
        NULL,          // Task handle
        0              // Core (0 or 1)
    );
}

void loop() {
    // Main loop runs on core 1
    vTaskDelay(pdMS_TO_TICKS(100));
}
```

Key settings:
- `CONFIG_FREERTOS_HZ`: tick rate (1000 = 1ms resolution)
- Pin tasks to specific cores on dual-core ESP32
- Use `vTaskDelay()` instead of `delay()` for cooperative multitasking"""
        },
        {
            "q": "How do I use PlatformIO with LittleFS on ESP32?",
            "a": """Configure LittleFS filesystem:

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
board_build.filesystem = littlefs
board_build.partitions = default.csv
```

Upload filesystem image:
```bash
# Build and upload filesystem
pio run -t uploadfs

# Or build filesystem image only
pio run -t buildfs
```

Place files in `data/` directory:
```
data/
├── config.json
├── index.html
└── style.css
```

Usage in code:
```cpp
#include <LittleFS.h>

void setup() {
    if (!LittleFS.begin(true)) {
        Serial.println("LittleFS mount failed");
        return;
    }
    File file = LittleFS.open("/config.json", "r");
    String content = file.readString();
    file.close();
}
```"""
        },
        {
            "q": "How do I configure OTA updates in PlatformIO for ESP32?",
            "a": """Set up Over-The-Air updates:

```ini
[env:esp32dev_ota]
platform = espressif32
board = esp32dev
framework = arduino
upload_protocol = espota
upload_port = 192.168.1.100
upload_flags =
    --port=3232
    --auth=mypassword
board_build.partitions = min_spiffs.csv
```

First flash via USB with OTA support:
```cpp
#include <ArduinoOTA.h>

void setup() {
    WiFi.begin("SSID", "password");
    while (WiFi.status() != WL_CONNECTED) delay(500);

    ArduinoOTA.setHostname("esp32-sensor");
    ArduinoOTA.setPassword("mypassword");
    ArduinoOTA.begin();
}

void loop() {
    ArduinoOTA.handle();
    // your code
}
```

Subsequent uploads use OTA:
```bash
pio run -e esp32dev_ota -t upload
```

Requirements:
- Partition table with OTA support (two app partitions)
- ESP32 and dev machine on same network
- Matching password in code and platformio.ini"""
        },
        {
            "q": "How do I use PlatformIO with Zephyr RTOS?",
            "a": """Configure Zephyr in PlatformIO:

```ini
[env:nucleo_f401re]
platform = ststm32
board = nucleo_f401re
framework = zephyr

; Zephyr-specific build flags
build_flags =
    -DCONFIG_SERIAL=y
    -DCONFIG_UART_CONSOLE=y
    -DCONFIG_GPIO=y
```

Zephyr uses a different project structure:
```
├── src/
│   └── main.c
├── prj.conf          # Zephyr configuration
├── CMakeLists.txt    # Zephyr CMake
└── platformio.ini
```

Create `prj.conf`:
```
CONFIG_GPIO=y
CONFIG_SERIAL=y
CONFIG_UART_CONSOLE=y
CONFIG_PRINTK=y
CONFIG_LOG=y
```

Example `src/main.c`:
```c
#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>

#define LED_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED_NODE, gpios);

int main(void) {
    gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);
    while (1) {
        gpio_pin_toggle_dt(&led);
        k_msleep(1000);
    }
    return 0;
}
```

Supported boards for Zephyr: nucleo_f103rb, nucleo_f401re, nucleo_f446re, nucleo_h743zi, nucleo_l476rg, and many more."""
        },
        {
            "q": "How do I set up a PlatformIO monorepo with shared libraries?",
            "a": """Use `lib_extra_dirs` and shared environments:

```
project-root/
├── shared-libs/
│   ├── common-utils/
│   │   ├── src/
│   │   │   └── utils.cpp
│   │   └── include/
│   │       └── utils.h
│   └── sensor-driver/
│       ├── src/
│       │   └── sensor.cpp
│       └── include/
│           └── sensor.h
├── firmware-a/
│   ├── src/
│   │   └── main.cpp
│   └── platformio.ini
└── firmware-b/
    ├── src/
    │   └── main.cpp
    └── platformio.ini
```

In each project's platformio.ini:
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
lib_extra_dirs =
    ../shared-libs
lib_deps =
    common-utils
    sensor-driver
```

For a single platformio.ini managing multiple firmwares:
```ini
[platformio]
src_dir = firmware-a/src
lib_extra_dirs = shared-libs

[env:firmware_a]
platform = espressif32
board = esp32dev
framework = arduino
src_filter = +<*> -<../firmware-b/*>

[env:firmware_b]
platform = espressif32
board = esp32-s3-devkitc-1
framework = arduino
src_dir = firmware-b/src
```"""
        },
        {
            "q": "How do I use PlatformIO with STM32CubeMX?",
            "a": """Integrate STM32CubeMX-generated code with PlatformIO:

```ini
[env:nucleo_f401re]
platform = ststm32
board = nucleo_f401re
framework = stm32cube

; Include CubeMX-generated HAL config
board_build.stm32cube.custom_config_header = yes
```

Steps:
1. Create project in STM32CubeMX, generate code
2. Copy generated files to PlatformIO project:
```
src/
├── main.c              # From CubeMX
├── stm32f4xx_it.c      # Interrupt handlers
├── stm32f4xx_hal_msp.c # HAL MSP init
└── system_stm32f4xx.c  # System init
include/
├── main.h
├── stm32f4xx_hal_conf.h
└── stm32f4xx_it.h
```

3. Build: `pio run -e nucleo_f401re`

For mixed Arduino + STM32Cube:
```ini
[env:nucleo_f401re_arduino]
platform = ststm32
board = nucleo_f401re
framework = arduino
build_flags =
    -DHAL_UART_MODULE_ENABLED
    -DHAL_I2C_MODULE_ENABLED
```

This lets you use Arduino libraries while accessing STM32 HAL when needed."""
        },
    ]
    for c in configs:
        emit(c["q"], c["a"])


def gen_board_comparison():
    """Board comparison and selection questions."""
    comparisons = [
        (["esp32dev", "esp32-s3-devkitc-1", "esp32-c3-devkitm-1"], "Which ESP32 variant should I choose for my IoT project?"),
        (["nanoatmega328", "uno", "pico"], "Compare Arduino Nano, Uno, and Raspberry Pi Pico for beginners."),
        (["nucleo_f103rb", "bluepill_f103c8"], "What's the difference between Nucleo F103RB and Blue Pill in PlatformIO?"),
        (["teensy41", "esp32-s3-devkitc-1", "nucleo_h743zi"], "Compare Teensy 4.1, ESP32-S3, and STM32H743 for high-performance projects."),
        (["esp32dev", "d1_mini"], "Should I use ESP32 or ESP8266 for my WiFi project?"),
        (["pico", "nucleo_f401re"], "Compare RP2040 Pico vs STM32F401 Nucleo for motor control."),
        (["esp32-c6-devkitc-1", "esp32-h2-devkitm-1"], "Compare ESP32-C6 and ESP32-H2 for Thread/Zigbee projects."),
    ]
    for board_ids, question in comparisons:
        rows = []
        for bid in board_ids:
            info = BOARDS[bid]
            rows.append(f"| {bid} | {info['mcu']} | {info['f_cpu']} | {info['flash']} | {info['ram']} | {', '.join(info['framework'])} |")
        table = "\n".join(rows)
        emit(
            question,
            f"""Here's a comparison of the boards in PlatformIO:

| Board ID | MCU | CPU Speed | Flash | RAM | Frameworks |
|----------|-----|-----------|-------|-----|------------|
{table}

All are configured in platformio.ini:
```ini
[env:{board_ids[0]}]
platform = {_platform_for(board_ids[0])}
board = {board_ids[0]}
framework = {random.choice(BOARDS[board_ids[0]]['framework'])}
```

Key considerations:
- **Processing power**: Higher MHz = faster computation
- **Memory**: More RAM = more concurrent tasks, larger buffers
- **Flash**: More flash = larger firmware, more filesystem space
- **Framework support**: Arduino is easiest, ESP-IDF/Zephyr give more control
- **Peripherals**: Check specific peripheral support (ADC channels, timers, DMA) in the datasheet"""
        )


def gen_lib_compat_modes():
    """Library compatibility modes."""
    modes = [
        ("off", "Disables compatibility checks. Use when you know the library works."),
        ("soft", "Default. Warns about potential incompatibilities but builds anyway."),
        ("strict", "Rejects libraries that don't declare compatibility with your board/framework."),
    ]
    for mode, desc in modes:
        emit(
            f"What does `lib_compat_mode = {mode}` do in PlatformIO?",
            f"""`lib_compat_mode = {mode}`: {desc}

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
lib_compat_mode = {mode}
lib_deps =
    bblanchon/ArduinoJson@^7.3.0
```

Use cases:
- `off`: Porting libraries from another platform; you handle compatibility yourself
- `soft`: Normal development; get warnings but don't block builds
- `strict`: Production CI; catch incompatible libraries early

If a library doesn't build with `strict`, check if it declares compatible platforms in `library.json`:
```json
{{
    "platforms": ["espressif32", "espressif8266", "atmelavr"]
}}
```"""
        )


def gen_filesystem_configs():
    """Filesystem configuration (SPIFFS, LittleFS, FAT)."""
    fs_types = [
        ("littlefs", "LittleFS", "Wear-leveled, power-loss safe, recommended for new projects"),
        ("spiffs", "SPIFFS", "Legacy, no directories, being phased out"),
        ("fatfs", "FAT", "Compatible with PC, good for SD cards"),
    ]
    for fs_id, fs_name, fs_desc in fs_types:
        board_id = random.choice([b for b, i in BOARDS.items() if "ESP32" in i["mcu"]])
        info = BOARDS[board_id]
        emit(
            f"How do I use {fs_name} filesystem with {info['mcu']} in PlatformIO?",
            f"""{fs_name}: {fs_desc}

```ini
[env:{board_id}]
platform = espressif32
board = {board_id}
framework = arduino
board_build.filesystem = {fs_id}
```

Upload files from `data/` directory:
```bash
pio run -t uploadfs
```

Usage in code:
```cpp
#include <{"LittleFS" if fs_id == "littlefs" else fs_name}.h>

void setup() {{
    if (!{"LittleFS" if fs_id == "littlefs" else fs_name}.begin(true)) {{
        Serial.println("{fs_name} mount failed");
        return;
    }}
    // List files
    File root = {"LittleFS" if fs_id == "littlefs" else fs_name}.open("/");
    File file = root.openNextFile();
    while (file) {{
        Serial.printf("File: %s, Size: %d\\n", file.name(), file.size());
        file = root.openNextFile();
    }}
}}
```"""
        )


def gen_monitor_filters():
    """Serial monitor filter configurations."""
    filters = [
        ("esp32_exception_decoder", "Decodes ESP32 crash backtraces into source file and line numbers"),
        ("time", "Adds timestamps to each line of serial output"),
        ("colorize", "Adds ANSI colors based on log level (E/W/I/D/V)"),
        ("log2file", "Saves serial output to a log file"),
        ("hexlify", "Shows raw bytes as hexadecimal"),
        ("direct", "Disables line buffering for real-time output"),
        ("send_on_enter", "Sends data only when Enter is pressed"),
    ]
    for f_name, f_desc in filters:
        emit(
            f"How do I use the `{f_name}` serial monitor filter in PlatformIO?",
            f"""The `{f_name}` filter: {f_desc}

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
monitor_filters = {f_name}
```

Combine multiple filters:
```ini
monitor_filters = {f_name}, time
```

Use from CLI:
```bash
pio device monitor -f {f_name}
```

{"This is essential for debugging ESP32 crashes — it converts raw backtrace addresses to readable function names and line numbers." if f_name == "esp32_exception_decoder" else f"Use `{f_name}` to {f_desc.lower()}."}"""
        )


def gen_extra_questions():
    """Additional miscellaneous Q&A pairs to reach target count."""
    extras = [
        {
            "q": "How do I speed up PlatformIO compilation?",
            "a": """Several strategies to speed up PlatformIO builds:

1. **Parallel compilation** (default, but verify):
```ini
[env:esp32dev]
; Use all CPU cores
build_flags = -j8
```

2. **ccache** (Linux/macOS):
```bash
# Install ccache
brew install ccache  # macOS
sudo apt install ccache  # Linux

# Enable in platformio.ini
[env:esp32dev]
custom_cache_dir = /tmp/pio-cache
```

3. **Selective library inclusion**:
```ini
lib_deps =
    bblanchon/ArduinoJson@^7.3.0
lib_ldf_mode = chain+   ; Only include actually used libraries
```

4. **Avoid full rebuilds**:
```bash
# Only rebuild changed files (default behavior)
pio run

# Force full rebuild only when needed
pio run -t clean && pio run
```

5. **SSD for .platformio**: Move `~/.platformio` to an SSD if on HDD."""
        },
        {
            "q": "How do I use custom board definitions in PlatformIO?",
            "a": """Create a custom board JSON file:

1. Create `boards/` directory in your project root
2. Create `boards/my_custom_board.json`:
```json
{
    "build": {
        "core": "esp32",
        "extra_flags": "-DARDUINO_CUSTOM_BOARD",
        "f_cpu": "240000000L",
        "f_flash": "80000000L",
        "flash_mode": "qio",
        "mcu": "esp32s3",
        "variant": "esp32s3"
    },
    "connectivity": ["wifi", "bluetooth"],
    "debug": {
        "openocd_target": "esp32s3.cfg"
    },
    "frameworks": ["arduino", "espidf"],
    "name": "My Custom ESP32-S3 Board",
    "upload": {
        "flash_size": "16MB",
        "maximum_ram_size": 327680,
        "maximum_size": 16777216,
        "require_upload_port": true,
        "speed": 460800
    },
    "url": "https://example.com",
    "vendor": "Custom"
}
```

3. Use in platformio.ini:
```ini
[env:my_custom_board]
platform = espressif32
board = my_custom_board
framework = arduino
```

PlatformIO automatically scans the `boards/` directory."""
        },
        {
            "q": "How do I use PlatformIO with Docker for reproducible builds?",
            "a": """Use the official PlatformIO Docker image:

```dockerfile
FROM python:3.12-slim

RUN pip install platformio

WORKDIR /project
COPY . .

RUN pio run -e esp32dev
```

Or use the `platformio/platformio-ci` image:
```yaml
# docker-compose.yml
services:
  build:
    image: python:3.12-slim
    volumes:
      - .:/project
      - pio-cache:/root/.platformio
    working_dir: /project
    command: >
      bash -c "pip install platformio &&
               pio run -e esp32dev"

volumes:
  pio-cache:
```

For GitHub Actions with Docker:
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: python:3.12-slim
    steps:
      - uses: actions/checkout@v4
      - run: pip install platformio
      - run: pio run
```

Cache `~/.platformio` between builds for speed."""
        },
        {
            "q": "How do I use pre/post build scripts in PlatformIO?",
            "a": """PlatformIO supports custom Python scripts for build hooks:

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
extra_scripts =
    pre:scripts/pre_build.py
    post:scripts/post_build.py
```

Pre-build script (`scripts/pre_build.py`):
```python
Import("env")

# Add build flags dynamically
import datetime
build_time = datetime.datetime.now().isoformat()
env.Append(CPPDEFINES=[
    ("BUILD_TIME", f'\\"{build_time}\\"'),
    ("GIT_REV", f'\\"{env.Execute("git rev-parse --short HEAD")}\\"'),
])

# Custom target
def before_build(source, target, env):
    print("Running pre-build checks...")

env.AddPreAction("buildprog", before_build)
```

Post-build script (`scripts/post_build.py`):
```python
Import("env")
import shutil

def after_build(source, target, env):
    firmware = str(target[0])
    shutil.copy(firmware, "output/firmware.bin")
    print(f"Firmware copied to output/")

env.AddPostAction("buildprog", after_build)
```"""
        },
        {
            "q": "What is the difference between `lib_deps` and `lib_extra_dirs` in PlatformIO?",
            "a": """`lib_deps` and `lib_extra_dirs` serve different purposes:

**`lib_deps`** — Downloads and manages external libraries:
```ini
lib_deps =
    bblanchon/ArduinoJson@^7.3.0       ; From PlatformIO Registry
    https://github.com/user/lib.git     ; From Git
    https://example.com/lib.zip         ; From URL
    file:///path/to/local/lib           ; From local path
    symlink:///path/to/local/lib        ; Symlink to local
```

**`lib_extra_dirs`** — Points to directories containing local libraries:
```ini
lib_extra_dirs =
    ../shared-libs
    ~/my-libraries
```

Key differences:
| Feature | lib_deps | lib_extra_dirs |
|---------|----------|----------------|
| Version management | Yes (semver) | No |
| Auto-download | Yes | No |
| Stored in | .pio/libdeps/ | Original location |
| Lock file | Yes (package-lock.json) | No |
| Git/URL sources | Yes | No |

Best practice: Use `lib_deps` for third-party libraries, `lib_extra_dirs` for your own shared code across projects."""
        },
        {
            "q": "How do I use PlatformIO with VS Code?",
            "a": """Install and configure PlatformIO IDE for VS Code:

1. **Install**: Search "PlatformIO IDE" in VS Code Extensions and install it.

2. **Create project**: Click the PlatformIO icon in the sidebar → "New Project" → Select board and framework.

3. **Key shortcuts**:
   - `Ctrl+Alt+B` / `Cmd+Alt+B` — Build
   - `Ctrl+Alt+U` / `Cmd+Alt+U` — Upload
   - `Ctrl+Alt+S` / `Cmd+Alt+S` — Serial Monitor
   - `Ctrl+Alt+T` / `Cmd+Alt+T` — Run Tests

4. **IntelliSense**: PlatformIO auto-generates `c_cpp_properties.json`:
```json
{
    "configurations": [{
        "name": "PlatformIO",
        "includePath": [
            "${workspaceFolder}/include",
            "${workspaceFolder}/src",
            "${HOME}/.platformio/packages/framework-arduinoespressif32/**"
        ],
        "defines": ["ARDUINO=10812", "ESP32=1"]
    }]
}
```

5. **Tasks**: PlatformIO registers as a VS Code Task Provider. Access via `Terminal → Run Task → PlatformIO`.

6. **Debugging**: Press F5 to start debugging (requires debug probe configured in platformio.ini).

7. **Status bar**: Shows current environment, port, and quick-access buttons."""
        },
        {
            "q": "How do I use build_src_filter in PlatformIO to include or exclude source files?",
            "a": """Use `build_src_filter` to control which files are compiled:

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino

; Include all, exclude tests and examples
build_src_filter =
    +<*>
    -<test/>
    -<examples/>

; Or include only specific files
build_src_filter =
    +<main.cpp>
    +<drivers/>
    -<drivers/deprecated/>
```

Filter syntax:
- `+<pattern>` — Include matching files
- `-<pattern>` — Exclude matching files
- Filters are applied in order (last match wins)
- Paths are relative to `src_dir`

Common patterns:
```ini
; Platform-specific code
[env:esp32dev]
build_src_filter =
    +<*>
    -<hal/stm32/>
    +<hal/esp32/>

[env:nucleo_f401re]
build_src_filter =
    +<*>
    -<hal/esp32/>
    +<hal/stm32/>
```

This is useful for:
- Cross-platform projects with platform-specific HAL code
- Excluding test or example files from production builds
- Conditional compilation without `#ifdef`"""
        },
    ]
    for e in extras:
        emit(e["q"], e["a"])

    # Generate parametric variations for common tasks
    tasks = [
        "blink an LED",
        "read an analog sensor",
        "send data over WiFi",
        "communicate via I2C",
        "use SPI with a display",
        "set up a web server",
        "read temperature from DS18B20",
        "control a servo motor",
        "send MQTT messages",
        "log data to SD card",
        "use deep sleep mode",
        "read a rotary encoder",
        "drive a stepper motor",
        "use CAN bus communication",
        "implement a PID controller",
        "use BLE to communicate with a phone",
        "read GPS data via UART",
        "drive WS2812B LED strip",
        "use DMA for fast ADC reads",
        "implement watchdog timer",
    ]
    for task in tasks:
        for board_id in random.sample(list(BOARDS.keys()), 3):
            info = BOARDS[board_id]
            fw = random.choice(info["framework"])
            emit(
                f"Give me a PlatformIO project setup to {task} on {info['mcu']} ({board_id}).",
                f"""platformio.ini for {task} on {info['mcu']}:

```ini
[env:{board_id}]
platform = {_platform_for(board_id)}
board = {board_id}
framework = {fw}
monitor_speed = 115200
{_libs_for_task(task)}
```

Build and upload:
```bash
pio run -e {board_id} -t upload
pio device monitor -b 115200
```

{info['mcu']} specs: {info['f_cpu']} CPU, {info['flash']} flash, {info['ram']} RAM.
{"Note: " + fw + " provides a simplified API. " if fw == "arduino" else "Using " + fw + " gives you full SDK access. "}
{"This board supports dual-core — pin time-critical code to core 0." if "ESP32" in info['mcu'] and "C3" not in info['mcu'] and "H2" not in info['mcu'] else ""}"""
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _platform_for(board_id: str) -> str:
    for plat, boards in PLATFORMS:
        if board_id in boards:
            return plat
    if "esp32" in board_id or "seeed_xiao_esp32" in board_id:
        return "espressif32"
    if "nucleo" in board_id or "bluepill" in board_id or "blackpill" in board_id:
        return "ststm32"
    if "teensy" in board_id:
        return "teensy"
    if "pico" in board_id or "rpipico" in board_id:
        return "raspberrypi"
    if "d1_mini" in board_id or "nodemcu" in board_id:
        return "espressif8266"
    return "atmelavr"


def _explain_flags(flags: list[str]) -> str:
    explanations = {
        "-DCORE_DEBUG_LEVEL=5": "- `-DCORE_DEBUG_LEVEL=5`: Enables verbose debug logging on ESP32",
        "-DCONFIG_ASYNC_TCP_RUNNING_CORE=1": "- `-DCONFIG_ASYNC_TCP_RUNNING_CORE=1`: Runs AsyncTCP on core 1",
        "-DBOARD_HAS_PSRAM": "- `-DBOARD_HAS_PSRAM`: Enables PSRAM support for extra memory",
        "-mfix-esp32-psram-cache-issue": "- `-mfix-esp32-psram-cache-issue`: Workaround for ESP32 rev1 PSRAM cache bug",
        "-DARDUINO_USB_MODE=1": "- `-DARDUINO_USB_MODE=1`: Uses native USB instead of USB-JTAG",
        "-DARDUINO_USB_CDC_ON_BOOT=1": "- `-DARDUINO_USB_CDC_ON_BOOT=1`: Maps Serial to USB CDC (ESP32-S3/C3)",
        "-DCONFIG_BT_NIMBLE_ROLE_CENTRAL_DISABLED": "- `-DCONFIG_BT_NIMBLE_ROLE_CENTRAL_DISABLED`: Saves flash by disabling BLE central role",
        "-DCONFIG_BT_NIMBLE_ROLE_OBSERVER_DISABLED": "- `-DCONFIG_BT_NIMBLE_ROLE_OBSERVER_DISABLED`: Saves flash by disabling BLE observer role",
        "-DCONFIG_SPIRAM_CACHE_WORKAROUND": "- `-DCONFIG_SPIRAM_CACHE_WORKAROUND`: PSRAM cache workaround for ESP32",
        "-Os": "- `-Os`: Optimize for size (smaller binary, slightly slower)",
        "-O2": "- `-O2`: Optimize for speed (moderate, good balance)",
        "-O3": "- `-O3`: Aggressive speed optimization (may increase binary size)",
        "-Wall -Wextra": "- `-Wall -Wextra`: Enable all common compiler warnings",
        "-std=gnu++17": "- `-std=gnu++17`: Use C++17 standard with GNU extensions",
        "-fno-rtti": "- `-fno-rtti`: Disable RTTI, saves ~2-5KB flash",
        "-fno-exceptions": "- `-fno-exceptions`: Disable C++ exceptions, saves ~10-20KB flash",
        '-DFIRMWARE_VERSION=\\"1.0.0\\"': '- `-DFIRMWARE_VERSION`: Define firmware version string accessible in code',
        "-DCONFIG_FREERTOS_HZ=1000": "- `-DCONFIG_FREERTOS_HZ=1000`: Set FreeRTOS tick rate to 1kHz (1ms resolution)",
        "-DCONFIG_ESP32_WIFI_TX_BUFFER_TYPE=0": "- `-DCONFIG_ESP32_WIFI_TX_BUFFER_TYPE=0`: Use static WiFi TX buffers (more deterministic)",
    }
    result = []
    for f in flags:
        result.append(explanations.get(f, f"- `{f}`: Custom build flag"))
    return "\n".join(result)


def _libs_for_task(task: str) -> str:
    task_libs = {
        "send data over WiFi": 'lib_deps =\n    bblanchon/ArduinoJson@^7.3.0',
        "communicate via I2C": "",
        "use SPI with a display": 'lib_deps =\n    adafruit/Adafruit SSD1306@^2.5.12\n    adafruit/Adafruit GFX Library@^1.11.11',
        "set up a web server": 'lib_deps =\n    me-no-dev/ESPAsyncWebServer@^1.2.7\n    me-no-dev/AsyncTCP@^1.1.4',
        "read temperature from DS18B20": 'lib_deps =\n    paulstoffregen/OneWire@^2.3.8\n    milesburton/DallasTemperature@^4.0.4',
        "send MQTT messages": 'lib_deps =\n    knolleary/PubSubClient@^2.8',
        "drive WS2812B LED strip": 'lib_deps =\n    fastled/FastLED@^3.7.8',
        "use BLE to communicate with a phone": 'lib_deps =\n    h2zero/NimBLE-Arduino@^2.1.2',
        "control a servo motor": 'lib_deps =\n    madhephaestus/ESP32Servo@^3.0.6',
    }
    return task_libs.get(task, "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gen_basic_ini()
    gen_multi_env()
    gen_lib_deps()
    gen_cli_commands()
    gen_debugging()
    gen_build_flags()
    gen_partition_tables()
    gen_esp_idf_components()
    gen_ci_cd()
    gen_error_fixes()
    gen_advanced_configs()
    gen_board_comparison()
    gen_lib_compat_modes()
    gen_filesystem_configs()
    gen_monitor_filters()
    gen_extra_questions()


if __name__ == "__main__":
    main()
