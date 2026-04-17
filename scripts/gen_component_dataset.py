#!/usr/bin/env python3
"""Generate rich electronic components Q&A dataset from structured data sources.

Generates training pairs across 5 categories:
1. Component specs Q&A (pinout, voltage, packages, features)
2. Component selection Q&A (use-case driven recommendations)
3. Cross-reference Q&A (equivalents, pin-compatibility)
4. BOM/sourcing Q&A (LCSC, JLCPCB, distributors)
5. Datasheet reading Q&A (how to interpret specs)

Also parses JITX open-components-database Stanza files for real pinout data.

Output: JSONL to stdout — pipe to file.

Usage::

    uv run python scripts/gen_component_dataset.py > data/components/train.jsonl
    uv run python scripts/gen_component_dataset.py --categories specs,selection --stats
    uv run python scripts/gen_component_dataset.py --jitx-path /tmp/jitx-odb/ --stats
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Component:
    """A component with known specs for Q&A generation."""
    name: str
    category: str
    manufacturer: str
    description: str
    specs: dict[str, str]
    pins: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    lcsc: str = ""
    notes: str = ""


def msg(user: str, assistant: str) -> dict:
    """Format a single Q&A pair as a chat messages dict."""
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


# ---------------------------------------------------------------------------
# Component database — ~200 components with real specs
# ---------------------------------------------------------------------------

COMPONENTS: list[Component] = [
    # ===== MCUs =====
    Component(
        name="STM32F103C8T6",
        category="mcu",
        manufacturer="STMicroelectronics",
        description="ARM Cortex-M3 MCU, 72 MHz, 64 KB Flash, 20 KB SRAM",
        specs={
            "core": "ARM Cortex-M3",
            "max_frequency": "72 MHz",
            "flash": "64 KB",
            "sram": "20 KB",
            "supply_voltage": "2.0V to 3.6V",
            "gpio_count": "37",
            "adc_channels": "10 (12-bit)",
            "timers": "4 (16-bit) + 1 (PWM advanced)",
            "uart": "3",
            "spi": "2",
            "i2c": "2",
            "usb": "USB 2.0 Full-speed",
            "can": "1",
            "operating_temp": "-40 to 85 C",
            "package": "LQFP-48",
        },
        packages=["LQFP-48", "LQFP-64 (STM32F103RBT6)", "QFN-36 (STM32F103T8U6)"],
        interfaces=["SPI", "I2C", "UART", "USB", "CAN"],
        alternatives=["GD32F103C8T6", "APM32F103C8T6", "CH32F103C8T6", "AT32F403ACGT7"],
        lcsc="C8734",
    ),
    Component(
        name="STM32H743VIT6",
        category="mcu",
        manufacturer="STMicroelectronics",
        description="ARM Cortex-M7 MCU, 480 MHz, 2 MB Flash, 1 MB SRAM, FPU+DSP",
        specs={
            "core": "ARM Cortex-M7 with double-precision FPU",
            "max_frequency": "480 MHz",
            "flash": "2 MB (dual-bank)",
            "sram": "1 MB (TCM + AXI + SRAM1-4)",
            "supply_voltage": "1.62V to 3.6V",
            "gpio_count": "82",
            "adc_channels": "3x 16-bit ADC, up to 36 channels",
            "dac": "2x 12-bit DAC",
            "timers": "22 timers (2x 32-bit, 10x 16-bit, 2x watchdog)",
            "uart": "8 (4 USART + 4 UART)",
            "spi": "6",
            "i2c": "4",
            "usb": "USB 2.0 HS OTG with PHY",
            "ethernet": "10/100 Ethernet MAC with IEEE 1588v2",
            "can": "2x CAN FD",
            "sdmmc": "2",
            "operating_temp": "-40 to 85 C",
            "package": "LQFP-100",
        },
        packages=["LQFP-100", "LQFP-144", "LQFP-176", "TFBGA-240"],
        interfaces=["SPI", "I2C", "UART", "USB HS", "Ethernet", "CAN FD", "SDMMC", "QSPI", "DCMI", "SAI", "HDMI-CEC"],
        alternatives=["STM32H750VBT6 (less flash, cheaper)", "STM32F767VIT6", "i.MXRT1062"],
        lcsc="C78977",
    ),
    Component(
        name="ESP32-S3-WROOM-1-N16R8",
        category="mcu",
        manufacturer="Espressif",
        description="Dual-core Xtensa LX7, 240 MHz, Wi-Fi + BLE 5.0, 16 MB Flash, 8 MB PSRAM",
        specs={
            "core": "Dual-core Xtensa LX7",
            "max_frequency": "240 MHz",
            "flash": "16 MB (external quad SPI)",
            "psram": "8 MB (octal SPI)",
            "sram": "512 KB",
            "supply_voltage": "3.0V to 3.6V",
            "gpio_count": "36",
            "adc_channels": "20 (12-bit SAR ADC)",
            "dac": "None (removed in S3)",
            "wifi": "Wi-Fi 802.11 b/g/n, 2.4 GHz",
            "bluetooth": "Bluetooth 5.0 LE",
            "usb": "USB OTG 1.1",
            "uart": "3",
            "spi": "4 (GPSPI)",
            "i2c": "2",
            "i2s": "2",
            "lcd_interface": "8/16-bit parallel, SPI",
            "camera_interface": "DVP 8/16-bit",
            "operating_temp": "-40 to 85 C",
            "package": "Module (18x25.5 mm)",
            "ai_acceleration": "Vector instructions for ML (ESP-NN)",
        },
        packages=["Module (18x25.5 mm)", "N4, N8, N16 flash variants", "R2, R8 PSRAM variants"],
        interfaces=["SPI", "I2C", "UART", "USB OTG", "I2S", "Wi-Fi", "BLE 5.0", "DVP Camera", "LCD parallel"],
        alternatives=["ESP32-S3-MINI-1 (smaller module)", "ESP32-C6 (Wi-Fi 6, RISC-V)", "ESP32-C3 (single-core RISC-V, cheaper)"],
        lcsc="C2913202",
    ),
    Component(
        name="ATmega328P-AU",
        category="mcu",
        manufacturer="Microchip",
        description="8-bit AVR MCU, 20 MHz, 32 KB Flash, 2 KB SRAM — the Arduino Uno MCU",
        specs={
            "core": "8-bit AVR",
            "max_frequency": "20 MHz (16 MHz typical with crystal)",
            "flash": "32 KB (0.5 KB bootloader)",
            "sram": "2 KB",
            "eeprom": "1 KB",
            "supply_voltage": "1.8V to 5.5V",
            "gpio_count": "23",
            "adc_channels": "6 (10-bit)",
            "timers": "3 (2x 8-bit, 1x 16-bit)",
            "uart": "1",
            "spi": "1",
            "i2c": "1 (TWI)",
            "pwm_channels": "6",
            "operating_temp": "-40 to 85 C",
            "package": "TQFP-32",
        },
        packages=["TQFP-32", "QFN-32 (MLF)", "DIP-28 (PDIP)"],
        interfaces=["SPI", "I2C/TWI", "UART"],
        alternatives=["ATmega328PB (more peripherals)", "ATmega4809", "STM32C011 (Cortex-M0+, similar price)"],
        lcsc="C14877",
    ),
    Component(
        name="RP2040",
        category="mcu",
        manufacturer="Raspberry Pi",
        description="Dual-core ARM Cortex-M0+, 133 MHz, 264 KB SRAM, no built-in flash",
        specs={
            "core": "Dual-core ARM Cortex-M0+",
            "max_frequency": "133 MHz (overclockable to ~250 MHz)",
            "flash": "None (external QSPI, typically 2-16 MB)",
            "sram": "264 KB (6 banks)",
            "supply_voltage": "1.8V to 3.3V (on-chip regulator from up to 5.5V via VREG_VIN)",
            "gpio_count": "30",
            "adc_channels": "4 (12-bit, 500 ksps)",
            "timers": "1 (with 8 alarm comparators)",
            "uart": "2",
            "spi": "2",
            "i2c": "2",
            "pwm_channels": "16 (8 slices x 2)",
            "pio": "2 PIO blocks (8 state machines) — programmable I/O",
            "usb": "USB 1.1 Host/Device",
            "operating_temp": "-20 to 85 C",
            "package": "QFN-56 (7x7 mm)",
        },
        packages=["QFN-56 (7x7 mm)"],
        interfaces=["SPI", "I2C", "UART", "USB 1.1", "PIO"],
        alternatives=["RP2350 (ARM+RISC-V, security)", "ESP32-C3 (has Wi-Fi)", "STM32G030 (cheaper, less GPIO)"],
        lcsc="C2040",
        notes="Unique PIO peripheral enables custom protocols (WS2812, VGA, SD card in software).",
    ),
    Component(
        name="nRF52840",
        category="mcu",
        manufacturer="Nordic Semiconductor",
        description="ARM Cortex-M4F, 64 MHz, BLE 5.3 + Thread + Zigbee + 802.15.4, 1 MB Flash, 256 KB SRAM",
        specs={
            "core": "ARM Cortex-M4F (FPU)",
            "max_frequency": "64 MHz",
            "flash": "1 MB",
            "sram": "256 KB",
            "supply_voltage": "1.7V to 5.5V (built-in DC-DC and LDO)",
            "gpio_count": "48",
            "adc_channels": "8 (12-bit, 200 ksps)",
            "bluetooth": "Bluetooth 5.3 LE (2 Mbps, Long Range, Advertising Extensions)",
            "802.15.4": "Thread, Zigbee",
            "nfc": "NFC-A tag",
            "usb": "USB 2.0 Full-speed",
            "uart": "2 (UARTE)",
            "spi": "3 (SPIM)",
            "i2c": "2 (TWIM)",
            "i2s": "1",
            "qspi": "1",
            "pwm_channels": "4 (4 channels each)",
            "crypto": "AES-128/256, SHA-256, ECC P-256 in hardware",
            "operating_temp": "-40 to 85 C",
            "package": "QFN-73 (7x7 mm, aQFN)",
        },
        packages=["QFN-73 (aQFN)", "WLCSP-94"],
        interfaces=["SPI", "I2C", "UART", "USB", "I2S", "QSPI", "BLE 5.3", "Thread", "Zigbee", "NFC"],
        alternatives=["nRF5340 (dual-core, more processing)", "ESP32-C6 (Wi-Fi 6 + BLE 5)", "STM32WB55 (BLE 5.4 + 802.15.4)"],
        lcsc="C190794",
    ),
    Component(
        name="STM32C011F4P6",
        category="mcu",
        manufacturer="STMicroelectronics",
        description="ARM Cortex-M0+, 48 MHz, 16 KB Flash — ultra-low-cost STM32",
        specs={
            "core": "ARM Cortex-M0+",
            "max_frequency": "48 MHz",
            "flash": "16 KB",
            "sram": "6 KB",
            "supply_voltage": "2.0V to 3.6V",
            "gpio_count": "15",
            "adc_channels": "8 (12-bit)",
            "timers": "5",
            "uart": "2 (USART)",
            "spi": "1",
            "i2c": "1",
            "operating_temp": "-40 to 85 C",
            "package": "TSSOP-20",
        },
        packages=["TSSOP-20", "SO-8 (STM32C011J4M6)"],
        interfaces=["SPI", "I2C", "UART"],
        alternatives=["STM32G030F6P6", "ATtiny1616", "PY32F002A"],
        lcsc="C5237931",
        notes="Cheapest STM32 family. The SO-8 variant fits in an 8-pin package.",
    ),
    Component(
        name="CH32V003F4U6",
        category="mcu",
        manufacturer="WCH",
        description="RISC-V MCU, 48 MHz, 16 KB Flash, 2 KB SRAM — $0.10 MCU",
        specs={
            "core": "RISC-V (RV32EC, QingKe V2A)",
            "max_frequency": "48 MHz",
            "flash": "16 KB",
            "sram": "2 KB",
            "supply_voltage": "3.3V or 5V",
            "gpio_count": "18",
            "adc_channels": "8 (10-bit)",
            "timers": "2 (16-bit) + 1 (advanced PWM)",
            "uart": "1",
            "spi": "1",
            "i2c": "1",
            "operating_temp": "-40 to 85 C",
            "package": "QFN-20 (3x3 mm)",
        },
        packages=["QFN-20", "TSSOP-20", "SOP-16", "SOP-8"],
        interfaces=["SPI", "I2C", "UART", "1-wire debug (SWIO)"],
        alternatives=["PY32F002A", "STM32C011", "ATtiny412"],
        lcsc="C5299908",
        notes="Ultra-cheap RISC-V MCU. Single-wire debug (not SWD/JTAG). WCH-LinkE programmer needed.",
    ),

    # ===== Voltage Regulators =====
    Component(
        name="LM7805",
        category="regulator",
        manufacturer="Texas Instruments / ON Semi / various",
        description="5V linear voltage regulator, 1.5A, TO-220",
        specs={
            "output_voltage": "5.0V (fixed)",
            "input_voltage": "7V to 35V",
            "dropout_voltage": "~2V typical",
            "max_output_current": "1.5A",
            "quiescent_current": "~5 mA",
            "line_regulation": "3 mV typical",
            "load_regulation": "15 mV typical",
            "thermal_shutdown": "Yes",
            "operating_temp": "-40 to 125 C",
            "package": "TO-220-3",
            "efficiency_note": "Linear regulator — significant heat at high Vin-Vout or high current",
        },
        packages=["TO-220-3", "D2PAK", "TO-263"],
        alternatives=["AMS1117-5.0 (LDO, lower dropout)", "TPS54331 (switching, much more efficient)", "LM2596-5.0 (switching, simple)"],
        lcsc="C347377",
    ),
    Component(
        name="AMS1117-3.3",
        category="regulator",
        manufacturer="Advanced Monolithic Systems",
        description="3.3V LDO regulator, 1A, SOT-223",
        specs={
            "output_voltage": "3.3V (fixed)",
            "input_voltage": "4.5V to 12V",
            "dropout_voltage": "1.1V at 800mA, 1.3V at 1A",
            "max_output_current": "1A",
            "quiescent_current": "5 mA typical",
            "line_regulation": "0.2% max",
            "load_regulation": "0.4% max",
            "output_capacitor": "22 uF tantalum or 22 uF low-ESR ceramic required",
            "operating_temp": "0 to 125 C",
            "package": "SOT-223",
        },
        packages=["SOT-223", "TO-252 (DPAK)", "SOT-89"],
        alternatives=["AP2112K-3.3 (lower quiescent, lower dropout)", "XC6220 (ultra-low Iq)", "ME6211 (lower dropout)"],
        lcsc="C6186",
    ),
    Component(
        name="TPS54331",
        category="regulator",
        manufacturer="Texas Instruments",
        description="3A step-down (buck) converter, 3.5V-28V input, 570 kHz",
        specs={
            "topology": "Step-down (buck) converter",
            "input_voltage": "3.5V to 28V",
            "output_voltage": "0.8V to Vin (adjustable via feedback divider)",
            "max_output_current": "3A",
            "switching_frequency": "570 kHz (fixed)",
            "efficiency": "Up to 92%",
            "quiescent_current": "2.2 mA typical",
            "internal_mosfet": "Yes (integrated high-side MOSFET)",
            "soft_start": "Internal, ~5 ms",
            "protection": "OVP, UVLO, thermal shutdown, cycle-by-cycle current limit",
            "operating_temp": "-40 to 125 C",
            "package": "SOIC-8 (exposed pad)",
        },
        packages=["SOIC-8 (exposed pad, PowerPAD)"],
        interfaces=[],
        alternatives=["LM2596 (simpler, lower freq)", "MP1584 (smaller, higher freq)", "TPS5430 (3A, pin-compatible upgrade)"],
        lcsc="C15769",
        notes="Needs external inductor (4.7-22 uH), bootstrap capacitor (100nF), output capacitor (100-220 uF).",
    ),
    Component(
        name="LM2596S-5.0",
        category="regulator",
        manufacturer="Texas Instruments / ON Semi",
        description="5V step-down (buck) converter, 3A, 150 kHz, TO-263-5",
        specs={
            "topology": "Step-down (buck) converter",
            "input_voltage": "4.5V to 40V",
            "output_voltage": "5.0V (fixed)",
            "max_output_current": "3A",
            "switching_frequency": "150 kHz (fixed)",
            "efficiency": "Up to 88%",
            "quiescent_current": "5 mA typical",
            "internal_mosfet": "Yes",
            "operating_temp": "-40 to 125 C",
            "package": "TO-263-5 (D2PAK-5)",
        },
        packages=["TO-263-5", "DIP-8 (through-hole version LM2596T)"],
        alternatives=["TPS54331 (higher efficiency, higher freq)", "MP1584 (much smaller)", "LM2576 (older, pin-compatible)"],
        lcsc="C29781",
    ),
    Component(
        name="AP2112K-3.3TRG1",
        category="regulator",
        manufacturer="Diodes Incorporated",
        description="3.3V LDO regulator, 600mA, ultra-low quiescent current, SOT-23-5",
        specs={
            "output_voltage": "3.3V (fixed)",
            "input_voltage": "2.5V to 6V",
            "dropout_voltage": "250 mV at 600 mA",
            "max_output_current": "600 mA",
            "quiescent_current": "55 uA typical",
            "line_regulation": "0.02 %/V",
            "load_regulation": "0.5 mV/mA",
            "output_noise": "50 uVrms",
            "enable_pin": "Yes (active high, with internal pull-up)",
            "protection": "Overcurrent, thermal shutdown, reverse current",
            "operating_temp": "-40 to 85 C",
            "package": "SOT-23-5",
        },
        packages=["SOT-23-5"],
        alternatives=["ME6211 (pin-compatible)", "XC6220 (even lower Iq)", "RT9013 (300mA, lower noise)"],
        lcsc="C51118",
    ),
    Component(
        name="XC6220B331MR-G",
        category="regulator",
        manufacturer="Torex",
        description="3.3V LDO, 700mA, 0.8 uA quiescent current, SOT-23-5",
        specs={
            "output_voltage": "3.3V (fixed)",
            "input_voltage": "1.0V to 6.0V",
            "dropout_voltage": "120 mV at 100 mA",
            "max_output_current": "700 mA",
            "quiescent_current": "0.8 uA typical",
            "line_regulation": "0.1 %/V",
            "enable_pin": "Yes (active high)",
            "protection": "Overcurrent (foldback), thermal shutdown",
            "operating_temp": "-40 to 85 C",
            "package": "SOT-23-5",
        },
        packages=["SOT-23-5", "USP-6B"],
        alternatives=["AP2112K-3.3 (higher current)", "TPS7A02 (0.25 uA Iq, TI)", "ME6211 (cheaper)"],
        lcsc="C86534",
        notes="Ideal for battery-powered devices where quiescent current matters.",
    ),
    Component(
        name="TPS63020DSJR",
        category="regulator",
        manufacturer="Texas Instruments",
        description="Buck-boost converter, 4A, 1.8V-5.5V input, adjustable output",
        specs={
            "topology": "Buck-boost (single inductor)",
            "input_voltage": "1.8V to 5.5V",
            "output_voltage": "1.2V to 5.5V (adjustable)",
            "max_output_current": "4A (buck mode), 2A (boost mode), 3A (Vin=Vout)",
            "switching_frequency": "2.4 MHz",
            "efficiency": "Up to 96%",
            "quiescent_current": "30 uA (PFM mode)",
            "operating_temp": "-40 to 85 C",
            "package": "VSON-14 (3.5x3.5 mm)",
        },
        packages=["VSON-14"],
        alternatives=["TPS63060 (higher voltage range)", "LTC3130 (lower Iq)", "TPS63802 (newer, better efficiency)"],
        lcsc="C94484",
        notes="Perfect for single Li-Ion/LiPo cell to 3.3V or 5V. Maintains regulation through full battery discharge.",
    ),

    # ===== Op-Amps =====
    Component(
        name="LM358",
        category="opamp",
        manufacturer="Texas Instruments / ON Semi / various",
        description="Dual op-amp, general purpose, single/dual supply",
        specs={
            "channels": "2 (dual)",
            "supply_voltage": "3V to 32V (single) or +/-1.5V to +/-16V (dual)",
            "gbw": "1.1 MHz",
            "slew_rate": "0.3 V/us",
            "input_offset_voltage": "2 mV typical, 7 mV max",
            "input_bias_current": "20 nA typical",
            "input_type": "Bipolar",
            "output_swing": "Ground to Vcc-1.5V",
            "supply_current": "1 mA per amplifier",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8", "DIP-8", "TSSOP-8", "MSOP-8"],
        alternatives=["MCP6002 (rail-to-rail, CMOS)", "LMV358 (lower voltage, R2R output)", "TLV2372 (better specs)"],
        lcsc="C7950",
    ),
    Component(
        name="OPA2134PA",
        category="opamp",
        manufacturer="Texas Instruments",
        description="Dual audio op-amp, JFET input, low distortion, low noise",
        specs={
            "channels": "2 (dual)",
            "supply_voltage": "+/-2.5V to +/-18V",
            "gbw": "8 MHz",
            "slew_rate": "20 V/us",
            "input_offset_voltage": "0.5 mV typical, 2 mV max",
            "input_bias_current": "2 pA typical (JFET input)",
            "thd": "0.00008% at 1 kHz",
            "input_noise_voltage": "8 nV/sqrt(Hz)",
            "output_swing": "+/-(Vs - 2V) at 10 mA",
            "supply_current": "4 mA per amplifier",
            "operating_temp": "-25 to 85 C",
            "package": "DIP-8",
        },
        packages=["DIP-8", "SOIC-8"],
        alternatives=["NE5532 (cheaper, BJT input)", "TL072 (cheaper, acceptable audio)", "OPA2604 (higher performance)"],
        lcsc="C116594",
        notes="Excellent for audio preamps, active filters, DAC output stages.",
    ),
    Component(
        name="TL072",
        category="opamp",
        manufacturer="Texas Instruments",
        description="Dual JFET-input op-amp, low noise, audio grade",
        specs={
            "channels": "2 (dual)",
            "supply_voltage": "+/-6V to +/-18V",
            "gbw": "3 MHz",
            "slew_rate": "13 V/us",
            "input_offset_voltage": "3 mV typical, 10 mV max",
            "input_bias_current": "20 pA typical (JFET)",
            "input_noise_voltage": "18 nV/sqrt(Hz)",
            "output_swing": "+/-(Vs - 3V)",
            "supply_current": "2.5 mA per amplifier",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8", "DIP-8"],
        alternatives=["OPA2134 (better audio specs)", "NE5532 (lower noise, BJT)", "LM358 (cheaper, single supply)"],
        lcsc="C6961",
    ),
    Component(
        name="AD8605ARTZ",
        category="opamp",
        manufacturer="Analog Devices",
        description="Single precision op-amp, rail-to-rail I/O, low noise, CMOS",
        specs={
            "channels": "1 (single)",
            "supply_voltage": "2.7V to 5.5V",
            "gbw": "10 MHz",
            "slew_rate": "5 V/us",
            "input_offset_voltage": "20 uV typical, 65 uV max",
            "input_bias_current": "1 pA typical (CMOS)",
            "input_noise_voltage": "7 nV/sqrt(Hz)",
            "output_swing": "Rail-to-rail (within 50mV of rails)",
            "supply_current": "1.5 mA",
            "operating_temp": "-40 to 125 C",
            "package": "SOT-23-5",
        },
        packages=["SOT-23-5", "MSOP-8 (dual: AD8606)"],
        alternatives=["MCP6001 (cheaper, lower specs)", "OPA333 (zero-drift, lower offset)", "LMV321 (cheaper)"],
        lcsc="C108790",
    ),
    Component(
        name="MCP6002-I/SN",
        category="opamp",
        manufacturer="Microchip",
        description="Dual op-amp, rail-to-rail I/O, 1 MHz, low power, CMOS",
        specs={
            "channels": "2 (dual)",
            "supply_voltage": "1.8V to 6.0V",
            "gbw": "1 MHz",
            "slew_rate": "0.6 V/us",
            "input_offset_voltage": "2 mV typical, 4.5 mV max",
            "input_bias_current": "1 pA typical (CMOS)",
            "output_swing": "Rail-to-rail (25 mV from rails at 5 mA)",
            "supply_current": "100 uA per amplifier",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8", "DIP-8", "MSOP-8", "SOT-23-5 (single: MCP6001)"],
        alternatives=["LMV358 (similar, different manufacturer)", "TSV912 (1.2 MHz, ST)", "TLV2372 (5.5 MHz, TI)"],
        lcsc="C7377",
    ),

    # ===== MOSFETs =====
    Component(
        name="IRF540N",
        category="mosfet",
        manufacturer="Infineon / International Rectifier",
        description="N-channel power MOSFET, 100V, 33A, TO-220",
        specs={
            "type": "N-channel enhancement mode",
            "vds_max": "100V",
            "id_continuous": "33A (at 25C case)",
            "rds_on": "44 mOhm at Vgs=10V",
            "vgs_threshold": "2V to 4V",
            "gate_charge": "71 nC total",
            "input_capacitance": "1700 pF",
            "body_diode": "Yes, 33A, trr=82ns",
            "max_vgs": "+/- 20V",
            "power_dissipation": "130W (TO-220, infinite heatsink)",
            "operating_temp": "-55 to 175 C",
            "package": "TO-220-3",
        },
        packages=["TO-220-3", "D2PAK"],
        alternatives=["IRLZ44N (logic-level gate)", "IRF3205 (55V, lower Rds)", "IPP045N10N (better specs, same class)"],
        lcsc="C2537",
        notes="NOT logic-level — needs 10V gate drive for rated Rds_on. Use IRLZ44N or AO3400 for 3.3V/5V logic.",
    ),
    Component(
        name="AO3400",
        category="mosfet",
        manufacturer="Alpha & Omega Semiconductor",
        description="N-channel MOSFET, 30V, 5.7A, logic-level gate, SOT-23",
        specs={
            "type": "N-channel enhancement mode",
            "vds_max": "30V",
            "id_continuous": "5.7A",
            "rds_on": "26 mOhm at Vgs=4.5V, 40 mOhm at Vgs=2.5V",
            "vgs_threshold": "0.65V to 1.45V",
            "gate_charge": "6.8 nC total",
            "input_capacitance": "600 pF",
            "max_vgs": "+/- 12V",
            "power_dissipation": "1.4W (SOT-23)",
            "operating_temp": "-55 to 150 C",
            "package": "SOT-23-3",
        },
        packages=["SOT-23-3"],
        alternatives=["SI2302 (20V, similar)", "IRLML6344 (lower Rds_on)", "AO3401 (P-channel equivalent)"],
        lcsc="C20917",
        notes="Excellent for logic-level switching (3.3V/5V gate). Very popular for JLCPCB basic parts.",
    ),
    Component(
        name="SI2302CDS-T1-GE3",
        category="mosfet",
        manufacturer="Vishay",
        description="N-channel MOSFET, 20V, 2.6A, logic-level, SOT-23",
        specs={
            "type": "N-channel enhancement mode",
            "vds_max": "20V",
            "id_continuous": "2.6A",
            "rds_on": "50 mOhm at Vgs=4.5V, 80 mOhm at Vgs=2.5V",
            "vgs_threshold": "0.5V to 1.2V",
            "gate_charge": "4 nC total",
            "max_vgs": "+/- 8V",
            "operating_temp": "-55 to 150 C",
            "package": "SOT-23-3",
        },
        packages=["SOT-23-3"],
        alternatives=["AO3400 (30V, lower Rds)", "IRLML2502 (20V, lower Rds)", "BSS138 (lower current, level shifter)"],
        lcsc="C10487",
    ),
    Component(
        name="IRLML6344TRPBF",
        category="mosfet",
        manufacturer="Infineon",
        description="N-channel MOSFET, 30V, 5A, very low Rds_on, SOT-23",
        specs={
            "type": "N-channel enhancement mode",
            "vds_max": "30V",
            "id_continuous": "5A",
            "rds_on": "22 mOhm at Vgs=4.5V, 29 mOhm at Vgs=2.5V",
            "vgs_threshold": "0.6V to 1.1V",
            "gate_charge": "10 nC total",
            "max_vgs": "+/- 12V",
            "operating_temp": "-55 to 150 C",
            "package": "SOT-23-3",
        },
        packages=["SOT-23-3"],
        alternatives=["AO3400 (cheaper)", "SI2302 (lower Vgs_th)", "IRLML0060 (60V version)"],
        lcsc="C181093",
    ),
    Component(
        name="BSS138",
        category="mosfet",
        manufacturer="ON Semiconductor / Nexperia",
        description="N-channel MOSFET, 50V, 200mA, SOT-23 — commonly used for level shifting",
        specs={
            "type": "N-channel enhancement mode",
            "vds_max": "50V",
            "id_continuous": "200 mA",
            "rds_on": "3.5 Ohm at Vgs=4.5V",
            "vgs_threshold": "0.8V to 1.5V",
            "gate_charge": "1 nC total",
            "max_vgs": "+/- 20V",
            "operating_temp": "-55 to 150 C",
            "package": "SOT-23-3",
        },
        packages=["SOT-23-3"],
        alternatives=["2N7002 (similar, slightly different specs)", "TXS0108E (dedicated level shifter IC)", "SN74LVC1T45 (single-bit level translator)"],
        lcsc="C21394",
        notes="Classic N-MOSFET for I2C level shifting (with pull-up resistors on both sides).",
    ),
    Component(
        name="AO3401A",
        category="mosfet",
        manufacturer="Alpha & Omega Semiconductor",
        description="P-channel MOSFET, -30V, -4A, logic-level gate, SOT-23",
        specs={
            "type": "P-channel enhancement mode",
            "vds_max": "-30V",
            "id_continuous": "-4A",
            "rds_on": "42 mOhm at Vgs=-4.5V, 65 mOhm at Vgs=-2.5V",
            "vgs_threshold": "-0.5V to -1.3V",
            "gate_charge": "5 nC total",
            "max_vgs": "+/- 12V",
            "operating_temp": "-55 to 150 C",
            "package": "SOT-23-3",
        },
        packages=["SOT-23-3"],
        alternatives=["DMP3099L (lower Rds_on)", "Si2301 (20V, lower Rds)", "DMG2305UX (lower Rds, SOT-23)"],
        lcsc="C15127",
        notes="Useful for high-side switching, reverse polarity protection, load switches.",
    ),

    # ===== Sensors =====
    Component(
        name="BME280",
        category="sensor",
        manufacturer="Bosch Sensortec",
        description="Combined humidity, pressure, and temperature sensor, I2C/SPI",
        specs={
            "measured_quantities": "Temperature, Humidity, Barometric Pressure",
            "temperature_range": "-40 to 85 C",
            "temperature_accuracy": "+/-1 C (typical), +/-0.5 C (0 to 65 C)",
            "temperature_resolution": "0.01 C",
            "humidity_range": "0% to 100% RH",
            "humidity_accuracy": "+/-3% RH",
            "pressure_range": "300 to 1100 hPa",
            "pressure_accuracy": "+/-1 hPa absolute",
            "supply_voltage": "1.71V to 3.6V",
            "supply_current": "3.6 uA at 1 Hz (humidity+pressure+temp)",
            "interface": "I2C (up to 3.4 MHz) or SPI (up to 10 MHz)",
            "i2c_address": "0x76 (SDO=GND) or 0x77 (SDO=VDD)",
            "operating_temp": "-40 to 85 C",
            "package": "LGA-8 (2.5x2.5x0.93 mm)",
        },
        packages=["LGA-8 (2.5x2.5 mm)"],
        interfaces=["I2C", "SPI"],
        alternatives=["BMP280 (no humidity, cheaper)", "BME680 (adds gas/VOC)", "SHT31 (humidity+temp only, more accurate)"],
        lcsc="C92489",
    ),
    Component(
        name="MPU-6050",
        category="sensor",
        manufacturer="InvenSense / TDK",
        description="6-axis IMU — 3-axis gyroscope + 3-axis accelerometer, I2C",
        specs={
            "accelerometer_range": "+/-2g, +/-4g, +/-8g, +/-16g (selectable)",
            "gyroscope_range": "+/-250, +/-500, +/-1000, +/-2000 dps (selectable)",
            "adc_resolution": "16-bit",
            "supply_voltage": "2.375V to 3.46V",
            "supply_current": "3.9 mA typical (gyro+accel)",
            "interface": "I2C (up to 400 kHz)",
            "i2c_address": "0x68 (AD0=GND) or 0x69 (AD0=VDD)",
            "aux_i2c": "Auxiliary I2C master for external magnetometer",
            "digital_motion_processor": "Yes (DMP for sensor fusion)",
            "fifo_buffer": "1024 bytes",
            "operating_temp": "-40 to 85 C",
            "package": "QFN-24 (4x4x0.9 mm)",
        },
        packages=["QFN-24 (4x4 mm)"],
        interfaces=["I2C"],
        alternatives=["ICM-20948 (9-axis, successor)", "LSM6DS3 (6-axis, ST)", "BMI270 (6-axis, lower power)"],
        lcsc="C24112",
        notes="End-of-life. Use ICM-42688-P or BMI270 for new designs.",
    ),
    Component(
        name="ADS1115IDGSR",
        category="sensor",
        manufacturer="Texas Instruments",
        description="16-bit 4-channel ADC, I2C, programmable gain amplifier",
        specs={
            "resolution": "16-bit",
            "channels": "4 single-ended or 2 differential",
            "sample_rate": "8 to 860 SPS (programmable)",
            "pga_gain": "2/3, 1, 2, 4, 8, 16x (FSR: +/-6.144V to +/-0.256V)",
            "input_voltage_range": "GND-0.3V to VDD+0.3V",
            "supply_voltage": "2.0V to 5.5V",
            "supply_current": "150 uA (continuous conversion)",
            "interface": "I2C (up to 3.4 MHz)",
            "i2c_address": "0x48, 0x49, 0x4A, 0x4B (4 selectable via ADDR pin)",
            "reference": "Internal 2.048V",
            "comparator": "Built-in threshold comparator with alert pin",
            "operating_temp": "-40 to 125 C",
            "package": "MSOP-10",
        },
        packages=["MSOP-10", "VSSOP-10"],
        interfaces=["I2C"],
        alternatives=["ADS1015 (12-bit, faster, cheaper)", "MCP3424 (18-bit, slower)", "ADS1219 (24-bit, 2 differential)"],
        lcsc="C37593",
    ),
    Component(
        name="INA226AIDGSR",
        category="sensor",
        manufacturer="Texas Instruments",
        description="High-side/low-side current and power monitor, 36V, I2C",
        specs={
            "measured_quantities": "Bus voltage, shunt voltage, current, power",
            "bus_voltage_range": "0V to 36V",
            "shunt_voltage_range": "+/-81.92 mV (full scale)",
            "adc_resolution": "16-bit",
            "shunt_voltage_lsb": "2.5 uV",
            "bus_voltage_lsb": "1.25 mV",
            "offset_voltage": "+/-10 uV max",
            "common_mode_rejection": "132 dB min",
            "supply_voltage": "2.7V to 5.5V",
            "supply_current": "330 uA (operating)",
            "interface": "I2C (up to 2.94 MHz, 16 selectable addresses)",
            "alert_pin": "Yes (over/under voltage, power, conversion ready)",
            "averaging": "1, 4, 16, 64, 128, 256, 512, 1024 samples",
            "operating_temp": "-40 to 125 C",
            "package": "MSOP-10",
        },
        packages=["MSOP-10"],
        interfaces=["I2C"],
        alternatives=["INA219 (26V max, simpler)", "INA228 (85V, 20-bit)", "PAC1921 (high-side only, simpler)"],
        lcsc="C138706",
    ),
    Component(
        name="MAX31855KASA+T",
        category="sensor",
        manufacturer="Analog Devices / Maxim",
        description="Cold-junction compensated thermocouple-to-digital converter, SPI, K-type",
        specs={
            "thermocouple_type": "K-type",
            "temperature_range": "-200 C to +1350 C (thermocouple)",
            "cold_junction_range": "-40 C to +125 C",
            "resolution": "14-bit (0.25 C per bit, thermocouple), 12-bit (0.0625 C, cold junction)",
            "accuracy": "+/-2 C (thermocouple, -200 to +700 C range)",
            "supply_voltage": "3.0V to 3.6V",
            "supply_current": "1.5 mA typical",
            "interface": "SPI (read-only, 5 MHz max)",
            "conversion_time": "100 ms typical",
            "fault_detection": "Open circuit, short to GND, short to VCC",
            "operating_temp": "-40 to 125 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8"],
        interfaces=["SPI"],
        alternatives=["MAX31856 (universal thermocouple, higher accuracy)", "MAX6675 (older, 12-bit)", "MCP9600 (I2C, multi-type)"],
        lcsc="C53862",
    ),
    Component(
        name="DS18B20",
        category="sensor",
        manufacturer="Analog Devices / Maxim",
        description="Digital temperature sensor, 1-Wire, +/-0.5C accuracy",
        specs={
            "temperature_range": "-55 C to +125 C",
            "accuracy": "+/-0.5 C (-10 C to +85 C)",
            "resolution": "9 to 12 bits programmable (0.5 C to 0.0625 C)",
            "supply_voltage": "3.0V to 5.5V",
            "supply_current": "1 mA typical (active), 750 nA (standby)",
            "interface": "1-Wire (parasitic power mode supported)",
            "conversion_time": "93.75 ms (9-bit) to 750 ms (12-bit)",
            "unique_id": "64-bit unique ROM code",
            "alarm_function": "Programmable high/low temperature alarm",
            "operating_temp": "-55 to 125 C",
            "package": "TO-92-3",
        },
        packages=["TO-92-3", "SOIC-8 (SMD version)", "Waterproof probe (with cable)"],
        interfaces=["1-Wire"],
        alternatives=["TMP117 (I2C, +/-0.1C)", "BME280 (adds humidity + pressure)", "SHT31 (I2C, humidity+temp)"],
        lcsc="C376006",
        notes="Multiple sensors can share a single data pin (1-Wire bus). 4.7K pull-up required on data line.",
    ),

    # ===== Passives =====
    Component(
        name="100nF 0402 MLCC",
        category="passive",
        manufacturer="Samsung / Murata / Yageo / various",
        description="100nF (0.1uF) ceramic capacitor, 0402, 16V or 25V, X5R/X7R",
        specs={
            "capacitance": "100 nF (0.1 uF)",
            "voltage_rating": "16V or 25V (common)",
            "dielectric": "X5R or X7R",
            "tolerance": "+/-10% (K) typical",
            "temp_coefficient": "X5R: -55 to +85 C, +/-15%; X7R: -55 to +125 C, +/-15%",
            "package": "0402 (1005 metric)",
            "dimensions": "1.0 x 0.5 mm",
            "typical_use": "Bypass/decoupling capacitor for IC power pins",
        },
        packages=["0402", "0603", "0805"],
        lcsc="C1525 (Samsung CL05B104KO5NNNC, 0402 16V X5R)",
        notes="Place as close as possible to IC power pins. One per VDD/VCC pin minimum.",
    ),
    Component(
        name="10K 0402 Resistor",
        category="passive",
        manufacturer="Yageo / UniOhm / various",
        description="10K ohm resistor, 0402, 1/16W, 1%",
        specs={
            "resistance": "10 kOhm",
            "tolerance": "1% (F)",
            "power_rating": "1/16W (0.0625W)",
            "voltage_rating": "50V max",
            "temp_coefficient": "100 ppm/C",
            "package": "0402 (1005 metric)",
            "dimensions": "1.0 x 0.5 mm",
            "typical_use": "Pull-up/pull-down resistors, voltage dividers, current limiting",
        },
        packages=["0402", "0603", "0805", "1206"],
        lcsc="C25744 (Yageo RC0402FR-0710KL)",
    ),
    Component(
        name="10uF 0805 MLCC",
        category="passive",
        manufacturer="Samsung / Murata / TDK",
        description="10uF ceramic capacitor, 0805, 10V or 16V, X5R",
        specs={
            "capacitance": "10 uF",
            "voltage_rating": "10V or 16V",
            "dielectric": "X5R",
            "tolerance": "+/-20% (M) typical",
            "dc_bias_effect": "Effective capacitance drops ~50% at rated voltage",
            "package": "0805 (2012 metric)",
            "typical_use": "Bulk decoupling, LDO output capacitor, power supply filtering",
        },
        packages=["0402 (limited to 6.3V)", "0603", "0805", "1206"],
        lcsc="C15850 (Samsung CL21A106KAYNNNE, 0805 25V X5R)",
        notes="DC bias effect is significant: a 10uF 0805 6.3V cap may only give ~5uF at 3.3V bias. Derate voltage by 2x.",
    ),
    Component(
        name="100uH Power Inductor",
        category="passive",
        manufacturer="Wurth / TDK / Bourns",
        description="100uH shielded power inductor, various current ratings",
        specs={
            "inductance": "100 uH",
            "dcr": "0.2-2 Ohm (varies by current rating)",
            "saturation_current": "0.3A to 3A (varies by size/model)",
            "rated_current": "Based on temperature rise, typically 70-80% of Isat",
            "shielding": "Shielded (semi-shielded or fully shielded)",
            "typical_use": "Buck/boost converter energy storage, LC filters, EMI filtering",
        },
        packages=["4x4 mm", "5x5 mm", "6x6 mm", "8x8 mm", "10x10 mm"],
        notes="Always verify Isat > peak inductor current. DCR causes conduction losses and voltage drop.",
    ),
    Component(
        name="Ferrite Bead 0603",
        category="passive",
        manufacturer="Murata / TDK / Wurth",
        description="Ferrite bead, 0603, 600 ohm @ 100 MHz, common for EMI filtering",
        specs={
            "impedance_100mhz": "600 Ohm @ 100 MHz (typical)",
            "dcr": "0.1-0.3 Ohm",
            "rated_current": "200 mA to 1A (varies)",
            "package": "0603 (1608 metric)",
            "typical_use": "Power supply filtering, separating analog/digital grounds, USB VBUS filtering",
        },
        packages=["0402", "0603", "0805"],
        lcsc="C1015 (Murata BLM18PG600SN1D, 0603 600ohm)",
        notes="Not a replacement for a capacitor. Place in series with power line, with bypass caps on both sides.",
    ),

    # ===== Interface / Communication ICs =====
    Component(
        name="CH340G",
        category="interface",
        manufacturer="WCH",
        description="USB to UART bridge, full-speed USB 2.0, up to 2 Mbps",
        specs={
            "usb": "USB 2.0 Full-speed (12 Mbps)",
            "uart_baud": "50 bps to 2 Mbps",
            "supply_voltage": "3.3V or 5V (internal 3.3V regulator)",
            "uart_signals": "TXD, RXD, CTS, DTR, DSR, RI, DCD, RTS",
            "modem_signals": "Yes (full handshake lines)",
            "crystal": "12 MHz external crystal required (CH340G), or no crystal (CH340C/K)",
            "driver_support": "Windows, macOS, Linux (built-in since 2022+)",
            "operating_temp": "-40 to 85 C",
            "package": "SOP-16",
        },
        packages=["SOP-16 (CH340G)", "ESSOP-10 (CH340C, no crystal)", "SOP-16 (CH340K, no crystal)"],
        interfaces=["USB", "UART"],
        alternatives=["CP2102 (Silicon Labs, popular)", "FT232RL (FTDI, premium)", "CH9102F (WCH, newer, USB-C friendly)"],
        lcsc="C14267",
        notes="CH340C and CH340K variants do not need an external crystal — saves BOM cost.",
    ),
    Component(
        name="MAX232",
        category="interface",
        manufacturer="Texas Instruments / Maxim / various",
        description="Dual RS-232 driver/receiver, 5V, charge pump",
        specs={
            "channels": "2 drivers + 2 receivers",
            "supply_voltage": "4.5V to 5.5V",
            "data_rate": "120 kbps (MAX232) or 1 Mbps (MAX232A)",
            "charge_pump_caps": "4x 1 uF (MAX232) or 4x 100 nF (MAX232A)",
            "output_voltage_swing": "+/-5V to +/-15V (RS-232 levels)",
            "input_threshold": "+/-3V (RS-232 standard)",
            "operating_temp": "-40 to 85 C",
            "package": "DIP-16 / SOIC-16",
        },
        packages=["DIP-16", "SOIC-16", "TSSOP-16"],
        alternatives=["MAX3232 (3.3V compatible)", "SP3232 (3.3V, cheaper)", "CH340G (if converting to USB instead)"],
        lcsc="C6612",
    ),
    Component(
        name="NE555",
        category="timer",
        manufacturer="Texas Instruments / various",
        description="Precision timer IC, monostable/astable operation",
        specs={
            "supply_voltage": "4.5V to 16V",
            "max_frequency": "~500 kHz",
            "output_current": "200 mA (source or sink)",
            "timing_accuracy": "Depends on external RC, ~1% typical",
            "duty_cycle": "Adjustable (50-100% astable, any with diode trick)",
            "trigger_voltage": "1/3 VCC",
            "threshold_voltage": "2/3 VCC",
            "discharge_pin": "Open-collector output",
            "operating_temp": "-40 to 85 C",
            "package": "DIP-8 / SOIC-8",
        },
        packages=["DIP-8", "SOIC-8", "MSOP-8", "SOT-23-5 (LMC555, CMOS)"],
        alternatives=["LMC555 (CMOS, low power, wider voltage)", "TLC555 (CMOS, TI)", "ICM7555 (CMOS, Renesas)"],
        lcsc="C46971",
    ),
    Component(
        name="CD4051BE",
        category="analog_switch",
        manufacturer="Texas Instruments / NXP / various",
        description="8-channel analog multiplexer/demultiplexer, CMOS",
        specs={
            "channels": "8:1 (8 inputs, 1 common output)",
            "supply_voltage": "3V to 18V (VDD-VSS, max 18V)",
            "on_resistance": "120 Ohm typical (at VDD=15V), 250 Ohm (at VDD=5V)",
            "analog_signal_range": "VSS to VDD",
            "channel_select": "3 address lines (A, B, C) + inhibit",
            "crosstalk": "-50 dB typical at 1 MHz",
            "switching_time": "~200 ns",
            "operating_temp": "-55 to 125 C",
            "package": "DIP-16 / SOIC-16",
        },
        packages=["DIP-16", "SOIC-16", "TSSOP-16"],
        alternatives=["CD4052 (dual 4:1)", "CD4053 (triple 2:1)", "ADG508F (lower Ron, wider supply)"],
        lcsc="C6524",
    ),
    Component(
        name="W25Q128JVSIQ",
        category="memory",
        manufacturer="Winbond",
        description="128 Mbit (16 MB) SPI NOR flash, 133 MHz, SOIC-8",
        specs={
            "capacity": "128 Mbit (16 MB)",
            "interface": "SPI (standard, dual, quad), up to 133 MHz",
            "supply_voltage": "2.7V to 3.6V",
            "page_size": "256 bytes",
            "sector_size": "4 KB (smallest erasable unit)",
            "block_size": "32 KB / 64 KB",
            "page_program_time": "0.7 ms typical",
            "sector_erase_time": "45 ms typical",
            "endurance": "100,000 erase/program cycles",
            "data_retention": "20 years",
            "read_current": "5 mA (normal read)",
            "standby_current": "1 uA",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8 (208 mil)", "SOIC-8 (150 mil)", "WSON-8 (8x6 mm)", "USON-8 (5x6 mm)"],
        interfaces=["SPI", "Dual SPI", "Quad SPI (QPI)"],
        alternatives=["W25Q64JV (64 Mbit, cheaper)", "GD25Q128E (GigaDevice, compatible)", "IS25LP128 (ISSI, compatible)"],
        lcsc="C97521",
    ),

    # ===== Connectors =====
    Component(
        name="USB Type-C Receptacle (16-pin)",
        category="connector",
        manufacturer="Various (Korean Hroparts, Jing Extension, GCT)",
        description="USB Type-C 2.0 receptacle, 16-pin, mid-mount or SMD",
        specs={
            "type": "USB Type-C receptacle (female)",
            "usb_standard": "USB 2.0 (adequate for most MCU projects)",
            "current_rating": "5A VBUS (USB PD capable with proper CC config)",
            "pins": "16 (simplified) or 24 (full)",
            "cc_resistors": "5.1K pull-down on CC1 and CC2 for UFP (device) role",
            "vbus_voltage": "5V default, up to 20V with USB PD",
            "mounting": "SMD (mid-mount, through-hole tabs for strength)",
            "mating_cycles": "10,000 minimum",
        },
        packages=["16-pin mid-mount", "16-pin SMD", "24-pin full (for USB 3.x)"],
        lcsc="C2765186 (Korean Hroparts TYPE-C-31-M-12)",
        notes="For USB 2.0 device: only need D+, D-, VBUS, GND, CC1, CC2. Add 5.1K pull-downs on CC1/CC2. Shield pin to GND via 1M + 4.7nF.",
    ),
    Component(
        name="JST-XH 2.54mm",
        category="connector",
        manufacturer="JST",
        description="JST XH series connector, 2.54mm pitch, wire-to-board",
        specs={
            "pitch": "2.54 mm",
            "current_rating": "3A per contact",
            "voltage_rating": "250V",
            "wire_gauge": "AWG 22-28",
            "positions": "2 to 16 (various)",
            "locking": "Friction lock (snap-in latch)",
            "mounting": "Through-hole vertical or right-angle",
            "mating_cycles": "30 minimum",
            "typical_use": "Battery connections, sensor cables, internal wiring",
        },
        packages=["Through-hole vertical", "Through-hole right-angle"],
        alternatives=["Molex KK 2.54mm", "JST-PH 2.0mm (smaller)", "JST-SH 1.0mm (smallest, Qwiic/STEMMA QT)"],
        lcsc="C158012 (XH-2A header)",
    ),
    Component(
        name="Pin Header 2.54mm",
        category="connector",
        manufacturer="Various",
        description="Standard 2.54mm (0.1 inch) pin header, male, through-hole",
        specs={
            "pitch": "2.54 mm (0.1 inch)",
            "current_rating": "3A per pin",
            "voltage_rating": "250V",
            "contact_resistance": "20 mOhm max",
            "insulation": "Nylon, UL94V-0",
            "mounting": "Through-hole",
            "typical_configurations": "1x40, 2x20, 1x6, 1x4, custom breakable",
            "typical_use": "Arduino/Raspberry Pi headers, debug ports, jumpers",
        },
        packages=["1-row", "2-row"],
        alternatives=["Female socket headers", "JST-SH for smaller pitch", "FPC connectors for flat cables"],
        lcsc="C124378 (1x40 male, straight)",
    ),

    # ===== Power Management / Battery =====
    Component(
        name="TP4056",
        category="charger",
        manufacturer="NanJing Top Power",
        description="1A Li-Ion/LiPo linear battery charger, SOP-8",
        specs={
            "chemistry": "Li-Ion / LiPo (single cell, 4.2V)",
            "charge_current": "Programmable up to 1A (set by RPROG resistor)",
            "charge_voltage": "4.2V +/-1%",
            "input_voltage": "4.5V to 8V (typically 5V USB)",
            "charge_method": "CC/CV (constant current / constant voltage)",
            "termination_current": "1/10 of programmed charge current",
            "status_pins": "CHRG (charging), STDBY (standby/done)",
            "thermal_regulation": "Internal (reduces current at high temperature)",
            "operating_temp": "-40 to 85 C",
            "package": "SOP-8",
            "rprog_formula": "RPROG = 1200 / I_charge_mA (e.g., 1.2K for 1A, 2K for 600mA)",
        },
        packages=["SOP-8"],
        alternatives=["MCP73831 (Microchip, 500mA, SOT-23-5)", "BQ24075 (TI, higher features, power path)", "LTC4054 (Analog Devices)"],
        lcsc="C725790",
        notes="Always pair with DW01A + 8205A for battery protection (overcurrent, overdischarge, overcharge).",
    ),
    Component(
        name="DW01A",
        category="protection",
        manufacturer="Fortune Semiconductor",
        description="Single-cell Li-Ion/LiPo battery protection IC, SOT-23-6",
        specs={
            "protection": "Overcharge (4.3V), overdischarge (2.4V), overcurrent (3A), short circuit",
            "supply_voltage": "2.0V to 5.0V",
            "supply_current": "3 uA typical (standby)",
            "overcharge_threshold": "4.3V +/-50mV (release at 4.15V)",
            "overdischarge_threshold": "2.4V +/-100mV (release at 3.0V)",
            "overcurrent_threshold": "150mV across sense resistor",
            "short_circuit_threshold": "1.35V across sense resistor",
            "delay": "Overcharge: 80ms, Overdischarge: 40ms, Overcurrent: 8ms",
            "operating_temp": "-40 to 85 C",
            "package": "SOT-23-6",
        },
        packages=["SOT-23-6"],
        alternatives=["S-8241 (Seiko)", "AP9101C (Diodes Inc)", "BQ29728 (TI, higher precision)"],
        lcsc="C14213",
        notes="Paired with dual N-MOSFET (FS8205A / 8205A) for switching. Standard Li-Ion protection circuit.",
    ),

    # ===== LED Drivers / Display =====
    Component(
        name="WS2812B",
        category="led",
        manufacturer="Worldsemi",
        description="Addressable RGB LED, 5050 package, integrated controller, single-wire protocol",
        specs={
            "led_type": "RGB, common anode, integrated controller",
            "supply_voltage": "3.5V to 5.3V",
            "supply_current": "~60 mA max (all colors full brightness)",
            "data_rate": "800 kbps (NZR protocol)",
            "colors": "16.7 million (256 levels per R/G/B)",
            "data_format": "24-bit (GRB order: 8G + 8R + 8B)",
            "refresh_rate": ">400 Hz",
            "cascade": "Unlimited (daisy-chain DIN to DOUT)",
            "timing": "T0H=400ns, T0L=850ns, T1H=800ns, T1L=450ns, reset>280us",
            "package": "5050 (5x5 mm SMD)",
        },
        packages=["5050 SMD", "3535 SMD (WS2812B-Mini)", "2020 SMD (WS2812C)"],
        alternatives=["SK6812 (RGBW variant, compatible protocol)", "APA102 (SPI, better for high refresh)", "WS2813 (dual data line, fault tolerant)"],
        lcsc="C2761795",
        notes="Add 100nF bypass cap per LED (or every 3-5 LEDs minimum). 300-500 ohm series resistor on data line. Consider level shifter for 3.3V MCU.",
    ),
    Component(
        name="SSD1306",
        category="display",
        manufacturer="Solomon Systech",
        description="128x64 OLED driver IC, I2C/SPI, commonly on 0.96 inch modules",
        specs={
            "resolution": "128 x 64 pixels",
            "interface": "I2C (up to 400 kHz) or SPI (up to 10 MHz)",
            "i2c_address": "0x3C or 0x3D",
            "supply_voltage": "1.65V to 3.3V (logic), 7V-15V (panel drive, internal charge pump)",
            "supply_current": "~20 mA typical (all pixels on)",
            "contrast_ratio": "2000:1 typical",
            "display_ram": "128x64 / 8 = 1024 bytes",
            "color_depth": "Monochrome (white, blue, or yellow/blue depending on panel)",
            "operating_temp": "-40 to 85 C",
            "package": "Module (0.96 inch typical)",
        },
        packages=["0.96 inch module", "1.3 inch module (SH1106 usually)", "bare IC COG/COF"],
        interfaces=["I2C", "SPI"],
        alternatives=["SH1106 (132x64, for 1.3 inch)", "SSD1309 (same but larger displays)", "ST7789 (color TFT, SPI)"],
        lcsc="Module (not bare IC typically)",
    ),

    # ===== Communication Modules =====
    Component(
        name="SX1276",
        category="rf",
        manufacturer="Semtech",
        description="LoRa transceiver, 137-1020 MHz, SPI, long-range sub-GHz",
        specs={
            "frequency_range": "137-1020 MHz (ISM bands: 433/868/915 MHz)",
            "modulation": "LoRa (CSS) + FSK/OOK",
            "max_output_power": "+20 dBm (100 mW)",
            "sensitivity": "-148 dBm (LoRa, SF12, BW 7.8 kHz)",
            "link_budget": "168 dB",
            "data_rate": "0.018 to 37.5 kbps (LoRa), up to 300 kbps (FSK)",
            "supply_voltage": "1.8V to 3.7V",
            "supply_current": "10.8 mA (RX), 120 mA (TX +20 dBm)",
            "sleep_current": "0.2 uA",
            "interface": "SPI",
            "operating_temp": "-40 to 85 C",
            "package": "QFN-28 (4x4 mm)",
        },
        packages=["QFN-28 (bare IC)", "RFM95W module (HopeRF)", "Ra-01 module (Ai-Thinker)"],
        interfaces=["SPI"],
        alternatives=["SX1262 (newer, better power efficiency, +22 dBm)", "SX1278 (pin-compatible, 137-525 MHz only)", "LLCC68 (lower cost, LoRa subset)"],
        lcsc="C90089",
    ),
    Component(
        name="CAN Transceiver MCP2551",
        category="interface",
        manufacturer="Microchip",
        description="High-speed CAN transceiver, 1 Mbps, 8-pin",
        specs={
            "standard": "ISO 11898-2 (CAN 2.0B)",
            "data_rate": "Up to 1 Mbps",
            "supply_voltage": "4.5V to 5.5V",
            "bus_pins": "CANH, CANL",
            "standby_current": "1 uA",
            "dominant_voltage": "CANH=3.5V, CANL=1.5V typical",
            "recessive_voltage": "CANH=CANL=2.5V",
            "slope_control": "Yes (RS pin)",
            "esd_protection": "+/-4 kV HBM",
            "operating_temp": "-40 to 125 C",
            "package": "DIP-8 / SOIC-8",
        },
        packages=["DIP-8", "SOIC-8"],
        alternatives=["SN65HVD230 (3.3V, CAN 2.0)", "MCP2542 (CAN FD, 5 Mbps)", "TJA1050 (NXP, equivalent)"],
        lcsc="C1523",
    ),

    # ===== ESD / Protection =====
    Component(
        name="USBLC6-2SC6",
        category="protection",
        manufacturer="STMicroelectronics",
        description="Low-capacitance ESD protection for USB 2.0 lines, SOT-23-6",
        specs={
            "channels": "2 bidirectional lines + VBUS",
            "working_voltage": "5.25V (VBUS), 3.6V (data lines)",
            "clamping_voltage": "7.5V at 1A (data lines)",
            "line_capacitance": "0.85 pF typical (data lines)",
            "esd_rating": "+/-15 kV (air), +/-8 kV (contact) per IEC 61000-4-2",
            "leakage_current": "10 nA max",
            "operating_temp": "-40 to 125 C",
            "package": "SOT-23-6",
        },
        packages=["SOT-23-6"],
        alternatives=["TPD2E2U06 (TI, similar)", "PRTR5V0U2X (Nexperia, ultra-low cap)", "SP0502BAHT (Littelfuse)"],
        lcsc="C7519",
        notes="Essential for USB ports. Place as close as possible to the connector. Route data lines through IC, not around it.",
    ),
    Component(
        name="TVS Diode SMAJ5.0A",
        category="protection",
        manufacturer="Littelfuse / various",
        description="TVS diode, 5V working voltage, 400W, unidirectional, SMA",
        specs={
            "working_voltage": "5.0V",
            "breakdown_voltage": "6.4V min",
            "clamping_voltage": "9.2V at 43.5A",
            "peak_pulse_current": "43.5A (8/20us)",
            "peak_pulse_power": "400W (10/1000us)",
            "leakage_current": "1 uA at 5V",
            "direction": "Unidirectional",
            "operating_temp": "-55 to 150 C",
            "package": "SMA (DO-214AC)",
        },
        packages=["SMA", "SMB (higher power)", "SMC (even higher)"],
        alternatives=["SMBJ5.0A (600W, larger)", "PESD5V0S1BA (SOD-323, lower power)", "SMAJ12A (12V version)"],
        lcsc="C35517",
    ),

    # ===== Oscillators / Crystals =====
    Component(
        name="8 MHz Crystal HC49S",
        category="crystal",
        manufacturer="Various",
        description="8 MHz quartz crystal, HC49S through-hole or SMD, 20 pF load",
        specs={
            "frequency": "8 MHz",
            "frequency_tolerance": "+/-20 ppm (at 25 C)",
            "frequency_stability": "+/-30 ppm (-20 to +70 C)",
            "load_capacitance": "20 pF (common for STM32)",
            "esr": "60 Ohm max",
            "drive_level": "200 uW max",
            "package": "HC49S (through-hole) or 3225 SMD (3.2x2.5 mm)",
        },
        packages=["HC49S", "HC49SMD", "3225 (3.2x2.5 mm SMD)", "2520 (2.5x2.0 mm SMD)"],
        lcsc="C32346 (HC49S, 8MHz, 20pF)",
        notes="Load capacitors: CL = 2*(Cload - Cstray). For 20pF load with 5pF stray: CL = 2*(20-5) = 30 pF, so use 2x 30pF caps. Stray is typically 3-7 pF.",
    ),
    Component(
        name="32.768 kHz Crystal",
        category="crystal",
        manufacturer="Various",
        description="32.768 kHz tuning fork crystal for RTC, low power",
        specs={
            "frequency": "32.768 kHz (2^15 Hz, divides to 1 Hz)",
            "frequency_tolerance": "+/-20 ppm",
            "load_capacitance": "6 pF or 12.5 pF (check MCU datasheet)",
            "esr": "35 kOhm max (tuning fork type)",
            "drive_level": "1 uW max",
            "package": "3215 (3.2x1.5 mm SMD) or cylindrical through-hole",
        },
        packages=["3215 SMD", "2012 SMD (2.0x1.2 mm)", "Cylindrical TH (2x6 mm)"],
        lcsc="C32346",
        notes="Very sensitive to PCB stray capacitance. Use 6.8pF or 10pF load caps typically. Keep traces short, guard ring recommended.",
    ),

    # ===== More MCUs =====
    Component(
        name="STM32G431CBU6",
        category="mcu",
        manufacturer="STMicroelectronics",
        description="ARM Cortex-M4F, 170 MHz, 128 KB Flash, motor control, USB",
        specs={
            "core": "ARM Cortex-M4F (FPU, DSP)",
            "max_frequency": "170 MHz",
            "flash": "128 KB",
            "sram": "32 KB",
            "supply_voltage": "1.71V to 3.6V",
            "gpio_count": "26",
            "adc_channels": "2x 12-bit ADC (5 Msps)",
            "dac": "3x 12-bit DAC",
            "timers": "11 (including HRTIM for motor control)",
            "uart": "3",
            "spi": "2",
            "i2c": "2",
            "usb": "USB 2.0 Full-speed (crystal-less)",
            "can": "1x CAN FD",
            "cordic": "Yes (hardware sin/cos/sqrt)",
            "operating_temp": "-40 to 85 C",
            "package": "UFQFPN-48",
        },
        packages=["UFQFPN-48", "LQFP-48", "LQFP-64 (RB variant)"],
        interfaces=["SPI", "I2C", "UART", "USB", "CAN FD"],
        alternatives=["STM32F303 (older, similar class)", "STM32G474 (more peripherals)", "ESP32-S3 (if Wi-Fi needed)"],
        lcsc="C529381",
        notes="Great for motor control (FOC) with CORDIC coprocessor and advanced timers. Crystal-less USB saves BOM.",
    ),
    Component(
        name="PY32F002AF15P6TU",
        category="mcu",
        manufacturer="Puya Semiconductor",
        description="ARM Cortex-M0+, 24 MHz, 20 KB Flash, ultra-low-cost, TSSOP-20",
        specs={
            "core": "ARM Cortex-M0+",
            "max_frequency": "24 MHz",
            "flash": "20 KB",
            "sram": "3 KB",
            "supply_voltage": "1.7V to 5.5V",
            "gpio_count": "17",
            "adc_channels": "8 (12-bit)",
            "timers": "5",
            "uart": "2",
            "spi": "1",
            "i2c": "1",
            "operating_temp": "-40 to 85 C",
            "package": "TSSOP-20",
        },
        packages=["TSSOP-20", "SOP-8 (PY32F002A)", "QFN-20"],
        interfaces=["SPI", "I2C", "UART"],
        alternatives=["CH32V003 (RISC-V, similar price)", "STM32C011 (ST ecosystem)", "ATtiny1616 (Microchip)"],
        lcsc="C5176073",
        notes="~$0.03-0.05 in quantity. Cheapest ARM Cortex-M0+ available. Uses SWD for programming/debug.",
    ),

    # ===== More Sensors =====
    Component(
        name="TMP117AIDRVR",
        category="sensor",
        manufacturer="Texas Instruments",
        description="High-accuracy digital temperature sensor, +/-0.1 C, 16-bit, I2C",
        specs={
            "temperature_range": "-55 C to +150 C",
            "accuracy": "+/-0.1 C (-20 to +50 C), +/-0.15 C (-40 to +70 C)",
            "resolution": "0.0078 C (16-bit, 7.8125 mC/LSB)",
            "supply_voltage": "1.8V to 5.5V",
            "supply_current": "3.5 uA (1 Hz conversion)",
            "interface": "I2C (up to 400 kHz)",
            "i2c_address": "0x48, 0x49, 0x4A, 0x4B (4 selectable)",
            "alert_pin": "Yes (programmable thresholds)",
            "nist_traceable": "Yes",
            "operating_temp": "-55 to 150 C",
            "package": "SOT-563 (6-pin, 1.6x1.2 mm)",
        },
        packages=["SOT-563"],
        interfaces=["I2C"],
        alternatives=["TMP116 (+/-0.2C, cheaper)", "DS18B20 (+/-0.5C, 1-Wire)", "STS40 (Sensirion, +/-0.2C)"],
        lcsc="C2677286",
    ),
    Component(
        name="SHT31-DIS-B",
        category="sensor",
        manufacturer="Sensirion",
        description="Digital humidity and temperature sensor, +/-2% RH, I2C",
        specs={
            "temperature_range": "-40 C to 125 C",
            "temperature_accuracy": "+/-0.2 C typical, +/-0.3 C max",
            "humidity_range": "0% to 100% RH",
            "humidity_accuracy": "+/-2% RH",
            "resolution": "0.01 C / 0.01 %RH",
            "supply_voltage": "2.4V to 5.5V",
            "supply_current": "2 uA (idle), 800 uA (measuring)",
            "interface": "I2C (up to 1 MHz)",
            "i2c_address": "0x44 (ADDR=GND) or 0x45 (ADDR=VDD)",
            "heater": "Built-in heater for defogging",
            "alert_pin": "Yes",
            "operating_temp": "-40 to 125 C",
            "package": "DFN-8 (2.5x2.5 mm)",
        },
        packages=["DFN-8 (2.5x2.5 mm)"],
        interfaces=["I2C"],
        alternatives=["SHT40 (newer, smaller, cheaper)", "BME280 (adds pressure)", "HDC1080 (TI, similar specs)"],
        lcsc="C78592",
    ),
    Component(
        name="VL53L0X",
        category="sensor",
        manufacturer="STMicroelectronics",
        description="Time-of-flight laser ranging sensor, I2C, up to 2m",
        specs={
            "measurement": "Distance (time-of-flight, 940 nm VCSEL laser)",
            "range": "Up to 2 m (long range mode, lower accuracy)",
            "accuracy": "+/-3% typical",
            "field_of_view": "25 degrees",
            "supply_voltage": "2.6V to 3.5V",
            "supply_current": "19 mA (ranging), 5 uA (standby)",
            "interface": "I2C (up to 400 kHz)",
            "i2c_address": "0x29 (default, reprogrammable at runtime)",
            "sample_rate": "Up to 50 Hz",
            "eye_safety": "Class 1 laser",
            "operating_temp": "-20 to 70 C",
            "package": "Optical LGA-12 (4.4x2.4x1.0 mm)",
        },
        packages=["LGA-12 (with optical window)"],
        interfaces=["I2C"],
        alternatives=["VL53L1X (up to 4m, better)", "VL53L4CD (up to 1.3m, cheaper)", "GP2Y0A21YK (Sharp, analog, no I2C)"],
        lcsc="C91199",
    ),

    # ===== Audio =====
    Component(
        name="MAX98357A",
        category="audio",
        manufacturer="Analog Devices / Maxim",
        description="I2S Class D mono amplifier with built-in DAC, 3.2W",
        specs={
            "output_power": "3.2W into 4 Ohm (at 5V), 1.8W into 8 Ohm",
            "supply_voltage": "2.5V to 5.5V",
            "thd_n": "0.015% at 1W/8Ohm",
            "snr": "93 dB (A-weighted)",
            "sample_rate": "8 to 96 kHz",
            "bit_depth": "16 to 32 bits",
            "interface": "I2S (BCLK, LRCLK, DIN)",
            "gain_select": "3 options via GAIN pin (3/6/9/12/15 dB)",
            "channel_select": "Left, Right, or (L+R)/2 via SD_MODE pin",
            "quiescent_current": "2.4 mA",
            "shutdown_current": "0.01 uA",
            "filterless": "Yes (no output filter needed for most speakers)",
            "operating_temp": "-40 to 85 C",
            "package": "QFN-16 (3x3 mm)",
        },
        packages=["QFN-16 (3x3 mm)"],
        interfaces=["I2S"],
        alternatives=["PAM8403 (analog input, cheaper, lower quality)", "TAS5720 (higher power, I2S)", "SSM2518 (ADI, stereo)"],
        lcsc="C604779",
    ),
    Component(
        name="INMP441",
        category="audio",
        manufacturer="InvenSense / TDK",
        description="I2S MEMS microphone, omnidirectional, 24-bit, low noise",
        specs={
            "type": "MEMS microphone (omnidirectional)",
            "sensitivity": "-26 dBFS (+/-1 dB)",
            "snr": "61 dB (A-weighted)",
            "frequency_response": "60 Hz to 15 kHz",
            "sample_rate": "Set by BCLK (typically 16-48 kHz)",
            "bit_depth": "24-bit",
            "supply_voltage": "1.62V to 3.6V",
            "supply_current": "1.4 mA",
            "interface": "I2S (BCLK, WS, SD)",
            "channel_select": "Left/Right via L/R pin",
            "aop": "120 dB SPL",
            "operating_temp": "-40 to 85 C",
            "package": "LGA (4.72 x 3.76 x 1.0 mm, bottom port)",
        },
        packages=["LGA (bottom port)", "LGA (top port variants)"],
        interfaces=["I2S"],
        alternatives=["SPH0645LM4H (Knowles, I2S)", "ICS-43434 (TDK, I2S, wider freq)", "MSM261S4030H0R (PDM, MEMSensing)"],
        lcsc="C406477",
    ),

    # ===== Motor Drivers =====
    Component(
        name="DRV8833",
        category="motor_driver",
        manufacturer="Texas Instruments",
        description="Dual H-bridge motor driver, 1.5A per channel, 2.7-10.8V",
        specs={
            "channels": "2 H-bridges (or 1 stepper motor)",
            "output_current": "1.5A per channel (2A peak)",
            "supply_voltage": "2.7V to 10.8V",
            "rds_on": "360 mOhm (high-side + low-side total)",
            "control": "IN/IN mode (2 pins per motor) or PHASE/ENABLE mode",
            "pwm_frequency": "Up to 50 kHz",
            "protection": "Overcurrent, thermal shutdown, UVLO",
            "sleep_current": "0.1 uA (nSLEEP pin)",
            "operating_temp": "-40 to 85 C",
            "package": "WSON-10 (3x3 mm)",
        },
        packages=["WSON-10 (3x3 mm)"],
        alternatives=["TB6612FNG (Toshiba, 1.2A, popular)", "A4988 (stepper driver, higher voltage)", "DRV8871 (single channel, 3.6A)"],
        lcsc="C92482",
    ),
    Component(
        name="TMC2209-LA",
        category="motor_driver",
        manufacturer="Trinamic / Analog Devices",
        description="Stepper motor driver, 2A RMS, 256 microstep, UART config, silent (StealthChop)",
        specs={
            "motor_type": "Bipolar stepper",
            "output_current": "2A RMS (2.8A peak) per coil",
            "supply_voltage": "4.75V to 29V",
            "microstep_resolution": "Up to 256 microsteps",
            "interface": "UART (single-wire) + STEP/DIR",
            "stealthchop": "Yes (silent operation, PWM chopper)",
            "spreadcycle": "Yes (precise control at higher speeds)",
            "stallguard": "Yes (sensorless homing)",
            "coolstep": "Yes (adaptive current reduction)",
            "rds_on": "170 mOhm (high-side + low-side)",
            "operating_temp": "-40 to 125 C",
            "package": "QFN-28 (5x5 mm)",
        },
        packages=["QFN-28 (5x5 mm)", "HTSSOP-28 (on breakout modules)"],
        interfaces=["UART", "STEP/DIR"],
        alternatives=["A4988 (simpler, cheaper, noisy)", "DRV8825 (TI, 2.5A, no silent mode)", "TMC2226 (lower cost TMC)"],
        lcsc="C100103",
    ),

    # ===== Miscellaneous =====
    Component(
        name="TXS0108E",
        category="level_shifter",
        manufacturer="Texas Instruments",
        description="8-bit bidirectional voltage level translator, auto-direction sensing",
        specs={
            "channels": "8 bidirectional",
            "port_a_voltage": "1.2V to 3.6V",
            "port_b_voltage": "1.65V to 5.5V",
            "data_rate": "110 Mbps (push-pull), 1.2 Mbps (open-drain)",
            "output_type": "Push-pull (NOT for I2C — use BSS138 or PCA9306 for I2C)",
            "enable_pin": "Yes (active high, disables outputs when low)",
            "supply_current": "40 uA (no load)",
            "operating_temp": "-40 to 85 C",
            "package": "TSSOP-20",
        },
        packages=["TSSOP-20"],
        alternatives=["SN74LVC8T245 (direction-controlled, reliable)", "BSS138 (for I2C, 1-ch)", "PCA9306 (I2C specific, 2-ch)"],
        lcsc="C17206",
        notes="NOT suitable for I2C (open-drain protocol). For I2C level shifting, use BSS138+pull-ups or PCA9306.",
    ),
    Component(
        name="74HC595",
        category="logic",
        manufacturer="NXP / TI / various",
        description="8-bit shift register with output latch, serial-in parallel-out",
        specs={
            "type": "SIPO shift register with output latch",
            "outputs": "8 parallel (active high, push-pull)",
            "supply_voltage": "2V to 6V",
            "output_current": "35 mA per output (at 5V)",
            "clock_frequency": "Up to 25 MHz (at 4.5V)",
            "interface": "SPI-like (SER, SRCLK, RCLK, nOE)",
            "cascade": "QH' (serial out) for daisy-chaining",
            "operating_temp": "-40 to 125 C",
            "package": "SOIC-16",
        },
        packages=["SOIC-16", "DIP-16", "TSSOP-16"],
        alternatives=["74HC164 (no output latch)", "MCP23S17 (I2C/SPI GPIO expander, 16-bit)", "PCA9685 (I2C PWM, 16-ch)"],
        lcsc="C5947",
    ),
    # ===== Additional MCUs =====
    Component(
        name="STM32F401CCU6",
        category="mcu",
        manufacturer="STMicroelectronics",
        description="ARM Cortex-M4 MCU, 84 MHz, 256 KB Flash, 64 KB SRAM, FPU",
        specs={
            "core": "ARM Cortex-M4F (single-precision FPU)",
            "max_frequency": "84 MHz",
            "flash": "256 KB",
            "sram": "64 KB",
            "supply_voltage": "1.7V to 3.6V",
            "gpio_count": "36",
            "adc_channels": "1x 12-bit ADC, 10 channels",
            "timers": "6 (1x 32-bit, 5x 16-bit)",
            "uart": "3 (2 USART + 1 UART)",
            "spi": "3",
            "i2c": "3",
            "usb": "USB 2.0 OTG Full-speed",
            "operating_temp": "-40 to 85 C",
            "package": "UFQFPN-48",
        },
        packages=["UFQFPN-48", "WLCSP-49"],
        interfaces=["SPI", "I2C", "UART", "USB OTG", "I2S"],
        alternatives=["STM32F411CEU6", "GD32F303CCT6", "APM32F407VGT6"],
        lcsc="C428058",
    ),
    Component(
        name="STM32F411CEU6",
        category="mcu",
        manufacturer="STMicroelectronics",
        description="ARM Cortex-M4 MCU, 100 MHz, 512 KB Flash, 128 KB SRAM, FPU",
        specs={
            "core": "ARM Cortex-M4F (single-precision FPU)",
            "max_frequency": "100 MHz",
            "flash": "512 KB",
            "sram": "128 KB",
            "supply_voltage": "1.7V to 3.6V",
            "gpio_count": "36",
            "adc_channels": "1x 12-bit ADC, 10 channels",
            "timers": "6 (1x 32-bit, 5x 16-bit)",
            "uart": "3",
            "spi": "5 (I2S capable)",
            "i2c": "3",
            "usb": "USB 2.0 OTG Full-speed",
            "operating_temp": "-40 to 85 C",
            "package": "UFQFPN-48",
        },
        packages=["UFQFPN-48", "WLCSP-49"],
        interfaces=["SPI", "I2C", "UART", "USB OTG", "I2S"],
        alternatives=["STM32F401CCU6", "WeAct Studio BlackPill (dev board)"],
        lcsc="C459792",
    ),
    Component(
        name="STM32L031K6T6",
        category="mcu",
        manufacturer="STMicroelectronics",
        description="ARM Cortex-M0+ ultra-low-power MCU, 32 MHz, 32 KB Flash, 8 KB SRAM",
        specs={
            "core": "ARM Cortex-M0+",
            "max_frequency": "32 MHz",
            "flash": "32 KB",
            "sram": "8 KB",
            "supply_voltage": "1.65V to 3.6V",
            "gpio_count": "25",
            "adc_channels": "1x 12-bit ADC, 10 channels",
            "timers": "5",
            "uart": "2 (1 USART + 1 LPUART)",
            "spi": "1",
            "i2c": "1",
            "supply_current": "76 uA/MHz (run), 0.29 uA (stop mode)",
            "operating_temp": "-40 to 85 C",
            "package": "LQFP-32",
        },
        packages=["LQFP-32"],
        interfaces=["SPI", "I2C", "UART", "LPUART"],
        alternatives=["STM32L011K4T6", "ATSAML10", "nRF52810"],
        lcsc="C94690",
    ),
    Component(
        name="STM32F030F4P6",
        category="mcu",
        manufacturer="STMicroelectronics",
        description="ARM Cortex-M0 MCU, 48 MHz, 16 KB Flash, 4 KB SRAM, ultra-cheap",
        specs={
            "core": "ARM Cortex-M0",
            "max_frequency": "48 MHz",
            "flash": "16 KB",
            "sram": "4 KB",
            "supply_voltage": "2.4V to 3.6V",
            "gpio_count": "15",
            "adc_channels": "1x 12-bit ADC, 11 channels",
            "timers": "5",
            "uart": "1",
            "spi": "1",
            "i2c": "1",
            "operating_temp": "-40 to 85 C",
            "package": "TSSOP-20",
        },
        packages=["TSSOP-20"],
        interfaces=["SPI", "I2C", "UART"],
        alternatives=["PY32F002AF15P6TU", "CH32V003", "STM32C011F4P6"],
        lcsc="C23922",
    ),
    Component(
        name="ESP32-C3-MINI-1-N4",
        category="mcu",
        manufacturer="Espressif",
        description="RISC-V single-core MCU module, 160 MHz, Wi-Fi + BLE 5, 4 MB Flash",
        specs={
            "core": "RISC-V 32-bit single-core",
            "max_frequency": "160 MHz",
            "flash": "4 MB (in-package)",
            "sram": "400 KB",
            "supply_voltage": "3.0V to 3.6V",
            "gpio_count": "22",
            "adc_channels": "2x 12-bit SAR ADC, 6 channels",
            "timers": "2x 54-bit general-purpose, 3x watchdog",
            "uart": "2",
            "spi": "3",
            "i2c": "1",
            "wifi": "802.11 b/g/n, 2.4 GHz",
            "bluetooth": "BLE 5.0",
            "operating_temp": "-40 to 85 C",
            "package": "Module (13x16.6mm)",
        },
        packages=["Module 13x16.6mm"],
        interfaces=["SPI", "I2C", "UART", "Wi-Fi", "BLE 5.0"],
        alternatives=["ESP32-C6 (Wi-Fi 6)", "ESP32-C2 (cheaper, less RAM)", "nRF52840 (BLE only)"],
        lcsc="C2934560",
    ),
    Component(
        name="ESP32-C6-WROOM-1-N8",
        category="mcu",
        manufacturer="Espressif",
        description="RISC-V dual-core MCU module, 160 MHz, Wi-Fi 6 + BLE 5 + 802.15.4, 8 MB Flash",
        specs={
            "core": "RISC-V 32-bit (HP core 160 MHz + LP core 20 MHz)",
            "max_frequency": "160 MHz",
            "flash": "8 MB (in-package)",
            "sram": "512 KB HP + 16 KB LP",
            "supply_voltage": "3.0V to 3.6V",
            "gpio_count": "23",
            "adc_channels": "1x 12-bit SAR ADC, 7 channels",
            "uart": "2",
            "spi": "1",
            "i2c": "2",
            "wifi": "802.11ax (Wi-Fi 6), 2.4 GHz",
            "bluetooth": "BLE 5.3",
            "thread": "IEEE 802.15.4 (Thread/Zigbee)",
            "operating_temp": "-40 to 85 C",
            "package": "Module (18x20mm)",
        },
        packages=["Module 18x20mm"],
        interfaces=["SPI", "I2C", "UART", "Wi-Fi 6", "BLE 5.3", "Thread/Zigbee"],
        alternatives=["ESP32-C3 (Wi-Fi 4, cheaper)", "ESP32-H2 (Thread only, no Wi-Fi)"],
        lcsc="C5361862",
    ),
    Component(
        name="ATmega32U4-AU",
        category="mcu",
        manufacturer="Microchip (Atmel)",
        description="AVR 8-bit MCU, 16 MHz, 32 KB Flash, 2.5 KB SRAM, native USB",
        specs={
            "core": "AVR 8-bit",
            "max_frequency": "16 MHz",
            "flash": "32 KB",
            "sram": "2.5 KB",
            "eeprom": "1 KB",
            "supply_voltage": "2.7V to 5.5V",
            "gpio_count": "26",
            "adc_channels": "12 (10-bit)",
            "timers": "4 (1x 8-bit, 1x 16-bit, 1x 10-bit high-speed)",
            "uart": "1",
            "spi": "1",
            "i2c": "1 (TWI)",
            "usb": "USB 2.0 Full-speed (native, no external chip)",
            "operating_temp": "-40 to 85 C",
            "package": "TQFP-44",
        },
        packages=["TQFP-44", "QFN-44"],
        interfaces=["SPI", "I2C", "UART", "USB"],
        alternatives=["ATmega328P (no USB)", "RP2040 (more powerful)", "CH552G (cheaper USB)"],
        lcsc="C44854",
    ),
    Component(
        name="STM32G474RET6",
        category="mcu",
        manufacturer="STMicroelectronics",
        description="ARM Cortex-M4F MCU, 170 MHz, 512 KB Flash, 128 KB SRAM, motor control",
        specs={
            "core": "ARM Cortex-M4F with FPU and DSP",
            "max_frequency": "170 MHz",
            "flash": "512 KB",
            "sram": "128 KB",
            "supply_voltage": "1.71V to 3.6V",
            "gpio_count": "51",
            "adc_channels": "5x 12-bit ADC, up to 42 channels",
            "dac": "4x 12-bit DAC",
            "timers": "17 timers (including advanced motor control)",
            "uart": "5",
            "spi": "4",
            "i2c": "4",
            "can": "3x FDCAN",
            "operating_temp": "-40 to 125 C",
            "package": "LQFP-64",
        },
        packages=["LQFP-64", "LQFP-48"],
        interfaces=["SPI", "I2C", "UART", "FDCAN", "USB"],
        alternatives=["STM32F446RE", "STM32H503RB"],
        lcsc="C1329825",
    ),
    Component(
        name="MSP430FR2355",
        category="mcu",
        manufacturer="Texas Instruments",
        description="16-bit ultra-low-power MCU, 24 MHz, 32 KB FRAM, 4 KB SRAM",
        specs={
            "core": "MSP430 16-bit RISC",
            "max_frequency": "24 MHz",
            "flash": "32 KB FRAM (non-volatile, 10^15 write cycles)",
            "sram": "4 KB",
            "supply_voltage": "1.8V to 3.6V",
            "gpio_count": "32",
            "adc_channels": "1x 12-bit SAR ADC, 16 channels",
            "dac": "2x 12-bit DAC",
            "timers": "4 (16-bit)",
            "uart": "2 (eUSCI)",
            "spi": "2 (eUSCI)",
            "i2c": "2 (eUSCI)",
            "supply_current": "118 uA/MHz (active), 0.35 uA (standby)",
            "operating_temp": "-40 to 105 C",
            "package": "TSSOP-38",
        },
        packages=["TSSOP-38", "QFN-40"],
        interfaces=["SPI", "I2C", "UART"],
        alternatives=["STM32L031", "ATSAML10", "RL78/G14"],
        lcsc="C525287",
    ),
    Component(
        name="ATSAME51J20A",
        category="mcu",
        manufacturer="Microchip",
        description="ARM Cortex-M4F MCU, 120 MHz, 1 MB Flash, 256 KB SRAM, CAN-FD",
        specs={
            "core": "ARM Cortex-M4F with FPU",
            "max_frequency": "120 MHz",
            "flash": "1 MB",
            "sram": "256 KB",
            "supply_voltage": "1.71V to 3.63V",
            "gpio_count": "51",
            "adc_channels": "2x 12-bit ADC, 16 channels",
            "dac": "2x 12-bit DAC",
            "timers": "8 (TC/TCC)",
            "uart": "8 (SERCOM)",
            "spi": "8 (SERCOM)",
            "i2c": "8 (SERCOM)",
            "can": "2x CAN-FD",
            "operating_temp": "-40 to 85 C",
            "package": "TQFP-64",
        },
        packages=["TQFP-64", "QFN-64"],
        interfaces=["SPI", "I2C", "UART", "CAN-FD", "USB"],
        alternatives=["STM32G474", "STM32H503", "LPC55S69"],
        lcsc="C648277",
    ),
    Component(
        name="GD32VF103CBT6",
        category="mcu",
        manufacturer="GigaDevice",
        description="RISC-V MCU, 108 MHz, 128 KB Flash, 32 KB SRAM",
        specs={
            "core": "Bumblebee RISC-V (RV32IMAC)",
            "max_frequency": "108 MHz",
            "flash": "128 KB",
            "sram": "32 KB",
            "supply_voltage": "2.6V to 3.6V",
            "gpio_count": "37",
            "adc_channels": "2x 12-bit ADC, 10 channels",
            "timers": "5",
            "uart": "5 (3 USART + 2 UART)",
            "spi": "3",
            "i2c": "2",
            "usb": "USB 2.0 Full-speed OTG",
            "can": "2",
            "operating_temp": "-40 to 85 C",
            "package": "LQFP-48",
        },
        packages=["LQFP-48"],
        interfaces=["SPI", "I2C", "UART", "USB", "CAN"],
        alternatives=["STM32F103C8T6", "CH32V303CBT6"],
        lcsc="C1331940",
    ),
    Component(
        name="W806",
        category="mcu",
        manufacturer="WinnerMicro",
        description="XT804 32-bit MCU, 240 MHz, 1 MB Flash, 288 KB SRAM, ultra-cheap",
        specs={
            "core": "CK804 (T-Head XT804) 32-bit",
            "max_frequency": "240 MHz",
            "flash": "1 MB",
            "sram": "288 KB",
            "supply_voltage": "3.3V (typ)",
            "gpio_count": "44",
            "adc_channels": "4 (16-bit sigma-delta)",
            "uart": "6",
            "spi": "2 (including QSPI)",
            "i2c": "1",
            "operating_temp": "-40 to 85 C",
            "package": "QFN-56",
        },
        packages=["QFN-56"],
        interfaces=["SPI", "I2C", "UART", "SDIO", "PSRAM"],
        alternatives=["ESP32-C3", "CH32V307"],
        lcsc="C2759865",
    ),
    # ===== Additional Regulators =====
    Component(
        name="ME6211C33M5G-N",
        category="regulator",
        manufacturer="Microne",
        description="600mA LDO, 3.3V fixed output, ultra-low dropout",
        specs={
            "output_voltage": "3.3V (fixed)",
            "max_output_current": "600 mA",
            "input_voltage": "2.0V to 6.0V",
            "dropout_voltage": "100 mV @ 100 mA",
            "quiescent_current": "40 uA",
            "output_noise": "45 uVrms",
            "operating_temp": "-40 to 85 C",
            "package": "SOT-23-5",
        },
        packages=["SOT-23-5"],
        alternatives=["AP2112K-3.3", "XC6220B331MR", "RT9013-33"],
        lcsc="C82942",
    ),
    Component(
        name="RT9013-33GB",
        category="regulator",
        manufacturer="Richtek",
        description="500mA LDO, 3.3V, ultra-low noise, SOT-23-5",
        specs={
            "output_voltage": "3.3V (fixed)",
            "max_output_current": "500 mA",
            "input_voltage": "2.2V to 5.5V",
            "dropout_voltage": "250 mV @ 500 mA",
            "quiescent_current": "25 uA",
            "output_noise": "15 uVrms (typ)",
            "operating_temp": "-40 to 85 C",
            "package": "SOT-23-5",
        },
        packages=["SOT-23-5"],
        alternatives=["ME6211C33", "AP2112K-3.3", "MIC5504-3.3YM5"],
        lcsc="C47773",
    ),
    Component(
        name="MP1584EN-LF-Z",
        category="regulator",
        manufacturer="Monolithic Power Systems",
        description="3A step-down converter, 28V input, 1.5 MHz switching frequency",
        specs={
            "output_voltage": "0.8V to VIN (adjustable)",
            "max_output_current": "3 A",
            "input_voltage": "4.5V to 28V",
            "switching_frequency": "1.5 MHz (fixed)",
            "efficiency": "Up to 92%",
            "quiescent_current": "100 uA (typ)",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-8 exposed pad",
        },
        packages=["SOIC-8 exposed pad"],
        alternatives=["TPS54331", "LM2596", "LM3150"],
        lcsc="C14259",
    ),
    Component(
        name="TPS63020DSJR",
        category="regulator",
        manufacturer="Texas Instruments",
        description="Buck-boost converter, 4A, 2.4 MHz, single Li-Ion to 3.3V/5V",
        specs={
            "output_voltage": "1.2V to 5.5V (adjustable)",
            "max_output_current": "4 A (buck), 2 A (boost)",
            "input_voltage": "1.8V to 5.5V",
            "switching_frequency": "2.4 MHz (typ)",
            "efficiency": "Up to 96%",
            "quiescent_current": "50 uA",
            "operating_temp": "-40 to 85 C",
            "package": "VSON-14 (3x3mm)",
        },
        packages=["VSON-14 (3x3mm)"],
        alternatives=["TPS63060 (wider input)", "LTC3113 (Analog Devices)"],
        lcsc="C130012",
    ),
    Component(
        name="HT7333-A",
        category="regulator",
        manufacturer="Holtek",
        description="250mA LDO, 3.3V fixed, ultra-low quiescent current, SOT-89",
        specs={
            "output_voltage": "3.3V (fixed)",
            "max_output_current": "250 mA",
            "input_voltage": "3.6V to 12V",
            "dropout_voltage": "300 mV @ 100 mA",
            "quiescent_current": "4 uA (typ)",
            "operating_temp": "-40 to 85 C",
            "package": "SOT-89",
        },
        packages=["SOT-89", "TO-92"],
        alternatives=["MCP1700-3302E", "LP5907-3.3"],
        lcsc="C14289",
    ),
    Component(
        name="MCP1700-3302E/TT",
        category="regulator",
        manufacturer="Microchip",
        description="250mA LDO, 3.3V fixed, low quiescent, SOT-23",
        specs={
            "output_voltage": "3.3V (fixed)",
            "max_output_current": "250 mA",
            "input_voltage": "2.3V to 6.0V",
            "dropout_voltage": "178 mV @ 250 mA",
            "quiescent_current": "1.6 uA (typ)",
            "operating_temp": "-40 to 125 C",
            "package": "SOT-23-3",
        },
        packages=["SOT-23-3", "TO-92"],
        alternatives=["HT7333", "LP5907-3.3"],
        lcsc="C39051",
    ),
    Component(
        name="TPS5430DDA",
        category="regulator",
        manufacturer="Texas Instruments",
        description="3A step-down converter, 36V max input, 500 kHz",
        specs={
            "output_voltage": "1.22V to 30V (adjustable)",
            "max_output_current": "3 A",
            "input_voltage": "5.5V to 36V",
            "switching_frequency": "500 kHz (fixed)",
            "efficiency": "Up to 95%",
            "operating_temp": "-40 to 125 C",
            "package": "HSOP-8 (PowerPAD)",
        },
        packages=["HSOP-8"],
        alternatives=["LM2596", "TPS54331", "LM2576"],
        lcsc="C47283",
    ),
    Component(
        name="SPX3819M5-L-3-3/TR",
        category="regulator",
        manufacturer="MaxLinear (Sipex)",
        description="500mA LDO, 3.3V, 16V max input, high PSRR",
        specs={
            "output_voltage": "3.3V (fixed)",
            "max_output_current": "500 mA",
            "input_voltage": "1.8V to 16V",
            "dropout_voltage": "340 mV @ 500 mA",
            "quiescent_current": "42 uA",
            "operating_temp": "-40 to 125 C",
            "package": "SOT-23-5",
        },
        packages=["SOT-23-5"],
        alternatives=["LM1117-3.3", "AMS1117-3.3", "AP2112K-3.3"],
        lcsc="C9055",
    ),
    # ===== Additional Op-amps =====
    Component(
        name="LM324",
        category="opamp",
        manufacturer="TI / ST / ON Semi",
        description="Quad general-purpose op-amp, single-supply, low power",
        specs={
            "channels": "4 (quad)",
            "supply_voltage": "3V to 32V (single) or +/-1.5V to +/-16V (dual)",
            "gbw": "1.2 MHz",
            "slew_rate": "0.5 V/us",
            "input_offset_voltage": "2 mV (typ), 7 mV (max)",
            "input_bias_current": "45 nA (typ)",
            "supply_current": "0.7 mA per amplifier",
            "output_type": "Rail-to-rail output (low only, not high rail)",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-14",
        },
        packages=["SOIC-14", "DIP-14", "TSSOP-14"],
        alternatives=["LM358 (dual)", "TLV2374 (rail-to-rail)", "MCP6004 (CMOS)"],
        lcsc="C7424",
    ),
    Component(
        name="LMV321IDBVR",
        category="opamp",
        manufacturer="Texas Instruments",
        description="Single RRIO CMOS op-amp, 1 MHz GBW, SOT-23-5",
        specs={
            "channels": "1 (single)",
            "supply_voltage": "2.7V to 5.5V",
            "gbw": "1 MHz",
            "slew_rate": "1 V/us",
            "input_offset_voltage": "7 mV (max)",
            "input_bias_current": "100 pA (typ)",
            "supply_current": "0.11 mA",
            "output_type": "Rail-to-rail input and output",
            "operating_temp": "-40 to 125 C",
            "package": "SOT-23-5",
        },
        packages=["SOT-23-5"],
        alternatives=["MCP6001 (higher GBW)", "AD8605 (lower noise)"],
        lcsc="C7684",
    ),
    Component(
        name="OPA1612AIDR",
        category="opamp",
        manufacturer="Texas Instruments",
        description="Dual ultra-low-noise audio op-amp, 80 MHz GBW, SoundPlus",
        specs={
            "channels": "2 (dual)",
            "supply_voltage": "+/-2.25V to +/-18V",
            "gbw": "80 MHz",
            "slew_rate": "27 V/us",
            "input_offset_voltage": "0.1 mV (typ)",
            "input_bias_current": "10 nA (typ)",
            "supply_current": "3.6 mA per channel",
            "noise": "1.1 nV/rtHz @ 1 kHz",
            "thd": "-136 dB (0.00005%) at 1 kHz, 3 Vrms",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8"],
        alternatives=["OPA2134 (lower cost)", "NE5532 (bipolar, classic)", "LME49720 (NatSemi)"],
        lcsc="C116506",
    ),
    Component(
        name="INA219AIDR",
        category="sensor",
        manufacturer="Texas Instruments",
        description="High-side current/power monitor, I2C, 26V, 12-bit ADC",
        specs={
            "input_voltage": "0V to 26V (bus voltage)",
            "shunt_voltage": "+/-320 mV (max) or +/-160 mV or +/-80 mV or +/-40 mV",
            "resolution": "12-bit ADC",
            "i2c_address": "0x40-0x4F (16 addresses via A0/A1)",
            "supply_voltage": "3.0V to 5.5V",
            "supply_current": "1 mA (max)",
            "accuracy": "+/-1% (gain error)",
            "operating_temp": "-40 to 125 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8", "SOT-23-6 (INA219AIDBVR)"],
        interfaces=["I2C"],
        alternatives=["INA226 (16-bit)", "INA260 (integrated shunt)", "MAX9611"],
        lcsc="C9443",
    ),
    # ===== Additional MOSFETs =====
    Component(
        name="2N7002",
        category="mosfet",
        manufacturer="NXP / ON Semi / various",
        description="60V N-channel MOSFET, 300mA, SOT-23, logic-level gate",
        specs={
            "type": "N-channel enhancement",
            "vds_max": "60V",
            "id_continuous": "300 mA",
            "rds_on": "5 Ohm @ Vgs=10V, 7.5 Ohm @ Vgs=4.5V",
            "vgs_threshold": "1.0V (typ), 2.5V (max)",
            "gate_charge": "750 pC (typ)",
            "power_dissipation": "300 mW",
            "operating_temp": "-55 to 150 C",
            "package": "SOT-23",
        },
        packages=["SOT-23"],
        alternatives=["BSS138 (similar, lower Rds)", "BSS123 (100V, 170mA)"],
        lcsc="C8545",
    ),
    Component(
        name="DMG2305UX-7",
        category="mosfet",
        manufacturer="Diodes Inc.",
        description="-20V P-channel MOSFET, 4.2A, SOT-23, low Rds(on)",
        specs={
            "type": "P-channel enhancement",
            "vds_max": "-20V",
            "id_continuous": "-4.2 A",
            "rds_on": "45 mOhm @ Vgs=-4.5V, 65 mOhm @ Vgs=-2.5V",
            "vgs_threshold": "-0.4V (min), -0.9V (max)",
            "gate_charge": "6.6 nC (typ)",
            "power_dissipation": "1.4 W",
            "operating_temp": "-55 to 150 C",
            "package": "SOT-23",
        },
        packages=["SOT-23"],
        alternatives=["AO3401A (lower Rds)", "Si2301 (lower Vgs_th)"],
        lcsc="C154825",
    ),
    Component(
        name="IRLZ44N",
        category="mosfet",
        manufacturer="Infineon (IR)",
        description="55V N-channel logic-level MOSFET, 47A, TO-220, low Rds(on)",
        specs={
            "type": "N-channel enhancement, logic-level gate",
            "vds_max": "55V",
            "id_continuous": "47 A @ 25 C",
            "rds_on": "22 mOhm @ Vgs=10V, 25 mOhm @ Vgs=5V",
            "vgs_threshold": "1.0V (min), 2.0V (max)",
            "gate_charge": "48 nC",
            "power_dissipation": "110 W (with heatsink)",
            "operating_temp": "-55 to 175 C",
            "package": "TO-220",
        },
        packages=["TO-220"],
        alternatives=["IRF540N (not logic-level)", "IRL540N", "IRLZ34N (60V, 30A)"],
        lcsc="C49632",
    ),
    Component(
        name="Si7021-A20-IM1",
        category="sensor",
        manufacturer="Silicon Labs",
        description="Humidity and temperature sensor, I2C, +/-3% RH, +/-0.4 C",
        specs={
            "humidity_accuracy": "+/-3% RH (20-80% range)",
            "temperature_accuracy": "+/-0.4 C (typical)",
            "humidity_range": "0-100% RH",
            "resolution": "12-bit RH, 14-bit temperature",
            "i2c_address": "0x40 (fixed)",
            "supply_voltage": "1.9V to 3.6V",
            "supply_current": "150 uA (active), 0.06 uA (standby)",
            "operating_temp": "-40 to 125 C",
            "package": "DFN-6 (3x3mm)",
        },
        packages=["DFN-6 (3x3mm)"],
        interfaces=["I2C"],
        alternatives=["SHT31 (higher accuracy)", "BME280 (+ pressure)", "HDC1080 (lower cost)"],
        lcsc="C85046",
    ),
    Component(
        name="ADXL345BCCZ-RL7",
        category="sensor",
        manufacturer="Analog Devices",
        description="3-axis digital accelerometer, +/-16g, I2C/SPI, 13-bit resolution",
        specs={
            "measurement_range": "+/-2g / +/-4g / +/-8g / +/-16g (selectable)",
            "resolution": "Up to 13-bit (4 mg/LSB at +/-2g)",
            "data_rate": "0.1 Hz to 3200 Hz (selectable)",
            "i2c_address": "0x1D (ALT HIGH) or 0x53 (ALT LOW)",
            "supply_voltage": "2.0V to 3.6V",
            "supply_current": "23 uA (100 Hz), 0.1 uA (standby)",
            "operating_temp": "-40 to 85 C",
            "package": "LGA-14 (3x5mm)",
        },
        packages=["LGA-14 (3x5mm)"],
        interfaces=["I2C", "SPI"],
        alternatives=["LIS3DH (STMicro)", "MMA8451Q (NXP)", "MPU6050 (+ gyro)"],
        lcsc="C9667",
    ),
    Component(
        name="BMP390",
        category="sensor",
        manufacturer="Bosch Sensortec",
        description="Barometric pressure sensor, I2C/SPI, +/-0.03 hPa relative accuracy",
        specs={
            "pressure_range": "300 to 1250 hPa",
            "pressure_accuracy": "+/-0.5 hPa (absolute), +/-0.03 hPa (relative)",
            "temperature_accuracy": "+/-0.5 C",
            "resolution": "24-bit pressure, 24-bit temperature",
            "i2c_address": "0x76 (SDO=GND) or 0x77 (SDO=VCC)",
            "supply_voltage": "1.7V to 3.6V",
            "supply_current": "3.4 uA @ 1 Hz, 0.06 uA (standby)",
            "operating_temp": "-40 to 85 C",
            "package": "LGA-10 (2x2mm)",
        },
        packages=["LGA-10 (2x2mm)"],
        interfaces=["I2C", "SPI"],
        alternatives=["BME280 (+ humidity)", "LPS22HB (STMicro)", "DPS310 (Infineon)"],
        lcsc="C2660547",
    ),
    Component(
        name="MAX6675ISA+T",
        category="sensor",
        manufacturer="Analog Devices (Maxim)",
        description="K-type thermocouple-to-digital converter, SPI, 12-bit, 0-1024 C",
        specs={
            "temperature_range": "0 C to 1024 C",
            "temperature_accuracy": "+/-3 C (typ)",
            "resolution": "0.25 C (12-bit + sign)",
            "conversion_time": "220 ms",
            "supply_voltage": "3.0V to 5.5V",
            "supply_current": "1.5 mA",
            "operating_temp": "-20 to 85 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8"],
        interfaces=["SPI"],
        alternatives=["MAX31855 (wider range, -200 to 1800 C)", "MAX31856 (multiple TC types)"],
        lcsc="C9649",
    ),
    Component(
        name="ACS712ELCTR-20A-T",
        category="sensor",
        manufacturer="Allegro MicroSystems",
        description="20A bidirectional Hall-effect current sensor, analog output",
        specs={
            "current_range": "+/-20 A",
            "sensitivity": "100 mV/A",
            "accuracy": "+/-1.5% (at 25 C)",
            "bandwidth": "80 kHz",
            "supply_voltage": "4.5V to 5.5V",
            "supply_current": "10 mA (typ)",
            "isolation_voltage": "2.1 kV (RMS)",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8"],
        interfaces=["Analog output (66 mV/A for 30A, 100 mV/A for 20A, 185 mV/A for 5A)"],
        alternatives=["INA226 (shunt-based, I2C)", "TMCS1108 (TI)", "ACS758 (50A/100A)"],
        lcsc="C10681",
    ),
    Component(
        name="VEML7700",
        category="sensor",
        manufacturer="Vishay",
        description="Ambient light sensor, I2C, 16-bit resolution, 0 to 120k lux",
        specs={
            "measurement_range": "0 to 120,000 lux",
            "resolution": "0.0036 lux/count (min gain, max integration)",
            "i2c_address": "0x10 (fixed)",
            "supply_voltage": "2.5V to 3.6V",
            "supply_current": "2 uA (typ), 0.5 uA (shutdown)",
            "spectral_response": "Close to human eye (CIE photopic curve)",
            "operating_temp": "-25 to 85 C",
            "package": "QFN (2x2mm)",
        },
        packages=["QFN (2x2mm)"],
        interfaces=["I2C"],
        alternatives=["BH1750 (cheaper)", "TSL2591 (wider range)", "APDS-9960 (+ gesture)"],
        lcsc="C2536068",
    ),
    Component(
        name="LIS3DH",
        category="sensor",
        manufacturer="STMicroelectronics",
        description="3-axis MEMS accelerometer, +/-16g, I2C/SPI, ultra-low power",
        specs={
            "measurement_range": "+/-2g / +/-4g / +/-8g / +/-16g (selectable)",
            "resolution": "16-bit / 12-bit / 10-bit / 8-bit (selectable)",
            "data_rate": "1 Hz to 5376 Hz",
            "i2c_address": "0x18 (SA0=GND) or 0x19 (SA0=VCC)",
            "supply_voltage": "1.71V to 3.6V",
            "supply_current": "11 uA (low-power 50 Hz), 2 uA (1 Hz)",
            "operating_temp": "-40 to 85 C",
            "package": "LGA-16 (3x3mm)",
        },
        packages=["LGA-16 (3x3mm)"],
        interfaces=["I2C", "SPI"],
        alternatives=["ADXL345 (Analog Devices)", "MMA8451Q (NXP)", "MPU6050 (+ gyro)"],
        lcsc="C9913",
    ),
    # ===== Additional Passives =====
    Component(
        name="4.7K 0402 Resistor",
        category="passive",
        manufacturer="Yageo / Uniroyal / various",
        description="4.7K ohm chip resistor, 0402, 1/16W, 1% tolerance",
        specs={
            "resistance": "4.7K ohm",
            "tolerance": "1% (F suffix)",
            "power_rating": "1/16 W (0.0625 W)",
            "temperature_coefficient": "+/-100 ppm/C",
            "package": "0402 (1005 metric)",
        },
        packages=["0402", "0603", "0805"],
        lcsc="C25900",
        notes="Standard I2C pull-up value for standard mode (100 kHz).",
    ),
    Component(
        name="100 ohm 0402 Resistor",
        category="passive",
        manufacturer="Yageo / Uniroyal / various",
        description="100 ohm chip resistor, 0402, 1/16W, 1% tolerance",
        specs={
            "resistance": "100 ohm",
            "tolerance": "1%",
            "power_rating": "1/16 W",
            "temperature_coefficient": "+/-100 ppm/C",
            "package": "0402 (1005 metric)",
        },
        packages=["0402", "0603", "0805"],
        lcsc="C25076",
        notes="Common for series termination, current limiting.",
    ),
    Component(
        name="1K 0402 Resistor",
        category="passive",
        manufacturer="Yageo / Uniroyal / various",
        description="1K ohm chip resistor, 0402, 1/16W, 1% tolerance",
        specs={
            "resistance": "1K ohm",
            "tolerance": "1%",
            "power_rating": "1/16 W",
            "temperature_coefficient": "+/-100 ppm/C",
            "package": "0402 (1005 metric)",
        },
        packages=["0402", "0603", "0805"],
        lcsc="C11702",
        notes="LED current limiting (3.3V, 2V Vf: I = 1.3 mA), MOSFET gate pull-down.",
    ),
    Component(
        name="47K 0402 Resistor",
        category="passive",
        manufacturer="Yageo / Uniroyal / various",
        description="47K ohm chip resistor, 0402, 1/16W, 1% tolerance",
        specs={
            "resistance": "47K ohm",
            "tolerance": "1%",
            "power_rating": "1/16 W",
            "temperature_coefficient": "+/-100 ppm/C",
            "package": "0402 (1005 metric)",
        },
        packages=["0402", "0603", "0805"],
        lcsc="C25819",
        notes="Weak pull-up/pull-down for low power applications, voltage dividers.",
    ),
    Component(
        name="5.1K 0402 Resistor",
        category="passive",
        manufacturer="Yageo / Uniroyal / various",
        description="5.1K ohm chip resistor, 0402, 1/16W, 1% tolerance",
        specs={
            "resistance": "5.1K ohm",
            "tolerance": "1%",
            "power_rating": "1/16 W",
            "temperature_coefficient": "+/-100 ppm/C",
            "package": "0402 (1005 metric)",
        },
        packages=["0402", "0603", "0805"],
        lcsc="C25905",
        notes="Required USB-C CC1/CC2 pull-down resistor for device (UFP) identification.",
    ),
    Component(
        name="22pF 0402 C0G Capacitor",
        category="passive",
        manufacturer="Samsung / Murata / various",
        description="22pF ceramic capacitor, 0402, C0G/NP0 dielectric, 50V",
        specs={
            "capacitance": "22 pF",
            "tolerance": "+/-5% (J code)",
            "voltage_rating": "50V",
            "dielectric": "C0G (NP0) — Class I, ultra-stable",
            "temperature_coefficient": "+/-30 ppm/C",
            "package": "0402 (1005 metric)",
        },
        packages=["0402", "0603"],
        lcsc="C1555",
        notes="Crystal load capacitor for 8/16 MHz crystals. C0G has no voltage derating.",
    ),
    Component(
        name="1uF 0402 X5R Capacitor",
        category="passive",
        manufacturer="Samsung / Murata / various",
        description="1uF ceramic capacitor, 0402, X5R dielectric, 16V",
        specs={
            "capacitance": "1 uF",
            "tolerance": "+/-20% (M code)",
            "voltage_rating": "16V",
            "dielectric": "X5R — Class II, good stability",
            "package": "0402 (1005 metric)",
        },
        packages=["0402", "0603", "0805"],
        lcsc="C52923",
        notes="VDDA filtering, charge pump, local bypass for sensitive analog ICs.",
    ),
    Component(
        name="22uF 0805 X5R Capacitor",
        category="passive",
        manufacturer="Samsung / Murata / various",
        description="22uF ceramic capacitor, 0805, X5R dielectric, 6.3V",
        specs={
            "capacitance": "22 uF",
            "tolerance": "+/-20% (M code)",
            "voltage_rating": "6.3V",
            "dielectric": "X5R — Class II",
            "package": "0805 (2012 metric)",
        },
        packages=["0805", "1206"],
        lcsc="C45783",
        notes="USB VBUS decoupling, LDO output cap, bulk energy storage. DC bias derating applies.",
    ),
    Component(
        name="10uH Inductor CDRH6D28",
        category="passive",
        manufacturer="Sumida / Wurth / various",
        description="10uH shielded power inductor, 3A saturation, 50 mOhm DCR",
        specs={
            "inductance": "10 uH",
            "saturation_current": "3 A",
            "dcr": "50 mOhm",
            "tolerance": "+/-20%",
            "package": "6.3x6.3x3mm (SMD shielded)",
        },
        packages=["6.3x6.3x3mm"],
        lcsc="C281149",
        notes="Suitable for TPS54331, MP1584 buck converter designs.",
    ),
    # ===== Additional Interface ICs =====
    Component(
        name="CP2102N-A02-GQFN24",
        category="interface",
        manufacturer="Silicon Labs",
        description="USB-to-UART bridge, CP2102N, up to 3 Mbps, QFN-24",
        specs={
            "interface": "USB 2.0 Full-speed to UART",
            "baud_rate": "Up to 3 Mbps",
            "data_bits": "5, 6, 7, 8",
            "gpio_count": "7 (multi-function)",
            "supply_voltage": "3.0V to 3.6V (or USB bus powered)",
            "supply_current": "10 mA (active), 100 uA (suspend)",
            "operating_temp": "-40 to 85 C",
            "package": "QFN-24 (4x4mm)",
        },
        packages=["QFN-24 (4x4mm)", "QFN-28 (5x5mm)"],
        interfaces=["USB", "UART"],
        alternatives=["CH340G (cheaper)", "FT232RL (FTDI, higher cost)", "PL2303 (Prolific)"],
        lcsc="C97490",
    ),
    Component(
        name="SN65HVD230DR",
        category="interface",
        manufacturer="Texas Instruments",
        description="CAN bus transceiver, 1 Mbps, 3.3V, SOIC-8",
        specs={
            "protocol": "CAN 2.0 (ISO 11898-2)",
            "data_rate": "Up to 1 Mbps",
            "supply_voltage": "3.0V to 3.6V",
            "standby_current": "370 uA",
            "bus_fault_protection": "+/-12V",
            "nodes_on_bus": "Up to 120",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8"],
        interfaces=["CAN"],
        alternatives=["MCP2551 (5V version)", "TJA1050 (NXP)", "VP230 (TI, lower cost)"],
        lcsc="C12084",
    ),
    Component(
        name="W5500",
        category="interface",
        manufacturer="WIZnet",
        description="Hardwired TCP/IP Ethernet controller, SPI, 10/100 Mbps, 8 sockets",
        specs={
            "protocol": "TCP, UDP, IPv4, ICMP, ARP, IGMP, PPPoE",
            "sockets": "8 simultaneous",
            "data_rate": "10/100 Mbps Ethernet",
            "spi_clock": "Up to 80 MHz",
            "buffer": "32 KB TX + 32 KB RX (configurable per socket)",
            "supply_voltage": "3.3V",
            "supply_current": "132 mA (active)",
            "operating_temp": "-40 to 85 C",
            "package": "LQFP-48",
        },
        packages=["LQFP-48", "QFN-48"],
        interfaces=["SPI", "Ethernet"],
        alternatives=["ENC28J60 (slower, cheaper)", "LAN8720 (PHY only, needs MAC)", "ESP32 (Wi-Fi)"],
        lcsc="C32843",
    ),
    Component(
        name="MAX485ESA+T",
        category="interface",
        manufacturer="Analog Devices (Maxim)",
        description="RS-485/RS-422 transceiver, half-duplex, 2.5 Mbps, 5V",
        specs={
            "protocol": "RS-485 / RS-422",
            "data_rate": "Up to 2.5 Mbps",
            "supply_voltage": "4.75V to 5.25V",
            "supply_current": "0.3 mA (shutdown), 0.9 mA (active)",
            "nodes_on_bus": "Up to 32 unit loads",
            "bus_fault_protection": "-7V to +12V",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8", "DIP-8"],
        interfaces=["RS-485"],
        alternatives=["SP3485 (3.3V)", "MAX3485 (3.3V)", "ISL3178 (wider fault protection)"],
        lcsc="C9012",
    ),
    # ===== Additional Protection =====
    Component(
        name="PRTR5V0U2X",
        category="protection",
        manufacturer="NXP",
        description="Ultra-low capacitance double ESD protection for USB 2.0 data lines",
        specs={
            "esd_rating": "+/-8 kV (contact), +/-15 kV (air gap)",
            "clamping_voltage": "5.5V @ 1A (TLP)",
            "line_capacitance": "0.35 pF (typ)",
            "leakage_current": "50 nA (max)",
            "standoff_voltage": "5.5V",
            "operating_temp": "-40 to 85 C",
            "package": "SOT-143B",
        },
        packages=["SOT-143B"],
        alternatives=["USBLC6-2SC6", "TPD2E009 (TI)", "IP4220CZ6 (NXP)"],
        lcsc="C12332",
    ),
    Component(
        name="SRV05-4-P-T7",
        category="protection",
        manufacturer="ProTek Devices",
        description="5-line ESD protection, USB 2.0 and I/O, ultra-low capacitance",
        specs={
            "esd_rating": "+/-30 kV (air gap), +/-25 kV (contact)",
            "clamping_voltage": "16V @ 8A",
            "line_capacitance": "0.85 pF (typ)",
            "standoff_voltage": "5V",
            "lines": "4 I/O + 1 VCC",
            "operating_temp": "-55 to 85 C",
            "package": "SOT-23-6",
        },
        packages=["SOT-23-6"],
        alternatives=["USBLC6-2SC6 (lower cap)", "TPD4E05U06 (TI, 4-line)"],
        lcsc="C85364",
    ),
    # ===== Power Management =====
    Component(
        name="BQ24075RGTR",
        category="charger",
        manufacturer="Texas Instruments",
        description="Li-Ion/Li-Po charger, USB-compatible, 1.5A, power path, QFN-16",
        specs={
            "chemistry": "Li-Ion / Li-Po (single cell 4.2V)",
            "charge_current": "Up to 1.5 A (programmable via ISET resistor)",
            "input_voltage": "4.35V to 6.5V (USB compatible)",
            "charge_accuracy": "+/-0.5% (voltage)",
            "power_path": "Yes (dynamic PPM, charges battery while powering system)",
            "precharge_current": "10% of fast charge",
            "termination_current": "10% of fast charge",
            "supply_current": "1.4 mA (charging), 15 uA (standby)",
            "operating_temp": "-40 to 85 C",
            "package": "VQFN-16 (3.5x3.5mm)",
        },
        packages=["VQFN-16 (3.5x3.5mm)"],
        alternatives=["MCP73871 (Microchip, power path)", "TP4056 (simpler, no power path)", "BQ25895 (buck-boost)"],
        lcsc="C14879",
    ),
    Component(
        name="TPS2113ADRBR",
        category="power",
        manufacturer="Texas Instruments",
        description="Auto-switching power mux, 2 inputs, 3A, break-before-make, SOT-23-6",
        specs={
            "input_voltage": "2.8V to 5.5V (each input)",
            "max_output_current": "3 A (continuous)",
            "rds_on": "70 mOhm (per switch)",
            "switchover_time": "<10 us (break-before-make)",
            "quiescent_current": "55 uA",
            "priority": "Configurable (IN1 preferred or auto-select highest)",
            "operating_temp": "-40 to 125 C",
            "package": "SOT-23-6",
        },
        packages=["SOT-23-6"],
        alternatives=["TPS2115A (auto-select)", "LTC4414 (power path controller)", "STMPS2141 (load switch)"],
        lcsc="C151361",
    ),
    # ===== Motor Drivers =====
    Component(
        name="A4988SETTR-T",
        category="motor_driver",
        manufacturer="Allegro MicroSystems",
        description="Bipolar stepper motor driver, 2A, microstepping, QFN-28",
        specs={
            "motor_type": "Bipolar stepper",
            "output_current": "2 A (peak), 1 A (RMS per phase without heatsink)",
            "supply_voltage": "8V to 35V (motor supply)",
            "logic_voltage": "3.0V to 5.5V",
            "microstepping": "Full, 1/2, 1/4, 1/8, 1/16",
            "rds_on": "0.32 Ohm (high-side + low-side)",
            "protection": "Overcurrent, thermal shutdown, undervoltage lockout",
            "operating_temp": "-20 to 85 C",
            "package": "QFN-28 (5x5mm)",
        },
        packages=["QFN-28 (5x5mm)"],
        alternatives=["TMC2209 (quieter, StealthChop)", "DRV8825 (2.5A, 1/32 step)", "TMC2130 (SPI config)"],
        lcsc="C89847",
    ),
    Component(
        name="L298N",
        category="motor_driver",
        manufacturer="STMicroelectronics",
        description="Dual full-bridge motor driver, 2A per bridge, 46V",
        specs={
            "motor_type": "DC motor (dual) or bipolar stepper (single)",
            "output_current": "2 A per bridge (3 A peak)",
            "supply_voltage": "4.5V to 46V (motor supply)",
            "logic_voltage": "5V",
            "saturation_drop": "1.7V (per transistor, total ~3.4V H-bridge)",
            "protection": "Internal clamp diodes (but external freewheeling diodes recommended)",
            "operating_temp": "-25 to 130 C",
            "package": "Multiwatt-15 (through-hole)",
        },
        packages=["Multiwatt-15"],
        alternatives=["DRV8833 (lower dropout, SMD)", "TB6612FNG (MOSFET, 1.2A)", "L9110S (cheaper, lower current)"],
        lcsc="C10408",
    ),
    # ===== Wireless/RF =====
    Component(
        name="nRF24L01+",
        category="rf",
        manufacturer="Nordic Semiconductor",
        description="2.4 GHz transceiver, SPI, 2 Mbps, ultra-low power",
        specs={
            "frequency": "2.4 GHz ISM band",
            "data_rate": "250 kbps / 1 Mbps / 2 Mbps",
            "output_power": "0 dBm (max)",
            "sensitivity": "-94 dBm (at 250 kbps)",
            "supply_voltage": "1.9V to 3.6V (5V tolerant I/O)",
            "supply_current": "11.3 mA (TX 0 dBm), 13.5 mA (RX), 900 nA (power-down)",
            "modulation": "GFSK",
            "range": "~100m (line of sight, 250 kbps)",
            "operating_temp": "-40 to 85 C",
            "package": "QFN-20 (4x4mm)",
        },
        packages=["QFN-20 (4x4mm)", "Module (PCB antenna)"],
        interfaces=["SPI"],
        alternatives=["CC2500 (TI, 2.4 GHz)", "SX1276 (LoRa, longer range)", "ESP32 (Wi-Fi/BLE)"],
        lcsc="C11537",
    ),
    Component(
        name="RFM95W-868S2",
        category="rf",
        manufacturer="HopeRF",
        description="LoRa transceiver module, 868 MHz, +20 dBm, SPI, based on SX1276",
        specs={
            "frequency": "868 MHz (EU ISM band)",
            "modulation": "LoRa (CSS) + FSK + OOK",
            "output_power": "+20 dBm (max, 100 mW)",
            "sensitivity": "-148 dBm (LoRa, SF12, BW 7.8 kHz)",
            "data_rate": "0.018 to 37.5 kbps (LoRa), up to 300 kbps (FSK)",
            "supply_voltage": "1.8V to 3.7V",
            "supply_current": "120 mA (TX +20 dBm), 12 mA (RX), 0.2 uA (sleep)",
            "range": "2-15 km (LoRa, line of sight)",
            "operating_temp": "-40 to 85 C",
            "package": "Module (16x16mm)",
        },
        packages=["Module (16x16mm)"],
        interfaces=["SPI"],
        alternatives=["SX1276 (bare chip)", "RFM69HCW (FSK only, lower cost)", "E22-900T22S (UART module)"],
        lcsc="C84806",
    ),
    # ===== Display ICs =====
    Component(
        name="ST7789V",
        category="display",
        manufacturer="Sitronix",
        description="TFT LCD controller, 240x320, 262K colors, SPI/Parallel",
        specs={
            "resolution": "240 x 320 pixels",
            "color_depth": "262K colors (18-bit), 65K (16-bit)",
            "interface": "SPI (4-wire) or 8/9/16/18-bit parallel",
            "spi_clock": "Up to 62.5 MHz (SPI write)",
            "supply_voltage": "2.4V to 3.3V (I/O), 2.6V to 3.6V (analog)",
            "supply_current": "2 mA (standby), ~15 mA (display on)",
            "operating_temp": "-30 to 70 C",
            "package": "On display module",
        },
        packages=["On module (commonly 1.3/1.54/2.0 inch TFT)"],
        interfaces=["SPI", "Parallel"],
        alternatives=["ILI9341 (320x240)", "ST7735 (128x160, smaller)", "GC9A01 (round, 240x240)"],
        lcsc="C91437",
    ),
    Component(
        name="ILI9341",
        category="display",
        manufacturer="ILITEK",
        description="TFT LCD controller, 240x320, 262K colors, SPI/Parallel, with GRAM",
        specs={
            "resolution": "240 x 320 pixels (QVGA)",
            "color_depth": "262K colors (18-bit)",
            "interface": "SPI (serial) or 8/16/18-bit parallel",
            "spi_clock": "Up to 10 MHz (SPI read), ~50 MHz (SPI write typical)",
            "gram": "172,800 bytes internal display RAM",
            "supply_voltage": "1.65V to 3.3V (I/O)",
            "operating_temp": "-30 to 70 C",
            "package": "On display module",
        },
        packages=["On module (commonly 2.4/2.8/3.2 inch TFT)"],
        interfaces=["SPI", "Parallel"],
        alternatives=["ST7789V (newer, faster SPI)", "HX8357 (480x320)", "RM68140"],
        lcsc="C50169",
    ),
    # ===== Audio ICs =====
    Component(
        name="PCM5102A",
        category="audio",
        manufacturer="Texas Instruments",
        description="32-bit stereo DAC, I2S, 112 dB SNR, no external components needed",
        specs={
            "resolution": "32-bit",
            "sample_rate": "8 kHz to 384 kHz",
            "snr": "112 dB (A-weighted)",
            "thd": "-93 dB (typ)",
            "dynamic_range": "112 dB",
            "supply_voltage": "3.3V (digital and analog)",
            "supply_current": "17 mA (typ)",
            "output": "Line-level stereo (2.1 Vrms)",
            "operating_temp": "-25 to 85 C",
            "package": "TSSOP-20",
        },
        packages=["TSSOP-20"],
        interfaces=["I2S"],
        alternatives=["ES9018K2M (higher end)", "WM8960 (+ ADC + headphone amp)", "MAX98357 (I2S speaker amp)"],
        lcsc="C39838",
    ),
    Component(
        name="PAM8403",
        category="audio",
        manufacturer="Diodes Inc.",
        description="3W stereo Class-D audio amplifier, filterless, 5V",
        specs={
            "channels": "2 (stereo)",
            "output_power": "3W per channel @ 4 ohm, 5V",
            "supply_voltage": "2.5V to 5.5V",
            "supply_current": "10 mA (idle), no-load",
            "thd": "10% @ 3W",
            "efficiency": "90% (typ)",
            "snr": "80 dB (typ)",
            "operating_temp": "-40 to 85 C",
            "package": "SOP-16",
        },
        packages=["SOP-16"],
        alternatives=["MAX98357A (I2S input)", "TPA3116D2 (50W)", "NS4168 (mono, I2S)"],
        lcsc="C26480",
    ),
    # ===== ADC/DAC =====
    Component(
        name="MCP3008-I/SL",
        category="sensor",
        manufacturer="Microchip",
        description="10-bit ADC, 8 channels, SPI, 200 ksps",
        specs={
            "resolution": "10-bit",
            "channels": "8 (single-ended) or 4 (differential)",
            "sample_rate": "200 ksps",
            "supply_voltage": "2.7V to 5.5V",
            "supply_current": "0.5 mA (typ, 5V)",
            "reference": "VDD (internal) or external",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-16",
        },
        packages=["SOIC-16", "DIP-16"],
        interfaces=["SPI"],
        alternatives=["ADS1115 (16-bit, I2C, slower)", "MCP3208 (12-bit)", "ADS7828 (12-bit, I2C, 8-ch)"],
        lcsc="C55770",
    ),
    Component(
        name="MCP4725A0T-E/CH",
        category="sensor",
        manufacturer="Microchip",
        description="12-bit DAC, I2C, single channel, with EEPROM",
        specs={
            "resolution": "12-bit",
            "channels": "1",
            "i2c_address": "0x60 (A0=GND) or 0x61 (A0=VCC)",
            "output": "Voltage output (0 to VDD), 25 mA source",
            "settling_time": "6 us (typ)",
            "supply_voltage": "2.7V to 5.5V",
            "supply_current": "210 uA (typ)",
            "operating_temp": "-40 to 125 C",
            "package": "SOT-23-6",
        },
        packages=["SOT-23-6"],
        interfaces=["I2C"],
        alternatives=["MCP4728 (4-ch, 12-bit)", "DAC8571 (16-bit, TI)", "AD5693R (16-bit, ADI)"],
        lcsc="C61423",
    ),
    # ===== Additional Memory =====
    Component(
        name="AT24C256C-SSHL-T",
        category="memory",
        manufacturer="Microchip",
        description="256 Kbit (32 KB) I2C EEPROM, 1 MHz, 1M write cycles",
        specs={
            "capacity": "256 Kbit (32 KB)",
            "interface": "I2C (up to 1 MHz)",
            "i2c_address": "0x50-0x57 (A0/A1/A2 pins)",
            "page_size": "64 bytes",
            "write_cycle_time": "5 ms",
            "endurance": "1,000,000 write cycles",
            "data_retention": "100 years at 25 C",
            "supply_voltage": "1.7V to 5.5V",
            "supply_current": "2 mA (active), 1 uA (standby)",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8", "SOT-23-5 (AT24C256C-MAHL-T)"],
        interfaces=["I2C"],
        alternatives=["W25Q128 (SPI Flash, 16 MB)", "CAT24C256 (ON Semi)", "FM24CL64B (FRAM, faster write)"],
        lcsc="C6482",
    ),
    Component(
        name="IS62WV6416DBLL-55TLI",
        category="memory",
        manufacturer="ISSI",
        description="1 Mbit (64K x 16) SRAM, 55 ns, parallel interface",
        specs={
            "capacity": "1 Mbit (64K x 16-bit words = 128 KB)",
            "interface": "Parallel (16-bit data, 16-bit address)",
            "access_time": "55 ns",
            "supply_voltage": "2.4V to 3.6V",
            "supply_current": "18 mA (active), 4 uA (standby)",
            "operating_temp": "-40 to 85 C",
            "package": "TSOP-44",
        },
        packages=["TSOP-44"],
        alternatives=["IS62WV12816 (2 Mbit)", "CY7C1041G (4 Mbit)", "PSRAM (QSPI, higher density)"],
        lcsc="C32671",
    ),
    # ===== Additional Connectors =====
    Component(
        name="Micro SD Card Socket (push-push)",
        category="connector",
        manufacturer="Various (TE, Molex, Hirose)",
        description="Micro SD card socket, push-push, SMD, with card detect",
        specs={
            "type": "Micro SD (TF card) push-push ejection",
            "contacts": "8 (plus 1-2 card detect and/or write protect)",
            "interface": "SPI or SDIO (4-bit)",
            "voltage": "3.3V logic levels",
            "current_rating": "0.5 A per contact",
            "insertion_force": "3.5 N (max)",
            "operating_temp": "-25 to 85 C",
            "package": "SMD",
        },
        packages=["SMD"],
        interfaces=["SPI", "SDIO"],
        lcsc="C91145",
    ),
    Component(
        name="SMA Connector (edge-mount)",
        category="connector",
        manufacturer="Various",
        description="SMA female edge-mount PCB connector, 50 ohm, 6 GHz",
        specs={
            "type": "SMA female (jack), edge-mount",
            "impedance": "50 ohm",
            "frequency_range": "DC to 6 GHz",
            "vswr": "1.3:1 (max, to 6 GHz)",
            "current_rating": "1 A (center contact)",
            "voltage_rating": "500 V (peak)",
            "package": "Through-hole + SMD tabs",
        },
        packages=["Edge-mount PCB"],
        lcsc="C88374",
    ),
    # ===== Additional Logic ICs =====
    Component(
        name="CD4051BE",
        category="analog_switch",
        manufacturer="TI / NXP / ON Semi",
        description="8-channel analog multiplexer/demultiplexer, single-supply",
        specs={
            "channels": "8:1 (single-pole, 8-throw)",
            "on_resistance": "120 Ohm (typ at VDD=5V)",
            "supply_voltage": "3V to 20V (VDD-VSS max 20V)",
            "signal_range": "VSS to VDD (analog signal range)",
            "supply_current": "0.2 uA (quiescent)",
            "bandwidth": "~10 MHz (-3 dB)",
            "operating_temp": "-55 to 125 C",
            "package": "DIP-16 / SOIC-16",
        },
        packages=["DIP-16", "SOIC-16", "TSSOP-16"],
        alternatives=["CD4052 (dual 4:1)", "CD4053 (triple 2:1)", "ADG708 (analog mux, lower Ron)"],
        lcsc="C176372",
    ),
    Component(
        name="SN74LVC1T45DCKR",
        category="level_shifter",
        manufacturer="Texas Instruments",
        description="Single-bit bidirectional voltage level translator, SOT-363",
        specs={
            "channels": "1 (bidirectional)",
            "port_a_voltage": "1.65V to 3.6V",
            "port_b_voltage": "2.3V to 5.5V",
            "data_rate": "Up to 420 Mbps",
            "propagation_delay": "1.8 ns (typ)",
            "supply_current": "5 uA (quiescent)",
            "operating_temp": "-40 to 125 C",
            "package": "SOT-363 (SC-70-6)",
        },
        packages=["SOT-363"],
        alternatives=["TXB0101 (auto-direction)", "SN74LVC2T45 (dual)", "BSS138 (discrete, slower)"],
        lcsc="C7843",
    ),
    Component(
        name="PCA9685PW",
        category="logic",
        manufacturer="NXP",
        description="16-channel 12-bit PWM driver, I2C, for LEDs and servos",
        specs={
            "channels": "16 (independently controllable)",
            "resolution": "12-bit (4096 steps per channel)",
            "pwm_frequency": "24 Hz to 1526 Hz (programmable)",
            "i2c_address": "0x40-0x7F (62 addresses via A0-A5)",
            "supply_voltage": "2.3V to 5.5V",
            "output_current": "25 mA per pin (sink), 10 mA (source)",
            "supply_current": "6 mA (typ, all channels on)",
            "operating_temp": "-40 to 85 C",
            "package": "TSSOP-28",
        },
        packages=["TSSOP-28"],
        interfaces=["I2C"],
        alternatives=["TLC5947 (24-ch, SPI, constant current)", "IS31FL3731 (matrix LED driver)"],
        lcsc="C78616",
    ),
    # ===== Additional Power Components =====
    Component(
        name="SS34",
        category="protection",
        manufacturer="Various",
        description="3A 40V Schottky barrier rectifier diode, SMA package",
        specs={
            "type": "Schottky rectifier",
            "max_reverse_voltage": "40V",
            "forward_current": "3 A",
            "forward_voltage_drop": "0.5V @ 3A",
            "reverse_leakage": "0.5 mA @ 40V",
            "operating_temp": "-65 to 125 C",
            "package": "SMA (DO-214AC)",
        },
        packages=["SMA (DO-214AC)"],
        alternatives=["SS14 (1A 40V)", "SS54 (5A 40V)", "MBR340 (3A 40V, through-hole)"],
        lcsc="C8678",
    ),
    Component(
        name="BAT54S",
        category="protection",
        manufacturer="NXP / ON Semi / various",
        description="Dual series Schottky diode, 30V, 200mA, SOT-23",
        specs={
            "type": "Dual Schottky diode (series connection, common cathode)",
            "max_reverse_voltage": "30V",
            "forward_current": "200 mA",
            "forward_voltage_drop": "0.24V @ 0.1 mA, 0.8V @ 200 mA",
            "reverse_leakage": "2 uA @ 25V",
            "operating_temp": "-65 to 125 C",
            "package": "SOT-23",
        },
        packages=["SOT-23"],
        alternatives=["BAT54 (single)", "BAT54C (common cathode)", "1N5819 (1A, through-hole)"],
        lcsc="C84653",
    ),
    # ===== More Sensors =====
    Component(
        name="MAX30102",
        category="sensor",
        manufacturer="Analog Devices (Maxim)",
        description="Pulse oximetry and heart rate sensor, I2C, integrated LEDs and photodetector",
        specs={
            "measurement": "SpO2 (pulse oximetry) and heart rate",
            "led_wavelength": "Red (660 nm) + IR (880 nm)",
            "resolution": "18-bit ADC",
            "sample_rate": "50 to 3200 samples/s",
            "i2c_address": "0x57 (fixed)",
            "supply_voltage": "1.8V (I/O) + 3.3V (LED)",
            "supply_current": "600 uA (measurement), 0.7 uA (shutdown)",
            "operating_temp": "-40 to 85 C",
            "package": "OLGA-14 (5.6x3.3mm)",
        },
        packages=["OLGA-14 (5.6x3.3mm)"],
        interfaces=["I2C"],
        alternatives=["MAX30101 (green LED + IR + red)", "MAX86150 (ECG + PPG)"],
        lcsc="C130789",
    ),
    Component(
        name="APDS-9960",
        category="sensor",
        manufacturer="Broadcom (Avago)",
        description="Gesture, proximity, ambient light, and color (RGBC) sensor, I2C",
        specs={
            "measurement": "Gesture detection, proximity, ambient light, RGBC color",
            "gesture_range": "Up to 30 cm",
            "proximity_range": "Up to 20 cm",
            "light_range": "0 to 37,889 lux",
            "i2c_address": "0x39 (fixed)",
            "supply_voltage": "2.4V to 3.6V",
            "supply_current": "170 uA (gesture), 790 uA (proximity), 2 uA (wait/sleep)",
            "operating_temp": "-40 to 85 C",
            "package": "Dual flat lead, 8-pin (3.94x2.36mm)",
        },
        packages=["DFN-8 (3.94x2.36mm)"],
        interfaces=["I2C"],
        alternatives=["VEML7700 (light only)", "VL53L0X (ToF ranging)", "PAJ7620U2 (gesture)"],
        lcsc="C88327",
    ),
    Component(
        name="TSL2591",
        category="sensor",
        manufacturer="AMS-OSRAM",
        description="High dynamic range light sensor, I2C, 600M:1 lux range",
        specs={
            "measurement_range": "188 ulux to 88,000 lux",
            "dynamic_range": "600,000,000:1",
            "channels": "Full spectrum (visible + IR) + IR only",
            "i2c_address": "0x29 (fixed)",
            "supply_voltage": "3.3V (typ)",
            "supply_current": "0.4 mA (active), 0.3 uA (sleep)",
            "operating_temp": "-30 to 80 C",
            "package": "DFN-6 (2x2.4mm)",
        },
        packages=["DFN-6 (2x2.4mm)"],
        interfaces=["I2C"],
        alternatives=["BH1750 (cheaper, lower range)", "VEML7700 (Vishay)", "OPT3001 (TI, human eye response)"],
        lcsc="C183299",
    ),
    Component(
        name="HX711",
        category="sensor",
        manufacturer="Avia Semiconductor",
        description="24-bit ADC for load cells (Wheatstone bridge), 80 Hz",
        specs={
            "resolution": "24-bit",
            "channels": "2 (CH_A: gain 128 or 64, CH_B: gain 32)",
            "sample_rate": "10 Hz or 80 Hz (selectable)",
            "input_voltage": "Differential, +/-20 mV (gain 128) or +/-40 mV (gain 64)",
            "supply_voltage": "2.6V to 5.5V (analog), 2.6V to 5.5V (digital)",
            "supply_current": "1.5 mA (normal), 1 uA (power-down)",
            "operating_temp": "-40 to 85 C",
            "package": "SOP-16",
        },
        packages=["SOP-16"],
        interfaces=["Custom serial (DOUT + SCK, NOT SPI)"],
        alternatives=["ADS1232 (TI, 24-bit, SPI)", "NAU7802 (Nuvoton, I2C)", "CS5532 (Cirrus)"],
        lcsc="C44616",
    ),
    Component(
        name="AS5600-ASOM",
        category="sensor",
        manufacturer="AMS-OSRAM",
        description="12-bit magnetic rotary position sensor (encoder), I2C, contactless",
        specs={
            "resolution": "12-bit (4096 positions per revolution)",
            "accuracy": "+/-1 degree (after calibration)",
            "output": "I2C (12-bit angle) + analog (0-100% VDD) + PWM",
            "i2c_address": "0x36 (fixed)",
            "magnet": "Diametrically magnetized (6mm recommended)",
            "supply_voltage": "3.0V to 3.6V",
            "supply_current": "6.5 mA (typ)",
            "operating_temp": "-40 to 125 C",
            "package": "SOIC-8",
        },
        packages=["SOIC-8"],
        interfaces=["I2C", "Analog", "PWM"],
        alternatives=["TLE5012B (Infineon, SPI, 15-bit)", "MT6701 (MagnTek, cheaper)", "MA730 (14-bit, SPI)"],
        lcsc="C183783",
    ),
    # ===== Timers & Misc =====
    Component(
        name="DS3231SN#",
        category="timer",
        manufacturer="Analog Devices (Maxim)",
        description="I2C real-time clock (RTC), TCXO, +/-2 ppm accuracy, SOIC-16",
        specs={
            "accuracy": "+/-2 ppm (0 to 40 C), +/-3.5 ppm (-40 to 85 C)",
            "timekeeping": "Seconds, minutes, hours, day, date, month, year (with leap year)",
            "alarms": "2 programmable alarms",
            "oscillator": "Internal 32.768 kHz TCXO (no external crystal)",
            "i2c_address": "0x68 (fixed)",
            "supply_voltage": "2.3V to 5.5V",
            "battery_backup": "Yes (VBAT pin for CR2032)",
            "supply_current": "200 uA (active), 3 uA (battery timekeeping)",
            "operating_temp": "-40 to 85 C",
            "package": "SOIC-16W",
        },
        packages=["SOIC-16W"],
        interfaces=["I2C"],
        alternatives=["DS1307 (cheaper, lower accuracy)", "PCF8563 (NXP, I2C)", "RV-3028 (MicroCrystal, ultra-low power)"],
        lcsc="C9868",
    ),
    Component(
        name="MCP23017-E/SS",
        category="logic",
        manufacturer="Microchip",
        description="16-bit I/O expander, I2C, with interrupt output",
        specs={
            "channels": "16 GPIO (2x 8-bit ports)",
            "i2c_address": "0x20-0x27 (A0/A1/A2 pins)",
            "output_current": "25 mA per pin (sink or source)",
            "supply_voltage": "1.8V to 5.5V",
            "supply_current": "1 mA (active), 1 uA (standby)",
            "features": "Internal pull-ups, interrupt on change, polarity inversion",
            "i2c_speed": "Up to 1.7 MHz",
            "operating_temp": "-40 to 85 C",
            "package": "SSOP-28",
        },
        packages=["SSOP-28", "DIP-28", "QFN-28"],
        interfaces=["I2C"],
        alternatives=["PCF8574 (8-bit, simpler)", "MCP23S17 (SPI version)", "PCA9555 (NXP, 16-bit)"],
        lcsc="C47023",
    ),
    Component(
        name="TCA9548APWR",
        category="logic",
        manufacturer="Texas Instruments",
        description="8-channel I2C multiplexer, 400 kHz, low voltage",
        specs={
            "channels": "8 (downstream I2C buses)",
            "i2c_address": "0x70-0x77 (A0/A1/A2 pins)",
            "supply_voltage": "1.65V to 5.5V",
            "supply_current": "10 uA (typ)",
            "i2c_speed": "Up to 400 kHz",
            "features": "Active-low reset, interrupt output",
            "operating_temp": "-40 to 85 C",
            "package": "TSSOP-24",
        },
        packages=["TSSOP-24", "QFN-24"],
        interfaces=["I2C"],
        alternatives=["TCA9548A (same die)", "PCA9548 (NXP)", "PCA9546A (4-ch)"],
        lcsc="C130026",
    ),
    # ===== Additional LEDs =====
    Component(
        name="SK6812MINI-E",
        category="led",
        manufacturer="Opsco/Worldsemi",
        description="Addressable RGBW LED, 3535 reverse-mount, WS2812B-compatible protocol",
        specs={
            "type": "RGBW addressable LED (4-channel)",
            "data_rate": "800 kHz (NRZ protocol, WS2812-compatible)",
            "supply_voltage": "3.7V to 5.3V",
            "supply_current": "~60 mA (all white full brightness)",
            "color_depth": "8-bit per channel (256 levels x 4 = 4 billion colors)",
            "viewing_angle": "120 degrees",
            "package": "3535 reverse-mount (solder pads on bottom)",
        },
        packages=["3535 reverse-mount"],
        alternatives=["WS2812B (RGB only)", "APA102 (SPI, higher refresh)", "SK6812 (standard mount)"],
        lcsc="C5149201",
    ),
    Component(
        name="APA102-2020",
        category="led",
        manufacturer="Shenzhen LED Color",
        description="Addressable RGB LED, 2020 package, SPI interface, 20 MHz refresh",
        specs={
            "type": "RGB addressable LED (SPI — separate clock + data)",
            "data_rate": "Up to 20 MHz SPI clock",
            "supply_voltage": "3.3V to 5.5V",
            "supply_current": "~40 mA (all white full brightness)",
            "color_depth": "8-bit per channel + 5-bit global brightness",
            "refresh_rate": "Up to 20 kHz (flicker-free for cameras)",
            "package": "2020 (2x2mm)",
        },
        packages=["2020"],
        interfaces=["SPI (clock + data)"],
        alternatives=["WS2812B (single-wire, cheaper)", "SK9822 (APA102 clone)", "SK6812 (single-wire)"],
        lcsc="C2917362",
    ),
    # ===== Power Switches =====
    Component(
        name="AP22804AW5-7",
        category="power",
        manufacturer="Diodes Inc.",
        description="Load switch, 2.1A, active-low enable, over-current protection, SOT-25",
        specs={
            "max_output_current": "2.1 A (continuous)",
            "input_voltage": "2.5V to 5.5V",
            "rds_on": "90 mOhm (typ)",
            "enable": "Active-low",
            "current_limit": "2.1 A (typ)",
            "quiescent_current": "0.4 uA (disabled), 55 uA (enabled)",
            "operating_temp": "-40 to 85 C",
            "package": "SOT-25 (SOT-23-5)",
        },
        packages=["SOT-25"],
        alternatives=["TPS2041B (TI, USB switch)", "STMPS2151 (STM, 500mA)", "SY6280 (Silergy)"],
        lcsc="C176295",
    ),
    Component(
        name="TPS22918DBVR",
        category="power",
        manufacturer="Texas Instruments",
        description="Load switch, 2A, ultra-low Rds(on), controlled rise time, SOT-23-6",
        specs={
            "max_output_current": "2 A",
            "input_voltage": "1.0V to 5.5V",
            "rds_on": "52 mOhm (at 3.6V VIN)",
            "rise_time": "Configurable (500 us typ with CT cap)",
            "quiescent_current": "0.6 uA (disabled), 18 uA (enabled)",
            "features": "Quick output discharge, configurable rise time",
            "operating_temp": "-40 to 125 C",
            "package": "SOT-23-6",
        },
        packages=["SOT-23-6"],
        alternatives=["AP22804 (current limit)", "SLG59M1515V (Dialog)", "RT9742 (Richtek, 2A)"],
        lcsc="C96320",
    ),
]


# ---------------------------------------------------------------------------
# Question templates per category
# ---------------------------------------------------------------------------

SPEC_QUESTIONS: list[tuple[str, str]] = [
    # (question_template, answer_key_or_special)
    ("What is the operating voltage range of the {name}?", "supply_voltage"),
    ("What is the maximum clock frequency of the {name}?", "max_frequency"),
    ("How much Flash memory does the {name} have?", "flash"),
    ("How much SRAM does the {name} have?", "sram"),
    ("How many GPIO pins does the {name} have?", "gpio_count"),
    ("What ADC capabilities does the {name} have?", "adc_channels"),
    ("What communication interfaces does the {name} support?", "_interfaces"),
    ("What package options are available for the {name}?", "_packages"),
    ("What is the operating temperature range of the {name}?", "operating_temp"),
    ("What is the core architecture of the {name}?", "core"),
    ("What is the output voltage of the {name}?", "output_voltage"),
    ("What is the maximum output current of the {name}?", "max_output_current"),
    ("What is the dropout voltage of the {name}?", "dropout_voltage"),
    ("What is the quiescent current of the {name}?", "quiescent_current"),
    ("What is the input voltage range of the {name}?", "input_voltage"),
    ("What is the Rds(on) of the {name}?", "rds_on"),
    ("What is the Vgs threshold of the {name}?", "vgs_threshold"),
    ("What is the maximum Vds of the {name}?", "vds_max"),
    ("What is the maximum drain current of the {name}?", "id_continuous"),
    ("What is the gate charge of the {name}?", "gate_charge"),
    ("What is the gain-bandwidth product of the {name}?", "gbw"),
    ("What is the slew rate of the {name}?", "slew_rate"),
    ("What is the input offset voltage of the {name}?", "input_offset_voltage"),
    ("What is the input bias current of the {name}?", "input_bias_current"),
    ("What is the temperature accuracy of the {name}?", "temperature_accuracy"),
    ("What is the humidity accuracy of the {name}?", "humidity_accuracy"),
    ("What is the pressure range of the {name}?", "pressure_range"),
    ("What is the I2C address of the {name}?", "i2c_address"),
    ("What resolution does the {name} have?", "resolution"),
    ("How many channels does the {name} have?", "channels"),
    ("What is the switching frequency of the {name}?", "switching_frequency"),
    ("What is the efficiency of the {name}?", "efficiency"),
    ("What is the data rate of the {name}?", "data_rate"),
    ("What is the sensitivity of the {name}?", "sensitivity"),
    ("What power does the {name} output?", "output_power"),
    ("What is the sample rate of the {name}?", "sample_rate"),
    ("What is the supply current of the {name}?", "supply_current"),
    ("Describe the {name} component.", "_description"),
    ("What are the main features of the {name}?", "_features"),
    ("What alternatives exist for the {name}?", "_alternatives"),
]

# Detailed selection questions by category
SELECTION_QUESTIONS: list[dict[str, str]] = [
    # Regulators
    {
        "q": "I need a 3.3V LDO regulator that can supply 600mA for an ESP32 project powered by a single Li-Ion cell (3.0-4.2V). What are my options?",
        "a": """For powering an ESP32 from a Li-Ion cell (3.0-4.2V) with a 3.3V 600mA LDO, consider these options:

1. **AP2112K-3.3** — 600mA, 250mV dropout at full load, 55 uA quiescent current. SOT-23-5 package. At 3.0V input (discharged battery), the dropout may be tight (Vout = 3.0 - 0.25 = 2.75V), so you may see regulation dropout near end of battery life. LCSC: C51118.

2. **ME6211C33** — 600mA, 100mV dropout, 40 uA Iq. Pin-compatible with AP2112K. Lower dropout helps maintain 3.3V down to ~3.4V cell voltage. Good availability on LCSC.

3. **XC6220B331MR** — 700mA, 120mV dropout at 100mA, ultra-low 0.8 uA Iq. Ideal if sleep current matters. SOT-23-5. LCSC: C86534.

4. **RT9013-33GB** — 500mA (slightly under your 600mA need), but very low noise (15 uVrms) — good if ADC accuracy matters. SOT-23-5.

**Recommendation**: For best battery life range, use the **ME6211** or **XC6220** due to lower dropout. For ESP32 specifically, check that the LDO can handle the Wi-Fi TX current spikes (peaks up to 500mA). Consider adding a 100uF bulk capacitor on the output.

If you need regulation all the way down to 3.0V cell voltage, consider a **buck-boost** like TPS63020 instead — it maintains 3.3V output whether input is above or below 3.3V.""",
    },
    {
        "q": "I need a step-down converter to go from 12V to 5V at 2A for a Raspberry Pi project. What should I use?",
        "a": """For a 12V to 5V step-down at 2A (10W), here are good options:

1. **TPS54331** — 3A max, 570 kHz switching, SOIC-8 with exposed pad. Very well documented with TI's WEBENCH design tool. Needs external inductor (10-22 uH), bootstrap cap, and output caps. Efficiency ~90% at 2A. LCSC: C15769. About $0.80.

2. **MP1584EN** — 3A, 1.5 MHz, tiny SOT-23-8 package. Higher switching frequency means smaller inductor (4.7 uH). Very popular on JLCPCB. Efficiency ~92%. LCSC: C14259. About $0.50.

3. **LM2596S-5.0** — 3A, 150 kHz, TO-263-5. Simple fixed-5V output, minimal external components. Larger inductor needed (33-68 uH) due to low frequency. Efficiency ~85%. Larger board space. LCSC: C29781.

4. **SY8205** — 5A, adjustable, SOT-23-6. Cheap and capable. Popular in Chinese dev boards. LCSC: C141906.

**Recommendation**: For a clean design, use **MP1584EN** (small, efficient, cheap, proven). For easiest layout and lowest component count, use **LM2596S-5.0** (everything is in the datasheet application circuit).

For the Raspberry Pi specifically, ensure output ripple is <100 mV. Use at least 2x 22 uF ceramic output caps plus a 220 uF electrolytic. The Pi is sensitive to voltage dips during SD card writes.""",
    },
    {
        "q": "What's a good MOSFET for switching a 12V 5A load with a 3.3V gate signal?",
        "a": """For switching 12V/5A (60W) with a 3.3V gate signal, you need a **logic-level N-channel MOSFET** with low Rds(on) at Vgs=3.3V. Key requirements:
- Vds >= 20V (12V * 1.5x safety margin minimum, 30V preferred)
- Id >= 5A continuous
- Low Rds(on) at Vgs = 3.3V (not just at Vgs = 10V!)
- Vgs(th) well below 3.3V

**Through-hole (TO-220)**:
1. **IRLZ44N** — 55V, 47A, Rds(on) = 22 mOhm at Vgs=4V. Vgs(th) = 1-2V. Classic logic-level MOSFET. At 3.3V gate, Rds will be higher (~35 mOhm), P_loss = 0.035 * 5^2 = 0.875W — manageable without heatsink. LCSC: C2536.

2. **IRL540N** — 100V, 36A, Rds(on) = 44 mOhm at Vgs=5V. Higher voltage rating but higher Rds at 3.3V.

**SMD (SOT-23)**:
3. **AO3400** — 30V, 5.7A, Rds(on) = 40 mOhm at Vgs=2.5V. Tiny SOT-23 package. P_loss = 0.04 * 25 = 1W — at the thermal limit of SOT-23 (1.4W max). Tight but workable with good PCB copper. LCSC: C20917.

4. **IRLML6344** — 30V, 5A, Rds(on) = 29 mOhm at Vgs=2.5V. Better Rds than AO3400, same package. LCSC: C181093.

**SMD (SO-8 / PowerPAK)**:
5. **Si4946BDY** — Dual N-FET, 60V, 5.3A each, SO-8. Good for two loads.
6. **IRLR2905** — 55V, 36A, DPAK. Very low Rds(on) = 27 mOhm at 4.5V. Overkill but reliable.

**Recommendation**: For SMD, use **IRLML6344** in SOT-23. For through-hole prototyping, **IRLZ44N** in TO-220. Both are specified at Vgs=2.5V which guarantees they work well at 3.3V.

Important: do NOT use IRF540N (non-logic-level, needs 10V gate drive).""",
    },
    {
        "q": "I need an I2C temperature sensor accurate to +/-0.5 C or better for a weather station. Recommendations?",
        "a": """For a weather station temperature sensor with +/-0.5C accuracy or better via I2C:

1. **TMP117** — +/-0.1C accuracy (-20 to +50C), 16-bit resolution (0.0078C), I2C, SOT-563. NIST-traceable. The gold standard for I2C temperature sensing. ~$2.50. LCSC: C2677286.

2. **SHT31-DIS** — +/-0.2C accuracy, I2C, also measures humidity (+/-2% RH). DFN-8 (2.5x2.5mm). Great for weather stations since you get both temp and humidity. ~$2.00. LCSC: C78592.

3. **BME280** — +/-0.5C accuracy (best case), plus humidity (+/-3% RH) and barometric pressure (+/-1 hPa). LGA-8 (2.5x2.5mm). Three sensors in one, ideal for weather stations. ~$3.00. LCSC: C92489.

4. **DS18B20** — +/-0.5C accuracy (-10 to +85C), but uses 1-Wire (not I2C). Available in waterproof probe form. Classic choice, ~$1.00.

5. **SHT40** — Successor to SHT31, +/-0.2C, smaller DFN-4 (1.5x1.5mm), lower power. ~$1.50.

**Recommendation**: For a weather station, use **BME280** if you want temp + humidity + pressure in one sensor. Use **TMP117** if temperature accuracy is the top priority. The BME280 at +/-0.5C may not meet your spec reliably — its temperature sensor is secondary to the pressure sensor.

For outdoor use, consider the sensor placement carefully: solar radiation shield, ventilation, and no direct sunlight on the sensor.""",
    },
    {
        "q": "I need to add Wi-Fi to an existing STM32 project. What modules should I consider?",
        "a": """For adding Wi-Fi to an STM32 project, several approaches:

**SPI-based Wi-Fi modules (STM32 stays as main MCU)**:
1. **ESP32-C3-MINI-1** — RISC-V, Wi-Fi + BLE 5, run AT firmware via UART or SPI. ~$1.50. Most popular co-processor approach. Flash AT firmware from Espressif, then send AT commands over UART.

2. **ESP-AT on any ESP module** — ESP32-WROOM-32, ESP32-S3, etc. with Espressif's AT firmware. UART interface (2 pins + optional flow control).

3. **WizFi360** — AT command set compatible with ESP-AT. Drop-in Wi-Fi module. ~$3.

**Network co-processor approach**:
4. **ATWINC1500** (Microchip) — SPI interface, handles full TCP/IP stack internally. STM32 sends high-level socket commands. Simpler integration but limited to basic sockets. ~$4.

5. **Inventek ISM43362** (used on STM32 Discovery boards) — SPI, AT-like commands. STM32 BSP support available.

**Full TCP/IP on STM32**:
6. **ENC28J60** or **W5500** — Ethernet (not Wi-Fi), but if wired is acceptable, much simpler. W5500 has hardware TCP/IP stack.

**Swap to Wi-Fi MCU**:
7. If early in design, consider switching to **STM32WB** (BLE only), **ESP32-S3** (Wi-Fi+BLE), or **STM32H5 + MXCHIP EMW3080** (STM32 with Wi-Fi expansion).

**Recommendation**: Easiest path is **ESP32-C3** running AT firmware over UART. You need only TX, RX, EN, and RST pins. Libraries exist for STM32 HAL. The ESP handles all Wi-Fi/TLS complexity. Add a level shifter if your STM32 runs at 5V logic.

For production, the ATWINC1500 has better long-term supply guarantees and Microchip support.""",
    },
    {
        "q": "What op-amp should I use for a battery-powered sensor node that needs to amplify a 0-50mV signal to 0-3.3V?",
        "a": """For amplifying 0-50mV to 0-3.3V on a battery-powered sensor (gain = 66x), key requirements:
- Single supply (battery, likely 3.3V)
- Rail-to-rail input AND output (signal starts at 0V)
- Low offset voltage (50mV full-scale means even 1mV offset = 2% error)
- Low power consumption
- Low noise at the gain you're using

**Best options**:

1. **OPA333AIDBVR** — Zero-drift (chopper), 2 uV max offset, 1.8-5.5V supply, 17 uA Iq, SOT-23-5. Rail-to-rail I/O. Offset won't drift with temperature. Best accuracy. ~$1.50. LCSC: C137148.

2. **MCP6001** — 4.5mV max offset, 1.8-6V supply, 100 uA Iq, SOT-23-5. Rail-to-rail I/O. Cheap (~$0.30) but higher offset. May need calibration.

3. **AD8605** — 65 uV max offset, 2.7-5.5V supply, 1.5 mA Iq, SOT-23-5. Rail-to-rail I/O, low noise (7 nV/rtHz). Higher power but excellent precision.

4. **LPV521** (TI) — 0.4 uA Iq (!), 7 mV offset, 1.6-5.5V. Ultra-low power but higher offset.

**Circuit considerations**:
- At 66x gain, use a non-inverting configuration: Vout = Vin * (1 + R2/R1). Set R1 = 1K, R2 = 65K (nearest standard: 64.9K 1%).
- Add a low-pass filter at the input to reduce noise: 1K + 100nF = 1.6 kHz cutoff.
- Use 0.1% or 1% resistors — at this gain, resistor tolerance directly affects gain accuracy.
- Rail-to-rail output typically can't reach 0V exactly — expect 10-50mV above ground under load.

**Recommendation**: **OPA333** for best accuracy without calibration. The zero-drift architecture eliminates offset drift over temperature, which is critical in sensor applications. Its 17 uA current is fine for battery-powered use.""",
    },
    {
        "q": "I need to drive a 12V LED strip (2A total) from a microcontroller. What components do I need?",
        "a": """To drive a 12V 2A LED strip from a microcontroller, you need an N-channel MOSFET for switching:

**Basic circuit**: MCU GPIO -> Gate resistor -> MOSFET gate. MOSFET source to GND, drain to LED strip negative. LED strip positive to 12V supply.

**MOSFET selection** (logic-level, Vds >= 20V, Id >= 2A, low Rds at 3.3V):

1. **AO3400** — SOT-23, 30V/5.7A, Rds(on) = 40 mOhm at Vgs=2.5V. Power loss = 0.04 * 4 = 0.16W. Well within SOT-23 limits. LCSC: C20917. ~$0.03.

2. **IRLML6344** — SOT-23, 30V/5A, Rds(on) = 29 mOhm at Vgs=2.5V. Even lower losses. LCSC: C181093. ~$0.08.

3. **IRLZ44N** — TO-220, 55V/47A, Rds(on) = 22 mOhm at Vgs=4V. Overkill but bulletproof for prototyping. LCSC: C2536.

**Supporting components**:
- **Gate resistor**: 100-470 ohm (limits inrush current to gate capacitance, protects GPIO)
- **Pull-down resistor**: 10K-100K from gate to GND (keeps MOSFET off during MCU boot/reset)
- **Flyback diode**: Not needed for LED strips (no inductive load), but add one if driving LED strips with long cables
- **Bypass cap**: 100nF near MOSFET drain-source to suppress switching transients

**For PWM dimming**:
- PWM frequency: 1-25 kHz (visible flicker below 200 Hz, audible coil whine above 20 kHz in some setups)
- The AO3400's gate charge (6.8nC) allows fast switching up to ~1 MHz
- Rise/fall time with 220 ohm gate resistor: ~1.5 us (fine for 1 kHz PWM)

**For RGB strips** (3 channels):
- Use 3x AO3400, one per color channel (R, G, B)
- Each channel controlled by separate PWM pin on MCU

**Recommendation**: **AO3400** for cost, **IRLML6344** for lowest heat. Both in SOT-23. Add 220R gate resistor + 100K pull-down.""",
    },
    {
        "q": "I need a current sensor for a battery management system measuring 0-20A. What should I use?",
        "a": """For 0-20A current sensing in a BMS, several approaches:

**Shunt resistor + amplifier approach**:
1. **INA226** + shunt resistor — Digital I2C output, 16-bit, measures both current and voltage simultaneously. Use a 5 mOhm shunt for 20A (100mV full-scale drop, 2W dissipation). Bidirectional measurement. LCSC: C138706. ~$2.00.

2. **INA219** + shunt resistor — Simpler I2C version, 12-bit, 26V max bus voltage. Cheaper. LCSC: C12904. ~$1.00.

3. **INA181A3** + shunt resistor — Analog output (voltage proportional to current), 200x gain, SOT-23-5. Feed into MCU ADC. Good if you already have a free ADC channel. LCSC: C380439.

**Hall-effect sensors** (galvanic isolation, no insertion loss):
4. **ACS712-20A** — Hall sensor, 20A range, analog output (100 mV/A), 5V supply. No shunt needed. DIP-8 or SOIC-8. ~$2.50. LCSC: C10681.

5. **ACS758LCB-020B** — 20A bidirectional, analog output (40 mV/A at 3.3V). CB package. Better accuracy than ACS712.

**Integrated solution**:
6. **INA228** — 85V bus voltage, 20-bit ADC, calculates power and energy in hardware. Premium option. ~$4.

**Shunt resistor selection** (if using INA226/219):
- 5 mOhm: 100mV at 20A, 2W dissipation. Use Bourns CSS2H-2512 (2512 package, 3W rated).
- 2 mOhm: 40mV at 20A, 0.8W. Less power loss but lower signal (still fine for INA226).
- 1 mOhm: 20mV at 20A, 0.4W. Minimum loss but requires high-side amplifier with low offset.

**Recommendation**: For a BMS, use **INA226** with a **2 mOhm 2512 shunt**. It gives you both current AND bus voltage measurement digitally via I2C, supports up to 36V bus, and has alert thresholds for overcurrent protection. The 16-bit resolution gives you ~0.6mA resolution with a 2 mOhm shunt.

For a Kelvin-connected shunt layout, route sense lines separately from power lines to avoid voltage drop errors in PCB traces.""",
    },
    {
        "q": "I need a stepper motor driver for a 3D printer. What are the best options for silent operation?",
        "a": """For silent 3D printer stepper motor operation, the Trinamic TMC series is the clear winner:

1. **TMC2209** — The sweet spot. 2A RMS (2.8A peak), UART config, StealthChop2 (silent), StallGuard4 (sensorless homing), CoolStep (adaptive current). Up to 256 microsteps. Single-wire UART for configuration. ~$1.50-3.00. LCSC: C100103.

2. **TMC2226** — Budget version of TMC2209 with fewer features. Still has StealthChop for silent operation. ~$1.00-2.00.

3. **TMC5160** — Premium. 4.4A RMS, SPI config, external MOSFETs possible for higher current. For large printers with NEMA23 motors. ~$5-8.

4. **TMC2240** — Newest generation, built-in MOSFETs, 3A RMS, SPI, improved StealthChop. ~$3-5.

**Comparison with non-Trinamic drivers**:
- **A4988** — 2A peak, 1/16 microstep max, NO silent mode. Loud. ~$0.50. Fine for CNC but not for quiet 3D printing.
- **DRV8825** — 2.5A, 1/32 microstep, NO silent mode. Slightly quieter than A4988 but still noisy.
- **LV8729** — 1.5A, 1/128 microstep, quieter than A4988/DRV8825 but no StealthChop equivalent.

**TMC2209 configuration for 3D printer**:
- Default: StealthChop2 (silent, good for low speeds)
- Switch to SpreadCycle above ~100 mm/s for better torque
- Set motor current via UART (Vref not needed)
- Enable StallGuard for sensorless homing (eliminates endstop switches)
- UART requires only 1 pin per driver (can share with address selection)

**Recommendation**: **TMC2209** for most 3D printers. Use UART mode (not standalone) to unlock all features. Marlin and Klipper firmware both have excellent TMC2209 support. If using Klipper, you can tune StealthChop thresholds and StallGuard sensitivity per-axis.""",
    },
    {
        "q": "I need a USB-C connector for a simple USB 2.0 device. What do I need on the schematic besides the connector?",
        "a": """For a USB 2.0 device with USB-C connector, here's the complete schematic:

**1. USB-C Connector** (16-pin simplified is fine for USB 2.0):
- LCSC: C2765186 (Korean Hroparts TYPE-C-31-M-12)
- Only need: VBUS, GND, D+, D-, CC1, CC2, Shield

**2. CC Configuration Resistors** (MANDATORY):
- 5.1K resistor from CC1 to GND
- 5.1K resistor from CC2 to GND
- These identify your device as a UFP (Upstream Facing Port / device)
- Without these, a USB-C host will NOT provide power

**3. ESD Protection** (strongly recommended):
- **USBLC6-2SC6** on D+/D- and VBUS. SOT-23-6. Protects against ESD up to 15kV. LCSC: C7519. ~$0.15.
- Place as close to the connector as possible

**4. VBUS Power**:
- 100nF + 10uF bypass capacitors on VBUS
- Optional: Schottky diode (e.g., SS14) if your circuit has another power source (prevents backfeed)
- Optional: Polyfuse (500mA-2A) for overcurrent protection

**5. Data Lines**:
- D+ and D- go to your MCU USB pins (or USB-UART bridge like CH340C)
- Route as differential pair, 90 ohm impedance
- Keep traces short (<50mm if possible)
- No series resistors needed for USB 2.0 (some MCUs have internal termination)
- 0402 ferrite beads are optional for EMI

**6. Shield Connection**:
- Connector shield to GND via 1M resistor in parallel with 4.7nF capacitor
- This provides ESD path while avoiding ground loops

**Complete BOM for USB-C port**:
| Part | Value | Package | LCSC |
|------|-------|---------|------|
| USB-C connector | 16-pin | Mid-mount | C2765186 |
| R (CC1) | 5.1K 1% | 0402 | C25905 |
| R (CC2) | 5.1K 1% | 0402 | C25905 |
| R (shield) | 1M | 0402 | C26083 |
| C (shield) | 4.7nF | 0402 | C53112 |
| C (VBUS) | 100nF | 0402 | C1525 |
| C (VBUS) | 10uF | 0805 | C15850 |
| ESD | USBLC6-2SC6 | SOT-23-6 | C7519 |

Total: ~8 components, ~$0.60 BOM cost.""",
    },
]

CROSSREF_QUESTIONS: list[dict[str, str]] = [
    {
        "q": "What can I use instead of the LM7805 that's more efficient?",
        "a": """The LM7805 is a linear regulator with ~40-60% efficiency (depending on input voltage). More efficient alternatives:

**Drop-in switching replacements** (pin-compatible TO-220):
1. **R-78E5.0-1.0** (RECOM) — Switching regulator in TO-220 pinout. 90%+ efficiency. 1A output. No external components. Direct plug-in replacement. ~$3-4. Available at Mouser/DigiKey.

2. **TSR 1-2450** (Traco) — Similar drop-in switching module. 500mA. 90% efficiency. ~$4.

**Board-level switching replacements** (different footprint, much cheaper):
3. **LM2596S-5.0** — Buck converter, 3A, TO-263-5. 85-88% efficiency. Needs inductor + diode + caps. ~$0.50. LCSC: C29781.

4. **MP1584EN** — Buck converter, 3A, SOT-23-8. 92% efficiency. Tiny footprint. ~$0.50. LCSC: C14259.

5. **TPS54331** — Buck converter, 3A, SOIC-8. 90% efficiency. Well-documented design process. ~$0.80. LCSC: C15769.

**Low-dropout linear** (if you want to stay linear):
6. **AMS1117-5.0** — LDO, 1A, SOT-223. 1.3V dropout (vs 2V for 7805). Still linear (similar efficiency) but works at lower input voltage (6.3V vs 7V). ~$0.10. LCSC: C6187.

**Efficiency comparison at 12V input, 5V/500mA output**:
- LM7805: P_waste = (12-5) * 0.5 = 3.5W, efficiency = 42%
- LM2596: P_waste ~0.5W, efficiency = 85%
- MP1584: P_waste ~0.3W, efficiency = 90%
- R-78E5.0: P_waste ~0.3W, efficiency = 90%

**Recommendation**: For new designs, use **MP1584EN** (cheapest, smallest, most efficient). For retrofitting existing 7805 boards without layout changes, use **R-78E5.0-1.0** (direct pin-compatible drop-in).""",
    },
    {
        "q": "Is the AMS1117-3.3 pin-compatible with the AP2112K-3.3?",
        "a": """No, the AMS1117-3.3 and AP2112K-3.3 are NOT pin-compatible. They have different packages and pinouts:

**AMS1117-3.3** (SOT-223):
- Pin 1: GND/Adjust
- Pin 2: Vout
- Pin 3: Vin
- Tab: Vout
- Package: SOT-223 (6.5 x 3.5 mm)

**AP2112K-3.3** (SOT-23-5):
- Pin 1: Vin
- Pin 2: GND
- Pin 3: EN (Enable)
- Pin 4: NC (or Bypass)
- Pin 5: Vout
- Package: SOT-23-5 (2.9 x 1.6 mm)

**Key differences**:
- Different package (SOT-223 vs SOT-23-5)
- Different pin count (3+tab vs 5)
- AP2112K has an enable pin (useful for power sequencing and sleep modes)
- AP2112K is much smaller (1/4 the board area)
- AP2112K has lower dropout (250mV vs 1.3V)
- AP2112K has lower Iq (55 uA vs 5 mA)
- AMS1117 has higher current (1A vs 600mA)

**Pin-compatible alternatives to AMS1117-3.3** (SOT-223):
- LM1117-3.3 (original, by TI)
- SPX3819M5-L-3.3 (lower dropout)

**Pin-compatible alternatives to AP2112K-3.3** (SOT-23-5):
- ME6211C33 (same pinout, lower dropout)
- XC6220B331MR (same pinout, lower Iq)
- RT9013-33GB (same pinout, lower noise)""",
    },
    {
        "q": "What's a drop-in replacement for the GD32F103C8T6 that's actually an STM32?",
        "a": """The GD32F103C8T6 is a clone of the STM32F103C8T6, and swapping back to genuine STM32 is mostly straightforward but with caveats:

**Direct replacement**: **STM32F103C8T6**
- Same LQFP-48 package and pinout
- Same peripheral set (3 UART, 2 SPI, 2 I2C, USB, CAN)
- Same 72 MHz Cortex-M3 core

**Key differences to watch**:
1. **Clock speed**: GD32F103 actually runs at 108 MHz, STM32F103 at 72 MHz. If your firmware relies on GD32's higher speed, it will be slower on genuine STM32.

2. **Flash latency**: GD32 uses different flash wait states. If your code sets flash latency explicitly for GD32, update for STM32.

3. **USB**: GD32's USB peripheral has subtle timing differences. USB code that works on GD32 may need minor adjustments on STM32.

4. **ADC**: GD32 ADC has slightly different calibration and may give different readings than STM32.

5. **Flash size**: Both are "64KB" parts, but GD32F103C8 actually has 128KB (undocumented extra), while STM32F103C8 strictly has 64KB. If your firmware uses >64KB, you'll need STM32F103CBT6 (128KB).

**Other genuine STM32 alternatives** (pin-compatible LQFP-48):
- **STM32F103CBT6** — Same as C8 but with 128KB flash guaranteed
- **STM32F303C8T6** — Cortex-M4F (FPU!), 72 MHz, better ADC
- **STM32G431C6U6** — Cortex-M4F, 170 MHz, much faster, USB, CAN FD

**Chinese alternatives** (same pinout, cheaper):
- **APM32F103C8T6** (Geehy/Apex) — Very compatible clone
- **CH32F103C8T6** (WCH) — Cheapest, some peripheral differences
- **AT32F403ACGT7** (Artery) — 240 MHz, pin-compatible, much faster""",
    },
    {
        "q": "What's the ESP32-C3 equivalent of the ESP32-S3's camera interface?",
        "a": """The ESP32-C3 does NOT have a camera (DVP) interface. This is a key difference between the ESP32 variants:

**Camera interface by ESP32 variant**:
- **ESP32 (original)**: 8-bit DVP camera interface via I2S peripheral (hack, limited)
- **ESP32-S2**: 8/16-bit DVP camera interface (dedicated peripheral)
- **ESP32-S3**: 8/16-bit DVP camera interface (dedicated LCD_CAM peripheral, best support)
- **ESP32-C3**: NO camera interface
- **ESP32-C6**: NO camera interface
- **ESP32-H2**: NO camera interface

**If you need a camera and want to use an ESP32-C3**, your options are:
1. **SPI camera modules** — OV2640 or OV5640 have SPI modes, but frame rates are very low (1-5 fps at QVGA)
2. **External USB camera** — ESP32-C3 doesn't have USB Host capability
3. **UART camera modules** — Grove serial camera, very low resolution

**Recommendation**: If you need a camera, use **ESP32-S3**. It's the only current-gen ESP32 with good camera support. The ESP32-S3 + OV2640 is the standard combination, with ESP-IDF providing mature drivers.

For comparison:
| Feature | ESP32-S3 | ESP32-C3 |
|---------|----------|----------|
| Core | Dual Xtensa LX7 | Single RISC-V |
| Camera | DVP 8/16-bit | None |
| USB | OTG 1.1 | None (GPIO USB on some) |
| AI accel | Vector instructions | None |
| Price | ~$2.50 | ~$1.00 |
| Wi-Fi | 802.11 b/g/n | 802.11 b/g/n |
| BLE | 5.0 | 5.0 |""",
    },
    {
        "q": "I'm using an NE555 timer but need something that works at 3.3V. What can I replace it with?",
        "a": """The classic NE555 requires at least 4.5V supply, so it won't work reliably at 3.3V. CMOS equivalents work from 1.5V to 16V:

**Drop-in CMOS replacements** (same pinout as NE555):
1. **LMC555** (TI) — 1.5V to 15V, CMOS, DIP-8/SOIC-8. Same pinout. 100 uA typical supply current (vs 3-15 mA for NE555). LCSC: C129837.

2. **TLC555** (TI) — 2V to 15V, CMOS, DIP-8/SOIC-8. Same pinout. 170 uA typical. LCSC: C7545.

3. **ICM7555** (Renesas) — 2V to 18V, CMOS, DIP-8/SOIC-8. Same pinout. Most directly compatible. LCSC: C114718.

4. **CSS555** (Custom Silicon Solutions) — 1.2V to 5.5V, ultra-low power (5 uA!). DIP-8/SOIC-8.

**Key differences from NE555**:
- CMOS versions have much lower output current (~20mA vs 200mA for NE555)
- CMOS versions have much lower supply current (100 uA vs 3 mA)
- Timing may differ slightly due to different threshold voltages under load
- CMOS outputs are rail-to-rail (NE555 output drops ~1.5V from VCC)
- CMOS versions are more sensitive to noise on power rail — add good bypassing

**For SOT-23-5 (tiny package)**:
5. **LMC555 in SOT-23-5** — TI offers this in a tiny 5-pin package
6. **TLV555 in SOT-23-5** — Even lower power variant

**Recommendation**: **LMC555** is the most universally compatible. Same pinout, works from 1.5V to 15V, and is widely available. Just update timing resistor values if your output load changed (CMOS output impedance is higher).""",
    },
]

BOM_SOURCING_QUESTIONS: list[dict[str, str]] = [
    {
        "q": "What's the LCSC part number for a 100nF 0402 capacitor suitable for decoupling?",
        "a": """For a 100nF 0402 decoupling capacitor, common LCSC options:

**Best choices**:
1. **C1525** — Samsung CL05B104KO5NNNC, 100nF, 0402, 16V, X5R, +/-10%. JLCPCB **basic part** (no extended fee). ~$0.002/pc in reel. This is the go-to choice.

2. **C307331** — Samsung CL05B104KB5NNNC, 100nF, 0402, 50V, X5R, +/-10%. Higher voltage rating, also basic part. Good if your design has >16V rails nearby.

3. **C14663** — Murata GRM155R71C104KA88D, 100nF, 0402, 16V, X7R, +/-10%. X7R has better temperature stability than X5R.

**JLCPCB basic vs extended**:
- Basic parts have no additional fee per unique part
- Extended parts add $3 per unique part number
- C1525 is basic — use it to save on assembly fees

**Typical usage per board**:
- 1x per IC power pin (VDD, VCC, VDDA, etc.)
- Place within 2mm of the pin
- Via to ground plane directly under or next to the cap

**Ordering tip**: Order 100nF 0402 caps in bulk (5000+ pcs reel). At LCSC they're ~$1 per 500 pcs. A typical MCU board uses 5-15 of these.""",
    },
    {
        "q": "Where can I source the STM32F103C8T6 in small quantities (1-10 pcs)?",
        "a": """For small quantities (1-10 pcs) of STM32F103C8T6:

**Authorized distributors** (genuine, guaranteed):
1. **Mouser** (mouser.com) — Usually in stock. ~$3-5 per unit for qty 1. Free shipping over $50. Genuine guaranteed.
2. **DigiKey** (digikey.com) — Same price range. Ships from USA, fast delivery. Genuine guaranteed.
3. **Farnell/Newark** (farnell.com / newark.com) — European/US distributor. Similar pricing.
4. **LCSC** (lcsc.com) — LCSC part C8734. ~$2-3 per unit. Ships from China (5-15 days). Genuine STM parts.

**Development boards** (cheaper than bare chips for prototyping):
5. **Blue Pill board** — STM32F103C8T6 on a breakout board with crystal, regulator, USB, headers. ~$2-3 on AliExpress. WARNING: many Blue Pills use counterfeit or GD32 chips labeled as STM32.
6. **WeAct Studio STM32F103** — Higher quality dev board, verified genuine STM32. Available on AliExpress.

**Marketplace** (risk of counterfeits):
7. **AliExpress** — $1-2 per chip, but HIGH RISK of counterfeit/remarked chips. Many are GD32 or recycled parts with STM32 markings.
8. **eBay** — Same counterfeit risks.

**How to verify authenticity**:
- Check lot code format (ST uses specific patterns)
- Use STM32CubeProgrammer to read device ID and flash size
- Genuine STM32F103C8 has 64KB flash; counterfeits may show 128KB (GD32 sign)
- ST offers STSAFE authentication on newer parts

**Recommendation**: For guaranteed genuine parts, buy from **DigiKey or Mouser** even if slightly more expensive. For cost-sensitive prototyping where a GD32 clone is acceptable, buy from **LCSC**. Never trust bare chips from AliExpress for production.""",
    },
    {
        "q": "Which package is cheapest for a 10K resistor at JLCPCB assembly?",
        "a": """At JLCPCB SMT assembly, the cheapest 10K resistor option is **0402 (basic part)**:

**JLCPCB basic 10K resistors by package**:

| Package | LCSC Part | Manufacturer | Price/pc (qty 100) | Basic? |
|---------|-----------|-------------|-------------------|--------|
| **0402** | **C25744** | Yageo RC0402FR-0710KL 1% | ~$0.001 | **Yes** |
| 0603 | C25804 | Yageo RC0603FR-0710KL 1% | ~$0.001 | Yes |
| 0805 | C17414 | Yageo RC0805FR-0710KL 1% | ~$0.001 | Yes |
| 1206 | C17902 | Yageo RC1206FR-0710KL 1% | ~$0.001 | Yes |

**The real cost difference is in assembly, not parts**:
- Basic parts: $0.00 extended component fee
- Extended parts: $3.00 extra per unique part number
- All standard resistor values in 0402/0603/0805/1206 are basic parts

**Assembly considerations by package**:
- **0402**: Smallest, saves board space, but harder to hand-rework. JLCPCB handles 0402 fine.
- **0603**: Good balance of size and hand-solderability.
- **0805**: Easiest to hand-solder if you need rework. Slightly larger pads.

**Cost optimization tips**:
1. Stick to basic parts — check JLCPCB's basic parts list
2. Minimize unique part numbers (each unique extended part adds $3)
3. Use the same package across your design (e.g., all 0402) for consistency
4. Standard values (10K, 4.7K, 100K, 1K) are always basic parts
5. 0.1% tolerance resistors are usually extended parts — use 1% unless precision is needed

**Recommendation**: Use **0402 for production** (C25744), **0603 for prototyping** (C25804) if you might need to hand-solder.""",
    },
    {
        "q": "I need to find JLCPCB basic parts for a complete USB-C to UART design. What LCSC part numbers should I use?",
        "a": """Here's a complete USB-C to UART design using only JLCPCB basic parts (no extended component fees):

**Core IC**:
| Part | LCSC | Description | Basic? | Price |
|------|------|-------------|--------|-------|
| CH340C | C84681 | USB-UART bridge (no crystal needed!) | **Yes** | ~$0.40 |

**USB-C Connector**:
| Part | LCSC | Description | Basic? | Price |
|------|------|-------------|--------|-------|
| TYPE-C 16P | C2765186 | USB-C receptacle, 16-pin | Yes | ~$0.08 |

**ESD Protection**:
| Part | LCSC | Description | Basic? | Price |
|------|------|-------------|--------|-------|
| USBLC6-2SC6 | C7519 | USB ESD protection, SOT-23-6 | Yes | ~$0.15 |

**Passives** (all 0402, all basic):
| Part | LCSC | Value | Purpose |
|------|------|-------|---------|
| C1525 | C1525 | 100nF 0402 | CH340C bypass caps (x2) |
| C15850 | C15850 | 10uF 0805 | VBUS bulk cap |
| C25905 | C25905 | 5.1K 0402 | CC1 pull-down |
| C25905 | C25905 | 5.1K 0402 | CC2 pull-down |
| C26083 | C26083 | 1M 0402 | Shield to GND |
| C53112 | C53112 | 4.7nF 0402 | Shield cap |

**LED indicators** (optional):
| Part | LCSC | Description | Basic? |
|------|------|-------------|--------|
| C72043 | C72043 | Green LED 0603 | Yes |
| C2286 | C2286 | Red LED 0603 | Yes |
| C25744 | C25744 | 10K 0402 (LED resistors) | Yes |

**Headers**:
| Part | LCSC | Description | Basic? |
|------|------|-------------|--------|
| C124378 | C124378 | 1x6 pin header 2.54mm | Yes |

**Total BOM cost**: ~$0.90 per board (excluding PCB)
**Extended part fees**: $0 (all basic parts!)
**JLCPCB assembly**: ~$2.00 setup + $0.90 parts for qty 5

**CH340C pinout note**: No external crystal needed (unlike CH340G). Saves one part and two load capacitors. UART pins: TXD, RXD, plus optional CTS/DTR/DSR/RTS for flow control and auto-reset on Arduino-style boards.""",
    },
    {
        "q": "What's the best way to source components for a prototype run of 10 boards?",
        "a": """For a 10-board prototype run, here's the optimal sourcing strategy:

**Option 1: JLCPCB PCB + SMT Assembly (recommended)**
- Order PCBs + SMT assembly in one go
- JLCPCB sources parts from LCSC automatically
- You provide BOM (LCSC part numbers) and CPL (placement) files
- Cost: ~$2 setup + parts + $0.50-1.50/board assembly
- Turnaround: 5-7 days manufacturing + 3-7 days shipping (DHL)
- **Pros**: Cheapest for small runs, good quality, no hand-soldering
- **Cons**: Only LCSC parts, some parts may be out of stock

**Option 2: PCBWay Assembly**
- Similar to JLCPCB but can source from multiple suppliers
- Slightly more expensive but better for complex BOMs
- Can handle parts you ship to them (consigned)

**Sourcing strategy by part type**:

| Part Type | Source | Why |
|-----------|--------|-----|
| Common passives (R, C, L) | LCSC (via JLCPCB) | Cheapest, basic parts |
| Common ICs (STM32, ESP32, etc.) | LCSC (via JLCPCB) | Usually in stock |
| Specialty/expensive ICs | DigiKey or Mouser | Guaranteed genuine, fast |
| Connectors | LCSC or DigiKey | Check JLCPCB basic list first |
| Through-hole parts | DigiKey, hand-solder | JLCPCB THT assembly is expensive |

**Cost optimization tips**:
1. Design with JLCPCB basic parts when possible ($0 extended fee)
2. Check LCSC stock BEFORE finalizing your BOM
3. Order a few extra of expensive/hard-to-find parts
4. For through-hole parts (pin headers, large connectors), hand-solder rather than paying THT assembly
5. Use JLCPCB's parts library search to verify availability

**BOM preparation**:
1. Export BOM from KiCad (Manufacturer + MPN + LCSC fields)
2. Map each part to an LCSC number
3. Verify all parts are in stock at LCSC
4. Generate CPL (pick-and-place) file from KiCad
5. Upload both to JLCPCB assembly order

**Budget estimate for 10 boards** (typical STM32 design):
- PCB (10 pcs, 2-layer, 100x100mm): $5-8
- SMT assembly (10 pcs, 20-30 parts): $20-40
- Components (LCSC): $10-30 depending on BOM
- Shipping (DHL Express): $15-20
- **Total: ~$50-100 for 10 assembled boards**""",
    },
]

DATASHEET_QUESTIONS: list[dict[str, str]] = [
    {
        "q": "How do I read the absolute maximum ratings table in a datasheet?",
        "a": """The Absolute Maximum Ratings (AMR) table lists conditions that, if exceeded, may permanently damage the device. Here's how to read it:

**What the AMR table tells you**:
- These are LIMITS, not operating conditions
- Exceeding ANY single parameter can destroy the device
- The device is NOT guaranteed to function at these values
- These are stress ratings only — design your circuit to stay well below them

**Common AMR parameters**:

| Parameter | What it means | Example (STM32F103) |
|-----------|--------------|---------------------|
| VDD to VSS | Max supply voltage | -0.3V to 4.0V |
| VIN (I/O pins) | Max voltage on any pin | VSS-0.3V to VDD+0.3V |
| VDDA | Max analog supply | -0.3V to 4.0V |
| IDD (total) | Max total supply current | 150 mA |
| I (per pin) | Max current per I/O pin | +/-25 mA |
| I (total, all I/O) | Max sum of all I/O currents | +/-80 mA |
| Tstg | Storage temperature | -65 to 150 C |
| Tj | Junction temperature | 150 C |

**How to use AMR in design**:
1. **Derate by 20-50%** — If AMR says 4.0V max, design for 3.6V max (the recommended operating range)
2. **Check ALL conditions** — Your circuit must never exceed ANY single AMR parameter, even during transients, power-on, or fault conditions
3. **ESD ratings are separate** — AMR doesn't cover ESD; check the ESD section
4. **Latch-up caution** — Exceeding VDD+0.3V on an input can trigger CMOS latch-up, which shorts VDD to GND and can destroy the chip even after the overvoltage is removed

**Common mistakes**:
- Applying 5V signals to a 3.3V MCU (exceeds VIN max of VDD+0.3V = 3.6V)
- Driving too many LEDs from I/O pins (exceeds total I/O current)
- Hot-plugging connectors (voltage spikes exceed AMR)
- Ignoring the "operating" ratings table (which has tighter limits for guaranteed functionality)

**Key distinction**: The "Recommended Operating Conditions" table (usually right after AMR) shows the range where the device is guaranteed to meet its electrical specifications. Always design to the operating range, not the absolute max.""",
    },
    {
        "q": "What does the thermal resistance theta-JA (θJA) mean in a MOSFET datasheet?",
        "a": """θJA (theta-JA) is the thermal resistance from the semiconductor junction (J) to the ambient air (A), measured in degrees Celsius per Watt (C/W). It tells you how much the junction temperature rises for each watt of power dissipated.

**Formula**: Tj = Ta + (θJA × Pd)
- Tj = junction temperature (C)
- Ta = ambient temperature (C)
- θJA = thermal resistance, junction to ambient (C/W)
- Pd = power dissipated in the device (W)

**Example** (AO3400 MOSFET, SOT-23):
- θJA = 100 C/W (typical for SOT-23 on minimal PCB copper)
- Ambient temperature: 25 C
- Power dissipation: Rds(on) × I² = 0.040 × 3² = 0.36 W
- Junction temperature: 25 + (100 × 0.36) = 61 C
- Max Tj is 150 C, so plenty of margin.

**But at 5A**: 25 + (100 × 0.040 × 25) = 125 C — getting close to the limit!

**Common θJA values by package**:
| Package | Typical θJA | Notes |
|---------|-------------|-------|
| SOT-23 | 200-350 C/W | With minimal copper |
| SOT-23 | 80-150 C/W | With good copper pour |
| SOIC-8 | 100-150 C/W | Depends on pad area |
| DPAK (TO-252) | 50-80 C/W | With 1 sq inch copper |
| TO-220 | 60 C/W | No heatsink |
| TO-220 | 5-20 C/W | With heatsink |
| QFN (exposed pad) | 30-60 C/W | Soldered to ground plane |

**Related thermal resistances**:
- **θJC** (junction to case): Used when a heatsink is attached. Much lower than θJA.
- **θJB** (junction to board): More relevant for SMD packages where heat flows to PCB.
- **Ψjt** (junction to top): For IR temperature measurement. Not a true thermal resistance.

**Design tips**:
1. θJA in the datasheet assumes a specific PCB (usually JEDEC 2-layer, 1 sq inch copper). Your actual θJA depends on YOUR PCB copper area.
2. More copper = lower θJA = cooler operation. Add ground plane copper under power components.
3. Use thermal vias under exposed pads (array of 0.3mm vias to inner/bottom ground plane).
4. If calculated Tj > 80% of Tj(max), consider a larger package or heatsink.
5. For MOSFETs: Rds(on) increases with temperature (positive temp coefficient), creating a thermal feedback loop. At high temperatures, the MOSFET gets hotter, which increases Rds(on), which generates more heat.""",
    },
    {
        "q": "How do I interpret the Rds(on) specification in a MOSFET datasheet? Why are there multiple values?",
        "a": """A MOSFET datasheet lists multiple Rds(on) values because the on-resistance depends heavily on gate voltage (Vgs) and junction temperature (Tj).

**Typical Rds(on) table** (example: AO3400):
| Parameter | Conditions | Min | Typ | Max | Unit |
|-----------|-----------|-----|-----|-----|------|
| Rds(on) | Vgs = 10V, Id = 5.7A | - | 22 | 30 | mOhm |
| Rds(on) | Vgs = 4.5V, Id = 5.7A | - | 26 | 40 | mOhm |
| Rds(on) | Vgs = 2.5V, Id = 4A | - | 30 | 50 | mOhm |

**How to read this**:

1. **Use the value matching YOUR gate voltage**: If driving from a 3.3V MCU, look at the Vgs=2.5V line (closest below your Vgs). The Rds(on) at Vgs=2.5V is much higher than at Vgs=10V.

2. **Use MAX, not TYP for worst-case**: The MAX column accounts for manufacturing variation. Design thermal management for MAX Rds(on).

3. **Temperature derating**: Datasheet Rds(on) is at 25 C junction temperature. At 125 C, Rds(on) typically increases by 1.5x to 2x. Check the "Rds(on) vs Temperature" graph.

**Why Rds(on) varies with Vgs**:
- More gate voltage = stronger channel inversion = lower resistance
- Below Vgs(th), the MOSFET is off (infinite resistance)
- At Vgs(th), it starts to turn on (very high resistance, linear region)
- At Vgs >> Vgs(th), the channel is fully enhanced (minimum resistance)
- The datasheet Rds(on) assumes the MOSFET is fully on

**Practical calculations**:
- Power loss: P = Rds(on) × Id²
- For 3A through AO3400 at Vgs=3.3V: P = 0.050 × 9 = 0.45W (use MAX value)
- For 3A through AO3400 at Vgs=10V: P = 0.030 × 9 = 0.27W

**Common mistakes**:
1. Using the headline Rds(on) (at Vgs=10V) when your gate voltage is 3.3V
2. Forgetting temperature derating (multiply Rds by 1.5x for hot environments)
3. Assuming the "typ" value instead of "max" for thermal design
4. Not checking that Vgs(th) MAX is well below your gate voltage (e.g., if Vgs(th) max is 2V and you drive at 2.5V, the MOSFET may only be partially on)

**Rule of thumb**: For reliable operation, ensure your Vgs is at least 2x the Vgs(th) MAX value.""",
    },
    {
        "q": "What do the different capacitor dielectric codes (C0G, X5R, X7R, Y5V) mean and when should I use each?",
        "a": """Capacitor dielectric codes describe the temperature stability and voltage characteristics of ceramic capacitors (MLCCs). The code is defined by EIA standards:

**Class I dielectrics** (stable, precise):
| Code | Temp Range | Tolerance | Typical Use |
|------|-----------|-----------|-------------|
| **C0G (NP0)** | -55 to +125 C | +/-30 ppm/C | Timing circuits, filters, oscillators, precision analog |

- Capacitance does NOT change with temperature, voltage, or age
- Available up to ~100 nF in reasonable sizes
- No piezoelectric effect (no microphonics)
- Most expensive per uF

**Class II dielectrics** (higher capacitance, less stable):
| Code | Temp Range | Cap Change | Typical Use |
|------|-----------|------------|-------------|
| **X7R** | -55 to +125 C | +/-15% | Decoupling, filtering, timing (non-critical) |
| **X5R** | -55 to +85 C | +/-15% | Decoupling, bulk capacitance |
| **X6S** | -55 to +105 C | +/-22% | Automotive, high-temp environments |

- Capacitance varies with temperature, DC bias voltage, and AC signal level
- Available in high values (up to 100 uF in 0805)
- DC bias effect: a 10 uF X5R cap at rated voltage may only give 5 uF effective
- Slight piezoelectric effect (can cause audible noise in audio circuits)

**Class III dielectrics** (maximum capacitance, poor stability):
| Code | Temp Range | Cap Change | Typical Use |
|------|-----------|------------|-------------|
| **Y5V** | -30 to +85 C | +22/-82% | Non-critical bypass, cost-sensitive designs |
| **Z5U** | +10 to +85 C | +22/-56% | Same as Y5V, slightly better |

- Capacitance drops dramatically with temperature and voltage
- At -30 C, a Y5V cap may lose 82% of its rated value!
- Cheapest per uF but least reliable

**Decision guide**:
| Application | Recommended | Why |
|-------------|-------------|-----|
| Crystal load caps | C0G/NP0 | Stability critical for frequency accuracy |
| Op-amp feedback | C0G/NP0 | Gain stability, no microphonics |
| IC decoupling (VDD) | X5R or X7R | High capacitance needed, stability less critical |
| LDO output cap | X5R or X7R | ESR matters more than stability |
| Bulk bypass | X5R | Maximum capacitance per size |
| Audio path | C0G/NP0 | No piezoelectric noise (microphonics) |
| Timing circuit (555) | C0G/NP0 | Temperature = frequency stability |
| DC blocking (audio) | X7R okay | Stability less critical for AC coupling |
| Power supply input | X7R | Voltage ripple, not precision |

**Key rule**: When in doubt, X7R for general use, C0G for precision.""",
    },
    {
        "q": "How do I calculate the load capacitors for an MCU crystal oscillator?",
        "a": """To calculate crystal load capacitors (CL1 and CL2), use this formula:

**Formula**: CL = 2 × (Cload - Cstray)

Where:
- **Cload** = crystal's specified load capacitance (from crystal datasheet)
- **Cstray** = parasitic capacitance on each crystal pin (PCB traces + MCU pin capacitance)
- **CL** = value for each of the two load capacitors (CL1 = CL2)

**Step-by-step example** (STM32F103 + 8 MHz crystal):

1. **Find Cload from crystal datasheet**: Typically 18 pF or 20 pF for 8 MHz HC49S. Let's say 20 pF.

2. **Estimate Cstray**: Combination of:
   - MCU pin capacitance: ~5 pF (check MCU datasheet, "I/O pin capacitance")
   - PCB trace capacitance: ~1-3 pF (depends on trace length and layer stack)
   - Total Cstray: ~5-7 pF typical. Use 5 pF as a starting estimate.

3. **Calculate CL**:
   CL = 2 × (20 - 5) = 30 pF
   Nearest standard value: 27 pF or 33 pF

4. **Choose standard values**:
   - If Cstray is on the low side (~3-4 pF): use 33 pF
   - If Cstray is on the high side (~6-7 pF): use 27 pF
   - When in doubt: 33 pF is more common and slightly more conservative

**Common Cload values and resulting CL**:
| Crystal Cload | Cstray (est.) | Calculated CL | Nearest Std |
|---------------|---------------|---------------|-------------|
| 8 pF | 3 pF | 10 pF | 10 pF |
| 10 pF | 5 pF | 10 pF | 10 pF |
| 12.5 pF | 5 pF | 15 pF | 15 pF |
| 18 pF | 5 pF | 26 pF | 27 pF |
| 20 pF | 5 pF | 30 pF | 33 pF |

**For 32.768 kHz crystals** (RTC):
- Cload is typically 6 pF or 12.5 pF
- Cstray matters MORE because the crystal drive level is very low
- For 6 pF Cload with 3 pF stray: CL = 2 × (6-3) = 6 pF → use 6.8 pF
- For 12.5 pF Cload with 3 pF stray: CL = 2 × (12.5-3) = 19 pF → use 18 pF or 22 pF

**Important tips**:
1. Use **C0G/NP0** capacitors for load caps (temperature stability)
2. Keep traces short between crystal, caps, and MCU pins
3. Don't route other signals near crystal traces
4. Add a ground plane under the crystal for shielding
5. If the oscillator doesn't start or frequency is off, adjust CL up or down
6. Too much CL = frequency pulls low, harder to start
7. Too little CL = frequency pulls high, more prone to spurious oscillation
8. Some MCUs (e.g., nRF52840, RP2040) have internal tunable load caps — check the datasheet before adding external ones""",
    },
]


# ---------------------------------------------------------------------------
# JITX Stanza file parser
# ---------------------------------------------------------------------------

def parse_jitx_stanza(path: Path) -> Component | None:
    """Parse a JITX open-components-database Stanza file into a Component."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return None

    name = ""
    manufacturer = ""
    mpn = ""
    pins: list[str] = []
    specs: dict[str, str] = {}

    # Extract component name
    m = re.search(r'name\s*=\s*"([^"]+)"', text)
    if m:
        name = m.group(1)

    m = re.search(r'manufacturer\s*=\s*"([^"]+)"', text)
    if m:
        manufacturer = m.group(1)

    m = re.search(r'mpn\s*=\s*"([^"]+)"', text)
    if m:
        mpn = m.group(1)

    # Extract pin names from pin-properties block
    pin_block = re.search(r'pin-properties\s*:\s*(.*?)(?:make-box-symbol|assign-landpattern|name\s*=)', text, re.DOTALL)
    if pin_block:
        for pm in re.finditer(r'\[(\w[\w\[\]\d]*)\s*\|', pin_block.group(1)):
            pin_name = pm.group(1)
            if pin_name not in ("pin", "Ref", "pads", "side"):
                pins.append(pin_name)

    # Extract power-pin voltage
    for pm in re.finditer(r'PowerPin\(min-typ-max\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', text):
        specs["supply_voltage"] = f"{pm.group(1)}V to {pm.group(3)}V (typical {pm.group(2)}V)"

    # Extract rated temperature
    m = re.search(r'rated-temperature.*?min-max\(([-\d.]+),\s*([-\d.]+)\)', text)
    if m:
        specs["operating_temp"] = f"{m.group(1)} to {m.group(2)} C"

    if not mpn and not name:
        return None

    display_name = mpn or name
    description = f"{manufacturer} {display_name}" if manufacturer else display_name

    # Determine category from path or content
    parent = path.parent.name
    category = "component"
    if any(k in text.lower() for k in ["accelerometer", "gyro", "sensor", "temperature"]):
        category = "sensor"
    elif any(k in text.lower() for k in ["regulator", "ldo", "buck", "boost"]):
        category = "regulator"
    elif any(k in text.lower() for k in ["mosfet", "transistor", "fet"]):
        category = "mosfet"
    elif any(k in text.lower() for k in ["mcu", "microcontroller", "cortex"]):
        category = "mcu"

    return Component(
        name=display_name,
        category=category,
        manufacturer=manufacturer,
        description=description,
        specs=specs,
        pins=pins,
    )


def load_jitx_components(jitx_path: Path) -> list[Component]:
    """Load all components from JITX open-components-database."""
    components = []
    if not jitx_path.exists():
        logger.warning("JITX path %s not found, skipping", jitx_path)
        return components

    comp_dir = jitx_path / "components"
    if not comp_dir.exists():
        logger.warning("JITX components dir not found at %s", comp_dir)
        return components

    for stanza_file in sorted(comp_dir.rglob("*.stanza")):
        c = parse_jitx_stanza(stanza_file)
        if c and c.pins:
            components.append(c)

    logger.info("Loaded %d components from JITX database", len(components))
    return components


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_spec_qa(components: list[Component]) -> list[dict]:
    """Generate component specs Q&A pairs."""
    pairs = []

    for comp in components:
        for q_template, key in SPEC_QUESTIONS:
            if key == "_interfaces":
                if not comp.interfaces:
                    continue
                q = q_template.format(name=comp.name)
                ifaces = ", ".join(comp.interfaces)
                a = f"The {comp.name} supports the following communication interfaces: {ifaces}."
                if comp.notes:
                    a += f"\n\nNote: {comp.notes}"
                pairs.append(msg(q, a))

            elif key == "_packages":
                if not comp.packages:
                    continue
                q = q_template.format(name=comp.name)
                pkgs = ", ".join(comp.packages)
                a = f"The {comp.name} is available in the following packages: {pkgs}."
                pairs.append(msg(q, a))

            elif key == "_description":
                q = q_template.format(name=comp.name)
                a = f"The {comp.name} is a {comp.description}, manufactured by {comp.manufacturer}."
                if comp.specs:
                    key_specs = []
                    for sk, sv in list(comp.specs.items())[:6]:
                        key_specs.append(f"- {sk.replace('_', ' ').title()}: {sv}")
                    a += "\n\nKey specifications:\n" + "\n".join(key_specs)
                if comp.notes:
                    a += f"\n\nNote: {comp.notes}"
                pairs.append(msg(q, a))

            elif key == "_features":
                q = q_template.format(name=comp.name)
                a = f"The {comp.name} ({comp.description}) has the following key features:\n\n"
                for sk, sv in comp.specs.items():
                    a += f"- **{sk.replace('_', ' ').title()}**: {sv}\n"
                if comp.interfaces:
                    a += f"\nInterfaces: {', '.join(comp.interfaces)}"
                if comp.packages:
                    a += f"\nAvailable packages: {', '.join(comp.packages)}"
                if comp.lcsc:
                    a += f"\nLCSC part number: {comp.lcsc}"
                pairs.append(msg(q, a))

            elif key == "_alternatives":
                if not comp.alternatives:
                    continue
                q = q_template.format(name=comp.name)
                alts = "\n".join(f"- {alt}" for alt in comp.alternatives)
                a = f"Alternatives to the {comp.name}:\n\n{alts}"
                pairs.append(msg(q, a))

            else:
                # Direct spec lookup
                if key not in comp.specs:
                    continue
                q = q_template.format(name=comp.name)
                a = f"The {comp.name} has a {key.replace('_', ' ')} of {comp.specs[key]}."
                # Add context
                if key == "supply_voltage" and "dropout_voltage" in comp.specs:
                    a += f" The dropout voltage is {comp.specs['dropout_voltage']}."
                elif key == "flash" and "sram" in comp.specs:
                    a += f" It also has {comp.specs['sram']} of SRAM."
                elif key == "rds_on" and "vgs_threshold" in comp.specs:
                    a += f" The gate threshold voltage (Vgs_th) is {comp.specs['vgs_threshold']}."
                if comp.lcsc:
                    a += f" LCSC part number: {comp.lcsc}."
                pairs.append(msg(q, a))

    return pairs


def generate_jitx_qa(components: list[Component]) -> list[dict]:
    """Generate Q&A pairs from JITX parsed components."""
    pairs = []

    for comp in components:
        # Pin-related questions
        if comp.pins:
            q = f"What are the pins of the {comp.name}?"
            pin_list = ", ".join(comp.pins)
            a = f"The {comp.name}"
            if comp.manufacturer:
                a += f" by {comp.manufacturer}"
            a += f" has the following pins: {pin_list}."
            if comp.specs.get("supply_voltage"):
                a += f" Operating voltage: {comp.specs['supply_voltage']}."
            if comp.specs.get("operating_temp"):
                a += f" Operating temperature: {comp.specs['operating_temp']}."
            pairs.append(msg(q, a))

            # How many pins
            q2 = f"How many pins does the {comp.name} have?"
            a2 = f"The {comp.name} has {len(comp.pins)} pins: {pin_list}."
            pairs.append(msg(q2, a2))

            # Power pins
            power_pins = [p for p in comp.pins if any(k in p.upper() for k in ["VDD", "VCC", "GND", "VSS", "VIN", "VOUT"])]
            if power_pins:
                q3 = f"What are the power pins of the {comp.name}?"
                a3 = f"The power-related pins of the {comp.name} are: {', '.join(power_pins)}."
                if comp.specs.get("supply_voltage"):
                    a3 += f" The supply voltage range is {comp.specs['supply_voltage']}."
                pairs.append(msg(q3, a3))

    return pairs


def generate_selection_qa() -> list[dict]:
    """Generate component selection Q&A pairs."""
    return [msg(item["q"], item["a"]) for item in SELECTION_QUESTIONS]


def generate_crossref_qa() -> list[dict]:
    """Generate cross-reference Q&A pairs."""
    return [msg(item["q"], item["a"]) for item in CROSSREF_QUESTIONS]


def generate_bom_qa() -> list[dict]:
    """Generate BOM/sourcing Q&A pairs."""
    return [msg(item["q"], item["a"]) for item in BOM_SOURCING_QUESTIONS]


def generate_datasheet_qa() -> list[dict]:
    """Generate datasheet reading Q&A pairs."""
    return [msg(item["q"], item["a"]) for item in DATASHEET_QUESTIONS]


# Additional cross-component Q&A generators

def generate_comparison_qa(components: list[Component]) -> list[dict]:
    """Generate component comparison Q&A by comparing components in the same category."""
    pairs = []
    by_category: dict[str, list[Component]] = {}
    for c in components:
        by_category.setdefault(c.category, []).append(c)

    for cat, comps in by_category.items():
        if len(comps) < 2:
            continue
        # Pairwise comparisons (limit to avoid explosion)
        for i in range(min(len(comps), 6)):
            for j in range(i + 1, min(len(comps), 6)):
                c1, c2 = comps[i], comps[j]
                q = f"What are the differences between {c1.name} and {c2.name}?"
                a = f"Comparing {c1.name} and {c2.name}:\n\n"
                a += f"**{c1.name}**: {c1.description}\n"
                a += f"**{c2.name}**: {c2.description}\n\n"
                a += "Key differences:\n"

                # Find common spec keys
                common_keys = set(c1.specs.keys()) & set(c2.specs.keys())
                diff_count = 0
                for sk in sorted(common_keys):
                    if c1.specs[sk] != c2.specs[sk]:
                        label = sk.replace("_", " ").title()
                        a += f"- **{label}**: {c1.name} = {c1.specs[sk]} vs {c2.name} = {c2.specs[sk]}\n"
                        diff_count += 1
                        if diff_count >= 8:
                            break

                if diff_count == 0:
                    a += "- Specs are very similar. Check datasheets for detailed differences.\n"

                # Package comparison
                if c1.packages and c2.packages:
                    a += f"\nPackages: {c1.name} comes in {', '.join(c1.packages[:3])}; {c2.name} comes in {', '.join(c2.packages[:3])}."

                pairs.append(msg(q, a))

    return pairs


def generate_pinout_qa(components: list[Component]) -> list[dict]:
    """Generate pinout-specific Q&A for components with known pins."""
    pairs = []

    for comp in components:
        if not comp.pins or len(comp.pins) < 3:
            continue

        # Categorize pins
        power = [p for p in comp.pins if any(k in p.upper() for k in ["VDD", "VCC", "GND", "VSS", "AVDD", "DVDD"])]
        comms = [p for p in comp.pins if any(k in p.upper() for k in ["SPI", "SDA", "SCL", "TX", "RX", "MOSI", "MISO", "SCK", "CS", "SS"])]
        gpio = [p for p in comp.pins if any(k in p.upper() for k in ["GPIO", "PA", "PB", "PC", "IO"])]
        interrupts = [p for p in comp.pins if "INT" in p.upper()]
        analog = [p for p in comp.pins if any(k in p.upper() for k in ["ADC", "AIN", "AOUT", "DAC"])]
        nc_pins = [p for p in comp.pins if "NC" in p.upper() or "nc" in p]

        if comms:
            q = f"What communication pins does the {comp.name} have?"
            a = f"The {comp.name} has the following communication-related pins: {', '.join(comms)}."
            if comp.interfaces:
                a += f" It supports: {', '.join(comp.interfaces)}."
            pairs.append(msg(q, a))

        if interrupts:
            q = f"Does the {comp.name} have interrupt outputs?"
            a = f"Yes, the {comp.name} has {len(interrupts)} interrupt pin(s): {', '.join(interrupts)}."
            pairs.append(msg(q, a))

        if analog:
            q = f"What analog pins does the {comp.name} have?"
            a = f"The {comp.name} has the following analog pins: {', '.join(analog)}."
            pairs.append(msg(q, a))

        if nc_pins:
            q = f"Are there any no-connect pins on the {comp.name}?"
            a = f"Yes, the {comp.name} has {len(nc_pins)} no-connect pin(s): {', '.join(nc_pins)}. These pins should be left unconnected in your schematic."
            pairs.append(msg(q, a))

    return pairs


def generate_parametric_qa(components: list[Component]) -> list[dict]:
    """Generate parametric / rephrased variations of spec questions."""
    pairs = []

    # Alternate phrasings for common specs
    REPHRASINGS: dict[str, list[str]] = {
        "supply_voltage": [
            "What voltage should I supply to the {name}?",
            "Can the {name} operate at 3.3V?",
            "What is the minimum supply voltage for the {name}?",
            "What is the maximum supply voltage for the {name}?",
            "Is the {name} 5V tolerant?",
            "What power supply does the {name} need?",
        ],
        "max_frequency": [
            "How fast can the {name} run?",
            "What is the clock speed of the {name}?",
            "Can I overclock the {name}?",
        ],
        "flash": [
            "How much program memory does the {name} have?",
            "Is {flash} of flash enough for a typical project with the {name}?",
            "What is the flash memory size of the {name}?",
        ],
        "rds_on": [
            "What is the on-resistance of the {name}?",
            "How much power will the {name} dissipate at 3A?",
            "What is the conduction loss of the {name}?",
        ],
        "output_voltage": [
            "What voltage does the {name} output?",
            "Is the {name} a fixed or adjustable regulator?",
        ],
        "max_output_current": [
            "How much current can the {name} deliver?",
            "What is the current capacity of the {name}?",
            "Can the {name} supply 500mA?",
        ],
        "gbw": [
            "What is the bandwidth of the {name}?",
            "At what frequency does the {name} gain drop to unity?",
        ],
        "i2c_address": [
            "What I2C address should I use to communicate with the {name}?",
            "Can I have multiple {name} on the same I2C bus?",
            "What is the default I2C address of the {name}?",
        ],
        "temperature_accuracy": [
            "How accurate is the {name} temperature reading?",
            "What is the measurement error of the {name}?",
        ],
        "channels": [
            "Is the {name} single or dual channel?",
            "How many independent channels does the {name} provide?",
        ],
    }

    for comp in components:
        for spec_key, questions in REPHRASINGS.items():
            if spec_key not in comp.specs:
                continue
            val = comp.specs[spec_key]
            for q_tmpl in questions:
                try:
                    q = q_tmpl.format(name=comp.name, **{spec_key: val})
                except KeyError:
                    q = q_tmpl.format(name=comp.name)

                # Build contextual answer
                a = f"The {comp.name} ({comp.description}) "

                if "Can" in q or "Is" in q or "should" in q.lower():
                    # Yes/no style question
                    a += f"has a {spec_key.replace('_', ' ')} specification of {val}. "
                    if spec_key == "supply_voltage":
                        if "3.3" in q:
                            operates = "3.3" in val or "3.0" in val or "2." in val or "1." in val
                            a += "Yes, " if operates else "This needs to be checked — "
                            a += f"the operating voltage range is {val}."
                        elif "5V tolerant" in q:
                            a += f"The supply voltage range is {val}. "
                            if "5.5" in val or "5.0" in val or "6" in val:
                                a += "The device can operate at 5V."
                            else:
                                a += "The device cannot operate at 5V directly. You would need a level shifter or voltage regulator."
                        else:
                            a += f"The operating voltage range is {val}."
                    elif spec_key == "max_output_current":
                        a += f"The maximum output current is {val}."
                        if "500mA" in q:
                            try:
                                current_val = float(re.search(r'([\d.]+)\s*[Am]', val).group(1))
                                unit = "A" if "A" in val.split()[0] else "mA"
                                if unit == "A":
                                    can = current_val >= 0.5
                                else:
                                    can = current_val >= 500
                                a += f" {'Yes' if can else 'No'}, it {'can' if can else 'cannot'} supply 500mA."
                            except (AttributeError, ValueError):
                                a += f" Check the datasheet for details at your target current."
                    elif spec_key == "i2c_address":
                        a += f"The I2C address is {val}."
                        if "multiple" in q.lower() or "same bus" in q.lower():
                            if "," in val or "or" in val:
                                a += " Multiple devices can coexist on the same bus using different address configurations."
                            else:
                                a += " Only one device with this address can be on a single I2C bus unless an I2C multiplexer (like TCA9548A) is used."
                    else:
                        a += f"The {spec_key.replace('_', ' ')} is {val}."
                elif "How much power" in q or "conduction loss" in q:
                    a += f"has an Rds(on) of {val}. "
                    a += "Power dissipation = Rds(on) x I^2. "
                    try:
                        rds_mohm = float(re.search(r'([\d.]+)\s*mOhm', val).group(1))
                        p_3a = rds_mohm / 1000 * 9
                        a += f"At 3A: P = {rds_mohm/1000:.4f} x 9 = {p_3a:.3f}W."
                    except (AttributeError, ValueError):
                        a += f"Calculate based on the Rds(on) value of {val}."
                elif "overclock" in q.lower():
                    a += f"has a rated maximum frequency of {val}. "
                    a += "Overclocking is possible on some parts but voids reliability guarantees and may cause instability. "
                    a += "Always verify stable operation with your specific workload and temperature conditions."
                elif "enough" in q.lower() and spec_key == "flash":
                    a += f"has {val} of flash memory. "
                    a += "For a typical bare-metal project (GPIO, UART, I2C), this is sufficient. "
                    a += "With RTOS or USB stack, you may need 64KB+. "
                    a += "With networking or TLS, consider 128KB+ or external flash."
                else:
                    a += f"has a {spec_key.replace('_', ' ')} of {val}."

                if comp.lcsc:
                    a += f" LCSC: {comp.lcsc}."

                pairs.append(msg(q, a))

    return pairs


def generate_design_pattern_qa() -> list[dict]:
    """Generate common electronic design pattern Q&A."""
    patterns = [
        {
            "q": "How do I design a 3.3V to 5V I2C level shifter?",
            "a": """The simplest and most reliable I2C level shifter uses two N-channel MOSFETs (BSS138 or 2N7002) with pull-up resistors:

**Circuit (per I2C line — repeat for SDA and SCL)**:
- MOSFET: BSS138 (SOT-23, 50V, 200mA)
- Gate: Connect to 3.3V (low-voltage side supply)
- Source: Connect to 3.3V side I2C line
- Drain: Connect to 5V side I2C line
- Pull-up on 3.3V side: 4.7K to 3.3V
- Pull-up on 5V side: 4.7K to 5V

**How it works**:
1. **3.3V side drives LOW**: MOSFET source goes low, Vgs > Vth, MOSFET turns on, pulls 5V side low too. Bidirectional!
2. **5V side drives LOW**: Body diode pulls source (3.3V side) low. Then Vgs > Vth turns MOSFET on fully.
3. **Both sides HIGH**: Pull-ups hold each side at their respective voltage (3.3V and 5V).

**BOM for 2 lines (SDA + SCL)**:
- 2x BSS138 (LCSC: C21394)
- 2x 4.7K 0402 resistors (3.3V pull-ups)
- 2x 4.7K 0402 resistors (5V pull-ups)

**Important notes**:
- Do NOT use TXS0108E for I2C — it's push-pull, not open-drain
- Pull-up values: 4.7K for standard mode (100 kHz), 2.2K for fast mode (400 kHz), 1K for fast mode+ (1 MHz)
- Works for any voltage combination (e.g., 1.8V to 3.3V)
- For multi-channel, consider PCA9306 (dedicated 2-ch I2C level shifter IC)""",
        },
        {
            "q": "How do I add reverse polarity protection to my circuit?",
            "a": """Several methods, from simplest to most efficient:

**Method 1: Series Schottky diode (simplest)**
- Place a Schottky diode (e.g., SS14, 1N5819) in series with the positive supply
- Pros: Simple, 1 component
- Cons: 0.3-0.5V voltage drop, power loss (P = Vf x I)
- Best for: Low-current circuits (<1A) where 0.3V drop is acceptable

**Method 2: P-channel MOSFET (best efficiency)**
- Place a P-MOSFET (e.g., AO3401A, Si2301) in the high-side power path
- Gate to power input (before protection), Source to load VCC, Drain to battery/input
- Add 10K gate-source resistor for defined off state
- Reverse voltage turns off MOSFET (Vgs ≈ 0V when reversed)
- Pros: Very low voltage drop (Rds(on) x I, typically <50mV), no heat
- Cons: 3 components, slightly more complex
- Best for: Battery-powered devices, high-current applications

**Method 3: TVS / Zener clamp**
- Not true reverse polarity protection, but protects against brief transients
- Place TVS diode across power rail

**P-MOSFET circuit detail**:
```
VIN ─── Drain ─┬─ Source ──── VOUT (to circuit)
                │
               Gate
                │
                └── 10K ──── Source (= VOUT)
```
- Normal polarity: Vgs = -(Vin), MOSFET ON, current flows
- Reverse polarity: Vgs = 0V or positive, MOSFET OFF, no current

**Recommended P-MOSFETs**:
- AO3401A: -30V, 4A, SOT-23, 42 mOhm (LCSC: C15127)
- DMG2305UX: -20V, 4.2A, SOT-23, 45 mOhm
- Si2301: -20V, 2.8A, SOT-23, 80 mOhm""",
        },
        {
            "q": "How do I debounce a mechanical switch or button in hardware?",
            "a": """Mechanical switches bounce for 1-10ms when pressed/released, causing multiple false triggers. Two approaches:

**Method 1: RC filter + Schmitt trigger (recommended)**
```
Switch ──┬── R (10K) ──┬── Schmitt trigger input (MCU GPIO with Schmitt)
         │             │
        GND           C (100nF to GND)
```
- Time constant: RC = 10K x 100nF = 1 ms
- The cap smooths the bouncing, Schmitt trigger provides clean edges
- Most MCU GPIO pins have built-in Schmitt triggers
- Total debounce time: ~5-10 ms (safe for typical switches)

**Component values for different debounce times**:
| R | C | RC time | Effective debounce |
|---|---|---------|-------------------|
| 10K | 100nF | 1 ms | ~5 ms |
| 10K | 1 uF | 10 ms | ~50 ms |
| 47K | 100nF | 4.7 ms | ~20 ms |

**Method 2: SR latch with SPDT switch (cleanest)**
- Use a CD4013 or 74HC74 SR flip-flop
- Connect switch Common to GND, NO to S, NC to R
- Output is perfectly clean, no bouncing at all
- Only works with SPDT (changeover) switches

**Method 3: Software debounce (cheapest)**
- No hardware needed beyond the pull-up/pull-down resistor
- Sample switch state in timer interrupt (every 1-5 ms)
- Require N consecutive same readings before accepting state change
- Typical: 5 samples at 5ms interval = 25ms debounce

**Pull-up/pull-down configuration**:
- Active LOW (recommended): 10K pull-up to VCC, switch to GND. MCU reads LOW when pressed.
- Active HIGH: 10K pull-down to GND, switch to VCC. MCU reads HIGH when pressed.
- Most MCUs have internal pull-ups (20K-50K) — enable in GPIO config to save a resistor.

**Recommended**: Use a 10K resistor + 100nF cap on the button, enable the MCU's internal pull-up, and add 5ms software debounce as belt-and-suspenders.""",
        },
        {
            "q": "How do I design a proper decoupling network for a microcontroller?",
            "a": """A proper decoupling network provides clean power to the MCU at all frequencies. Follow this hierarchy:

**Per-pin capacitors** (highest priority):
- **100nF (0.1uF) ceramic cap per VDD/VCC pin**, placed within 2mm of the pin
- Use X5R or X7R dielectric, 0402 or 0603 package
- Via directly to ground plane (no trace to GND, use via-in-pad or adjacent via)
- These handle high-frequency noise (1-100 MHz)
- LCSC: C1525 (Samsung CL05B104KO5NNNC, 0402, 16V)

**Example for STM32F103C8T6 (LQFP-48)**:
- Pin 1 (VBAT): 100nF
- Pin 24 (VDD): 100nF
- Pin 36 (VDD): 100nF
- Pin 48 (VDD): 100nF
- Pin 9 (VDDA): 100nF + 1uF (analog supply needs extra filtering)
- Total: 5x 100nF + 1x 1uF

**Bulk capacitors** (one per power rail):
- **10uF ceramic** near power entry point (handles medium-frequency noise, 100kHz-10MHz)
- **100uF electrolytic** or **22uF ceramic** if powering from long cable or connector
- Place near the VDD pin that's closest to the power input

**VDDA (analog supply)**:
- Extra filtering: 100nF + 1uF, possibly with a ferrite bead (600 ohm @ 100MHz) in series
- This prevents digital noise from corrupting ADC readings
- Ferrite bead: LCSC C1015 (Murata BLM18PG600SN1D, 0603)

**Layout rules**:
1. Place caps on the SAME layer as the MCU (no vias between cap and MCU pin)
2. Via to ground plane directly at the cap's GND pad
3. Keep traces short — inductance kills high-frequency decoupling
4. Use a solid ground plane (don't break it under the MCU)
5. Route VDD traces OVER the ground plane for good return path

**Common mistakes**:
- Long traces from cap to MCU pin (adds inductance, defeats the purpose)
- Missing VBAT decoupling (causes RTC and backup domain issues)
- No bulk cap (brownout during current spikes)
- Breaking ground plane under MCU (creates antenna loops)""",
        },
        {
            "q": "How do I design a voltage divider for ADC input on a 3.3V MCU?",
            "a": """A voltage divider scales a higher voltage down to the MCU's ADC range (typically 0-3.3V).

**Basic formula**: Vout = Vin x R2 / (R1 + R2)

**Example: Measure 0-12V battery voltage with 3.3V ADC**:
- Target: 12V -> 3.3V (or slightly less for safety margin)
- Design for: 12V -> 3.0V (leaves 0.3V margin)
- Ratio: 3.0/12 = 0.25
- Choose R2 = 10K, then R1 = R2 x (12/3.0 - 1) = 10K x 3 = 30K
- Use R1 = 30K (nearest standard: 30K or 33K)
- With R1=33K, R2=10K: Vout = 12 x 10/(33+10) = 2.79V at 12V input
- At 14V (max with safety): Vout = 14 x 10/(33+10) = 3.26V (still under 3.3V)

**Component selection**:
- Use 1% tolerance resistors (mismatch = measurement error)
- R1+R2 total impedance: 10K-100K range
  - Too low (<1K): wastes current (Idivider = Vin/(R1+R2))
  - Too high (>1M): ADC input impedance loading, noise pickup
- For 12V battery: R1=33K + R2=10K = 43K total, I = 12/43K = 0.28mA (fine)

**Protection**:
- Add 100nF cap from ADC input to GND (low-pass filter for noise)
- Add 100 ohm resistor in series between divider and ADC pin (current limiting)
- Optional: TVS or Zener clamp on ADC pin for overvoltage protection

**Software conversion**:
```c
// STM32 HAL, 12-bit ADC (0-4095)
float voltage = (float)adc_value / 4095.0f * 3.3f * (33.0f + 10.0f) / 10.0f;
// Multiply by divider ratio inverse: (R1+R2)/R2 = 43/10 = 4.3
```

**For measuring battery voltage**:
- Add a MOSFET switch to disconnect the divider when not measuring (saves current in sleep mode)
- Or use very high values (R1=330K, R2=100K) to minimize standby current (0.028 mA)
- High-value dividers need longer ADC sample time for accurate readings""",
        },
        {
            "q": "How do I connect an SPI device to a microcontroller?",
            "a": """SPI (Serial Peripheral Interface) uses 4 signals for full-duplex communication:

**Signals**:
| Signal | Direction | Description |
|--------|-----------|-------------|
| SCK (SCLK) | MCU -> Device | Clock signal |
| MOSI (SDI) | MCU -> Device | Data: Master Out, Slave In |
| MISO (SDO) | Device -> MCU | Data: Master In, Slave Out |
| CS (SS, nCS) | MCU -> Device | Chip Select (active LOW) |

**Connections**:
```
MCU                    SPI Device
SCK  ──────────────── SCK
MOSI ──────────────── SDI/MOSI/DIN
MISO ──────────────── SDO/MISO/DOUT
GPIO ──────────────── CS/SS (active low)
3.3V ──────────────── VCC
GND  ──────────────── GND
```

**Multiple devices on same bus**:
- SCK, MOSI, MISO are shared (directly connected)
- Each device gets its own CS pin (separate GPIO per device)
- Only one CS should be LOW at a time

**Pull-ups/pull-downs**:
- CS: 10K pull-up to VCC (keeps device deselected during MCU boot)
- MISO: typically needs no pull-up (driven by slave)
- Leave MISO floating risk: if no device selected, MISO is hi-Z (add 10K pull-up if needed)

**SPI Mode configuration** (critical — must match device datasheet):
| Mode | CPOL | CPHA | Clock Idle | Sample Edge |
|------|------|------|------------|-------------|
| 0 | 0 | 0 | Low | Rising |
| 1 | 0 | 1 | Low | Falling |
| 2 | 1 | 0 | High | Falling |
| 3 | 1 | 1 | High | Rising |
- Mode 0 is most common (W25Q128, BME280, ADS1115)
- Mode 3 is also common (SD cards, some ADCs)
- Check your device's datasheet for the correct mode!

**Speed**:
- Start slow (1-4 MHz) during development
- Increase to device's rated speed once working
- Long wires (>10cm) may require slower speeds

**Layout tips**:
- Keep SPI traces short (<10cm for >10 MHz)
- Add 33-100 ohm series resistors on SCK and MOSI for EMI (optional, helps with ringing)
- Bypass cap (100nF) close to each SPI device's VCC pin
- Ground plane under SPI traces""",
        },
        {
            "q": "How do I design a basic Li-Ion battery charging circuit?",
            "a": """A minimal Li-Ion charging circuit uses 3 ICs: charger + battery protection.

**BOM (all JLCPCB basic parts)**:
| IC | Function | Package | LCSC |
|----|----------|---------|------|
| TP4056 | CC/CV charger (1A) | SOP-8 | C725790 |
| DW01A | Battery protection (OVP, ODP, OCP, SCP) | SOT-23-6 | C14213 |
| FS8205A | Dual N-MOSFET (protection switch) | SOT-23-6 | C32254 |

**Circuit**:
```
USB 5V ─── TP4056 ─── BAT+ ─── DW01A+FS8205A ─── LOAD+
                       │                              │
                    Li-Ion cell                      LOAD
                       │                              │
              GND ─────┴──────────────────────────── LOAD-
```

**TP4056 configuration**:
- RPROG resistor sets charge current: RPROG = 1200 / I_mA
  - 1.2K = 1000mA, 2K = 600mA, 4.7K = 250mA, 10K = 120mA
- CHRG pin: LOW while charging (connect LED + 1K resistor to VCC)
- STDBY pin: LOW when charge complete (connect LED + 1K resistor to VCC)
- Input bypass: 100nF + 10uF on VIN

**DW01A + FS8205A protection**:
- Overcharge: disconnects at 4.3V (protects cell from overvoltage)
- Overdischarge: disconnects at 2.4V (protects cell from deep discharge)
- Overcurrent: disconnects at ~3A
- Short circuit: disconnects in microseconds
- 100nF cap between VCC and VSS of DW01A

**Charge current selection guide**:
| Battery Capacity | Max Charge Rate | RPROG Value |
|-----------------|-----------------|-------------|
| 500 mAh | 500 mA (1C) | 2.4K |
| 1000 mAh | 1000 mA (1C) | 1.2K |
| 2000 mAh | 1000 mA (0.5C) | 1.2K |
| 3000 mAh | 1000 mA (0.33C) | 1.2K |

**Important safety notes**:
- NEVER charge Li-Ion above 4.2V or below 0C
- Maximum charge rate should not exceed 1C (capacity in mAh = max charge mA)
- Add a PTC fuse (1A-2A) in series for extra protection
- TP4056 can get hot — provide adequate copper area for thermal dissipation
- For production, consider MCP73831 (Microchip) for better quality/documentation""",
        },
        {
            "q": "How do I implement a power-on reset circuit for a microcontroller?",
            "a": """Most modern MCUs have built-in power-on reset (POR), but external reset circuits help with:
- Slow-rising power supplies
- Brown-out conditions
- Manual reset button
- Watchdog reset

**Method 1: RC reset circuit (simplest)**
```
VCC ─── R (10K) ──┬── nRST (MCU reset pin, active LOW)
                   │
                  C (100nF) ── GND
                   │
             [Push button] ── GND (optional)
```
- Time constant: 10K x 100nF = 1ms (VCC reaches 63% in 1ms)
- nRST stays LOW until cap charges above the reset threshold (~70% VCC)
- Add 100nF cap to filter noise on the reset line
- Push button shorts cap to GND for manual reset

**Method 2: Dedicated reset IC (recommended for production)**
- **STM809** or **STM1061**: Simple voltage supervisor, SOT-23
- Holds reset LOW until VCC exceeds threshold (2.93V for 3.3V systems)
- Release delay: ~240ms after threshold (clean startup)
- Also monitors for brown-out (re-asserts reset if VCC drops)

- **MAX809** (Analog Devices): Same function, popular alternative
- **TPS3839** (TI): Ultra-low power (0.15 uA), SOT-23

**Connection for reset IC**:
```
VCC ─── VCC pin ─── Reset IC ─── nRST output ──── MCU nRST
GND ─── GND pin     │
                    100nF bypass cap
```

**Method 3: MCU internal POR + bypass cap only**
- Most STM32, ESP32, RP2040 have adequate internal POR
- Just add 100nF cap on nRST pin and optional reset button
- STM32: 100nF on NRST, 100nF on BOOT0 (pulled to GND via 10K)

**When you NEED an external reset IC**:
- Power supply rise time > 10ms
- Multiple ICs need synchronized power-up sequencing
- Safety-critical applications (industrial, automotive)
- Brown-out is possible (battery-powered, long cables)

**Reset button circuit**:
- Add 100nF cap across button for debouncing
- 10K pull-up to VCC (or use internal pull-up if available)
- Some MCUs have specific reset pin requirements — check the datasheet""",
        },
        {
            "q": "What's the proper way to connect an I2C bus with multiple devices?",
            "a": """I2C is a 2-wire bus (SDA + SCL) that supports multiple devices. Here's the proper setup:

**Basic wiring**:
```
VCC ──┬── 4.7K ──┬── 4.7K ──┐
      │          │           │
MCU ──┤          │           │
 SDA ─┼──────────┼───────────┼─── Device 1 SDA ─── Device 2 SDA ─── ...
 SCL ─┼──────────┤           │
      │          └───────────┼─── Device 1 SCL ─── Device 2 SCL ─── ...
 GND ─┴──────────────────────┴─── Device 1 GND ─── Device 2 GND
```

**Pull-up resistors** (one pair for the entire bus, NOT per device):
| I2C Speed | Bus Capacitance | Pull-up Value |
|-----------|----------------|---------------|
| Standard (100 kHz) | <100 pF | 10K |
| Standard (100 kHz) | 100-400 pF | 4.7K |
| Fast (400 kHz) | <100 pF | 4.7K |
| Fast (400 kHz) | 100-400 pF | 2.2K |
| Fast Mode+ (1 MHz) | <100 pF | 2.2K |
| Fast Mode+ (1 MHz) | 100-400 pF | 1K |

**Formula**: R_min = (VCC - 0.4V) / 3mA, R_max = t_rise / (0.8473 x C_bus)

**Address conflicts** (common pitfall):
- Each device must have a unique address
- Some devices have configurable addresses (via ADDR pin):
  - BME280: 0x76 or 0x77
  - ADS1115: 0x48, 0x49, 0x4A, 0x4B (4 options)
  - INA226: 16 address options
- If you need two identical devices with the same fixed address, use an **I2C multiplexer** (TCA9548A, 8 channels)

**Bus length limits**:
- Standard mode: up to 1-2 meters with proper pull-ups
- Fast mode: up to 30-50 cm
- Over 50cm: reduce speed, lower pull-up resistance, or use I2C bus extender (P82B715)
- For long distances (>2m): switch to RS-485 or CAN bus

**Voltage level mixing**:
- All devices on the bus MUST use the same voltage level
- For mixed 3.3V + 5V devices: use BSS138 level shifter or PCA9306
- Never connect a 3.3V device directly to a 5V I2C bus

**Common mistakes**:
- Pull-ups on every breakout board (paralleled pull-ups = too low resistance). Remove extras.
- Missing pull-ups entirely (some breakout boards don't include them)
- SCL and SDA swapped (nothing works but nothing breaks — just swap and retry)
- Wrong address (use an I2C scanner sketch to find devices)""",
        },
        {
            "q": "How do I select a fuse or polyfuse for circuit protection?",
            "a": """Fuses protect against overcurrent due to shorts or faults. Selection criteria:

**Key parameters**:
1. **Rated current** (I_hold): Maximum continuous current without tripping
2. **Trip current** (I_trip): Minimum current that causes the fuse to blow/trip
3. **Voltage rating**: Must exceed your circuit's maximum voltage
4. **Breaking capacity**: Maximum fault current the fuse can safely interrupt
5. **Speed**: Fast-blow vs slow-blow (time-delay)

**Selection formula**: I_hold = 1.25 x I_normal (25% margin above normal operating current)

**Example: USB-powered device drawing 500mA max**:
- I_hold = 1.25 x 0.5 = 0.625A
- Choose: 750mA polyfuse (PTC) or 1A fuse
- Voltage: >5V (USB VBUS)

**Fuse types**:

| Type | Resets? | Speed | Best For |
|------|---------|-------|----------|
| Glass (5x20mm) | No | Fast or Slow | Prototyping, bench supplies |
| SMD (0603-1812) | No | Fast | Production boards |
| **PTC/Polyfuse** | **Yes** | Slow (seconds) | USB, battery, any auto-reset needed |
| Ceramic (5x20mm) | No | Fast | High breaking capacity |

**PTC/Polyfuse (self-resetting)** — most popular for electronics:
- Trips at overcurrent, resets when current is removed
- Trip time: 0.5-10 seconds depending on overcurrent magnitude
- Does NOT provide instantaneous protection (use TVS for that)

**Common PTC values and LCSC parts**:
| Rating | Package | LCSC | Application |
|--------|---------|------|-------------|
| 200mA | 0805 | C89657 | Sensor inputs |
| 500mA | 1206 | C70075 | USB device |
| 1A | 1812 | C70068 | Motor, LED strip |
| 2A | 1812 | C193019 | Power input |

**USB-specific requirements**:
- USB 2.0 device: 500mA max, use 500mA-750mA PTC on VBUS
- USB 3.0: 900mA max
- USB PD: up to 5A at 20V — use appropriate fuse rating

**Placement**:
- Place fuse as close to the power input connector as possible
- Before any other protection (TVS, regulator, etc.)
- Fuse -> TVS -> Regulator -> Circuit (this order)

**TVS + Fuse combination**:
- Fuse alone doesn't protect against voltage spikes
- TVS alone doesn't protect against sustained overcurrent
- Use both: Fuse in series, TVS in parallel (after fuse)""",
        },
    ]
    return [msg(p["q"], p["a"]) for p in patterns]


def generate_troubleshooting_qa() -> list[dict]:
    """Generate common troubleshooting Q&A for electronics."""
    issues = [
        {
            "q": "My I2C device is not responding. How do I debug this?",
            "a": """I2C communication failures are common. Systematic debug checklist:

**Step 1: Verify wiring**
- SDA and SCL connected correctly (not swapped)
- GND connected between MCU and device
- VCC connected and correct voltage
- Pull-up resistors present (check with multimeter: SDA and SCL should read ~VCC with no traffic)

**Step 2: Check address**
- Run an I2C scanner sketch to find all devices on the bus
- Common address mistakes:
  - Arduino/ESP32: 7-bit address (e.g., 0x68)
  - Some datasheets show 8-bit address (left-shifted, e.g., 0xD0 = 0x68 << 1)
  - Check if ADDR pin is correctly connected (floating = undefined address)
- BME280: 0x76 (SDO=GND) or 0x77 (SDO=VCC)
- MPU6050: 0x68 (AD0=GND) or 0x69 (AD0=VCC)

**Step 3: Check pull-ups**
- Measure resistance from SDA to VCC and SCL to VCC (should be 1K-10K)
- Too many pull-ups? Multiple breakout boards each with their own pull-ups in parallel = too low total resistance
- No pull-ups? The bus won't work at all (SDA/SCL float randomly)

**Step 4: Check voltage levels**
- Is the device 3.3V and the MCU 5V (or vice versa)? You need a level shifter.
- Measure VCC at the device pin (not just at the power supply)
- Check that the device is actually powered (some have enable pins)

**Step 5: Oscilloscope/logic analyzer**
- Check SCL: clean clock signal at expected frequency?
- Check SDA: data transitions visible?
- Look for ACK/NACK: after address byte, device should pull SDA LOW (ACK)
- If you see NACK after address: wrong address or device not responding
- If SCL is stuck LOW: bus is locked. Power cycle the device, or send 9 clock pulses on SCL to reset.

**Step 6: Software checks**
- Correct I2C peripheral selected (MCU may have multiple I2C controllers)
- Clock speed not too high (start at 100 kHz)
- GPIO pins configured for open-drain alternate function (not push-pull!)
- Check for bus timeout in firmware (some I2C peripherals need explicit timeout handling)

**Common fixes**:
1. Add 4.7K pull-ups if missing
2. Remove duplicate pull-ups (desolder from breakout boards)
3. Lower clock speed to 100 kHz
4. Power cycle the device
5. Check solder joints under microscope""",
        },
        {
            "q": "My STM32 won't program or connect to the debugger. What should I check?",
            "a": """STM32 programming/debug failure is usually a wiring or configuration issue:

**SWD interface check** (2-wire debug):
- SWDIO: Connected to PA13 (check schematic)
- SWCLK: Connected to PA14 (check schematic)
- GND: Connected between debugger and target
- VCC/VREF: Connected (some debuggers need this for level detection)
- NRST: Connected (optional but helpful for forced reset)

**Common causes and fixes**:

1. **PA13/PA14 remapped in firmware**:
   - If your code reconfigures PA13/PA14 as GPIO, SWD is disabled
   - Fix: Hold BOOT0 HIGH during reset (enters bootloader mode, SWD works)
   - STM32CubeProgrammer: "Connect under Reset" option

2. **BOOT0 pin misconfigured**:
   - BOOT0 should be pulled LOW via 10K resistor for normal boot
   - If floating, MCU may boot from wrong memory region

3. **Power issue**:
   - Measure VDD at the MCU pin (should be 3.0-3.6V)
   - Check all VDD and VSS pins are connected (missing one = random behavior)
   - Check VDDA is connected (even if you don't use ADC)

4. **Debugger speed too high**:
   - Reduce SWD clock in your IDE (try 100 kHz instead of 4 MHz)
   - Long wires between debugger and target need lower speed

5. **Flash read protection enabled**:
   - If RDP Level 1 or 2 is set, debugger access is blocked
   - Level 1: Can be removed (erases all flash) via STM32CubeProgrammer
   - Level 2: PERMANENT lock, chip is bricked for debugging forever

6. **Wrong debugger setting**:
   - Select "SWD" not "JTAG" in your IDE
   - Check target voltage matches (3.3V target needs 3.3V-compatible debugger)
   - ST-Link: try both "Normal" and "Under Reset" connection modes

7. **PCB issues**:
   - Check for solder bridges on QFP/QFN pins (especially VDD/VSS adjacent to SWDIO/SWCLK)
   - Verify the MCU is properly soldered (reflow if in doubt)
   - Check that decoupling caps are present and soldered

**Recovery procedure**:
1. Connect BOOT0 to VCC (3.3V)
2. Power cycle the board
3. Connect via STM32CubeProgrammer using UART or SWD
4. Erase flash / remove read protection
5. Disconnect BOOT0 from VCC (reconnect pull-down)
6. Program normally""",
        },
        {
            "q": "My voltage regulator is overheating. What's wrong?",
            "a": """Regulator overheating is a thermal design issue. Causes and solutions:

**Step 1: Calculate power dissipation**

For **linear regulators** (LM7805, AMS1117, AP2112):
```
P_dissipated = (Vin - Vout) x Iload
```
Example: AMS1117-3.3 with 5V input, 500mA load:
P = (5.0 - 3.3) x 0.5 = 0.85W — this is significant for a SOT-223!

Example: LM7805 with 12V input, 1A load:
P = (12 - 5) x 1.0 = 7W — way too much, needs massive heatsink or switching regulator!

For **switching regulators** (TPS54331, LM2596):
```
P_dissipated = Pout x (1/efficiency - 1)
```
Example: TPS54331, 12V to 5V, 2A, 90% efficiency:
P = (5 x 2) x (1/0.9 - 1) = 10 x 0.111 = 1.1W — spread across IC + inductor + diode

**Step 2: Check thermal limits**

| Package | Max Power (no heatsink, 25C ambient) |
|---------|--------------------------------------|
| SOT-23-5 | 0.3-0.5W |
| SOT-223 | 1.0-1.5W |
| DPAK (TO-252) | 1.5-2.0W |
| TO-220 (no heatsink) | 1.0-2.0W |
| TO-220 (heatsink) | 5-15W |

**Solutions**:

1. **Reduce (Vin - Vout)** — the dropout voltage determines heat:
   - If going from 12V to 3.3V with LDO: 8.7V drop = massive heat
   - Use a buck converter instead (TPS54331, MP1584)
   - Or pre-regulate: 12V -> 5V (buck) -> 3.3V (LDO)

2. **Reduce load current**:
   - Check if all the current is actually needed
   - Add sleep modes for peripherals
   - Use more efficient loads (e.g., LED with PWM instead of resistor dimming)

3. **Improve thermal dissipation**:
   - Add copper pour connected to the regulator's ground/tab pad
   - Use thermal vias under the package
   - Add a heatsink (TO-220)
   - Ensure adequate airflow

4. **Switch to a switching regulator**:
   - LDO with large Vin-Vout drop is always wasteful
   - Buck converter: 85-95% efficiency
   - Buck-boost: maintains output even when Vin drops below Vout

**Rule of thumb**: If (Vin - Vout) x Iload > 0.5W for SOT-223 or > 1W for DPAK, consider a switching regulator.""",
        },
        {
            "q": "My ADC readings are noisy or inaccurate. How do I improve them?",
            "a": """ADC noise and inaccuracy have multiple causes. Address them systematically:

**Hardware improvements**:

1. **Separate analog power supply (VDDA)**:
   - Use a ferrite bead (600 ohm @ 100MHz) + 10uF + 100nF between VDD and VDDA
   - This prevents digital switching noise from reaching the ADC
   - Never connect VDDA directly to VDD without filtering

2. **Input impedance matching**:
   - STM32 ADC needs <10K source impedance (check your MCU's datasheet)
   - If reading through a voltage divider (e.g., 33K + 10K), add a 100nF cap at the ADC pin
   - Or add an op-amp buffer (MCP6001, unity gain) between divider and ADC

3. **Bypass capacitors**:
   - 100nF on VDDA and VREF pins (as close as possible)
   - 100nF at the sensor/source
   - Optional: 1nF-10nF at the ADC input pin (low-pass filter)

4. **Ground plane**:
   - Solid ground plane under the ADC traces
   - No digital traces routed under or near analog traces
   - Separate analog and digital ground planes, connected at a single point near the MCU

5. **Signal routing**:
   - Keep analog traces short
   - Route analog traces away from switching regulators, crystals, and digital buses
   - Use guard traces (grounded traces) around sensitive analog signals

**Software improvements**:

1. **Oversampling + averaging**:
   ```c
   uint32_t sum = 0;
   for (int i = 0; i < 16; i++) {
       sum += HAL_ADC_GetValue(&hadc1);
   }
   uint16_t avg = sum / 16;  // 16x oversampling = ~2 extra bits of resolution
   ```
   - 4x oversampling = +1 bit effective resolution
   - 16x oversampling = +2 bits
   - 64x oversampling = +3 bits

2. **Increase sample time**:
   - Longer ADC sample time = more accurate for high-impedance sources
   - STM32: set sample time to 239.5 or 480 cycles for voltage divider inputs

3. **Calibration**:
   - Run ADC self-calibration at startup (STM32: HAL_ADCEx_Calibration_Start)
   - For absolute accuracy: measure a known reference voltage and calculate offset/gain correction

4. **Digital filtering**:
   - Moving average (simple, adds latency)
   - Exponential moving average: `filtered = alpha * new + (1-alpha) * filtered` (alpha = 0.1-0.3)
   - Median filter (removes spikes): take 5 samples, sort, use middle value

**Common noise sources to eliminate**:
- Switching regulator: use LDO for analog power, or add LC filter
- PWM outputs: synchronize ADC sampling to PWM dead zones
- Motor drivers: massive noise source, needs proper filtering
- Wi-Fi/BLE transmission: ESP32 ADC is very noisy during Wi-Fi TX""",
        },
    ]
    return [msg(p["q"], p["a"]) for p in issues]


def generate_passive_value_qa() -> list[dict]:
    """Generate Q&A about standard passive component values and selection."""
    qa_pairs = []

    # E24 resistor series
    e24_values = [
        "1.0", "1.1", "1.2", "1.3", "1.5", "1.6", "1.8", "2.0", "2.2", "2.4",
        "2.7", "3.0", "3.3", "3.6", "3.9", "4.3", "4.7", "5.1", "5.6", "6.2",
        "6.8", "7.5", "8.2", "9.1",
    ]

    # Common resistor applications
    resistor_apps = [
        ("1K", "LED current limiting (3.3V supply, 2V LED: I = (3.3-2)/1K = 1.3mA), pull-down for MOSFET gates"),
        ("2.2K", "I2C pull-ups for fast mode (400 kHz), LED current limiting at higher currents"),
        ("4.7K", "I2C pull-ups for standard mode (100 kHz), general pull-up/pull-down"),
        ("10K", "General pull-up/pull-down, voltage dividers, MOSFET gate pull-down, reset circuits"),
        ("47K", "High-impedance pull-ups, audio circuits, weak pull-ups for low power"),
        ("100K", "Very weak pull-ups, touch sensing, high-impedance voltage dividers"),
        ("330", "USB series termination resistors (for impedance matching), LED current limiting"),
        ("0", "Zero-ohm jumper — used to bridge traces, optional component placement, testing points"),
        ("120", "CAN bus termination resistor (one at each end of the bus)"),
        ("5.1K", "USB-C CC1/CC2 pull-down resistors (identifies device as UFP)"),
    ]

    for value, application in resistor_apps:
        q = f"What is a {value} ohm resistor commonly used for in electronics?"
        a = f"A {value} ohm resistor is commonly used for: {application}."
        a += f"\n\nAvailable in standard packages: 0402 (1/16W), 0603 (1/10W), 0805 (1/8W), 1206 (1/4W)."
        a += " Use 1% tolerance (F suffix) unless you need higher precision."
        qa_pairs.append(msg(q, a))

    # Common capacitor applications
    cap_apps = [
        ("100nF (0.1uF)", "Bypass/decoupling capacitor for IC power pins. Place one as close as possible to each VDD/VCC pin. The most frequently used capacitor in digital circuits.", "X5R or X7R", "0402 or 0603"),
        ("10uF", "Bulk decoupling, LDO output capacitor, power supply filtering. DC bias effect reduces effective capacitance — a 10uF 0805 at 3.3V bias may only give 5-7uF.", "X5R", "0805 or 1206"),
        ("1uF", "Analog supply filtering (VDDA), charge pump capacitors, local bypassing for sensitive ICs.", "X5R or X7R", "0402 or 0603"),
        ("22uF", "USB VBUS decoupling, switching regulator output, bulk energy storage. Often used as LDO output cap.", "X5R", "0805 or 1206"),
        ("100pF", "RF decoupling, high-frequency filtering, ESD protection circuits. C0G dielectric preferred.", "C0G/NP0", "0402"),
        ("10nF", "EMI filtering, snubber circuits, timing circuits. Good general-purpose filter value.", "X7R", "0402 or 0603"),
        ("4.7nF", "USB shield filtering (in parallel with 1M resistor), RC filter applications.", "X7R", "0402"),
        ("22pF", "Crystal load capacitors (for 8MHz with 12.5pF load spec), RF matching.", "C0G/NP0", "0402"),
    ]

    for value, application, dielectric, package in cap_apps:
        q = f"What is a {value} capacitor used for and what dielectric should I choose?"
        a = f"A {value} capacitor is used for: {application}"
        a += f"\n\nRecommended dielectric: **{dielectric}**. Typical package: **{package}**."
        a += "\n\nDielectric guide: C0G/NP0 for precision/timing, X7R for general filtering, X5R for bulk decoupling."
        qa_pairs.append(msg(q, a))

    # Standard inductor values for switching regulators
    inductor_qa = [
        {
            "q": "How do I select an inductor for a buck converter?",
            "a": """Inductor selection for a buck converter depends on the switching frequency, input/output voltage, and load current:

**Formula**: L = (Vin - Vout) x Vout / (Vin x fsw x delta_I)

Where:
- Vin = input voltage
- Vout = output voltage
- fsw = switching frequency
- delta_I = desired ripple current (typically 20-40% of Iload)

**Example**: 12V to 5V, 2A load, 500kHz switching, 30% ripple:
- delta_I = 0.3 x 2 = 0.6A
- L = (12-5) x 5 / (12 x 500000 x 0.6) = 35/3600000 = 9.7 uH
- Choose: **10 uH** (nearest standard value)

**Saturation current (Isat)**: Must be > Iload + delta_I/2 = 2 + 0.3 = 2.3A minimum. Choose 3A+ for margin.

**DCR (DC resistance)**: Lower is better. Target <100 mOhm for 2A+ converters. P_loss = DCR x I^2.

**Common inductor values by converter IC**:
| Converter IC | Recommended L | Isat Min |
|-------------|---------------|----------|
| TPS54331 (570kHz) | 10-22 uH | 4A |
| MP1584 (1.5MHz) | 4.7-10 uH | 4A |
| LM2596 (150kHz) | 33-68 uH | 4A |
| TPS63020 (2.4MHz) | 1-2.2 uH | 5A |""",
        },
    ]
    for item in inductor_qa:
        qa_pairs.append(msg(item["q"], item["a"]))

    return qa_pairs


def generate_pcb_design_qa() -> list[dict]:
    """Generate PCB design best practices Q&A."""
    pcb_qa = [
        {
            "q": "What are the minimum trace widths and clearances for JLCPCB?",
            "a": """JLCPCB design rules for standard PCB (2-layer, 1oz copper):

**Minimum specifications**:
| Parameter | Minimum | Recommended |
|-----------|---------|-------------|
| Trace width | 0.127mm (5mil) | 0.2mm (8mil)+ |
| Trace spacing | 0.127mm (5mil) | 0.2mm (8mil)+ |
| Via drill | 0.2mm (8mil) | 0.3mm (12mil) |
| Via annular ring | 0.15mm (6mil) | 0.25mm (10mil) |
| Via pad diameter | drill + 0.3mm | drill + 0.5mm |
| Board thickness | 0.4mm | 1.6mm (standard) |
| Copper to edge | 0.3mm | 0.5mm |
| Hole to hole | 0.5mm | 0.8mm |

**Trace width for current carrying**:
| Current | 1oz Cu (External) | 2oz Cu (External) |
|---------|-------------------|-------------------|
| 0.5A | 0.25mm (10mil) | 0.15mm |
| 1.0A | 0.5mm (20mil) | 0.3mm |
| 2.0A | 1.0mm (40mil) | 0.6mm |
| 3.0A | 1.5mm (60mil) | 1.0mm |
| 5.0A | 2.5mm (100mil) | 1.5mm |

**Impedance-controlled traces** (for USB, HDMI, etc.):
- USB 2.0: 90 ohm differential pair (trace width depends on stack-up)
- JLCPCB controlled impedance: use their stack-up calculator
- Typical 2-layer: 0.3mm trace width, 0.15mm gap for 90 ohm differential

**Tips for passing JLCPCB DRC**:
1. Run JLCPCB's own DRC plugin for KiCad (available on their site)
2. Set your KiCad DRC rules to match JLCPCB minimums
3. Panelize using JLCPCB's panel tool (not your own)
4. Avoid micro-vias and blind/buried vias (extra cost)
5. Standard via: 0.3mm drill, 0.6mm pad (safe, cheap)""",
        },
        {
            "q": "How should I route a USB 2.0 differential pair on a 2-layer PCB?",
            "a": """USB 2.0 requires a 90 ohm differential impedance pair. On a typical 2-layer 1.6mm PCB:

**Trace geometry** (1oz copper, FR4, 1.6mm board):
- Trace width: 0.3-0.4mm (12-16mil)
- Trace gap: 0.15-0.2mm (6-8mil)
- Reference plane: ground pour on the opposite layer directly underneath

**Routing rules**:
1. **Keep D+ and D- length matched**: within 0.15mm (6mil) of each other
2. **Route as a pair**: traces always parallel, same gap throughout
3. **Maintain ground reference**: ensure solid copper pour (ground) under the entire USB pair on the other layer
4. **Avoid vias**: route D+/D- entirely on one layer if possible
5. **No stubs**: route directly from connector to MCU/USB IC, no T-junctions
6. **Maximum length**: <50mm for USB 2.0 Full-speed (12 Mbps), <20mm for High-speed (480 Mbps)
7. **Avoid crossing other traces**: especially clock signals and switching regulator traces

**Series resistors** (optional for USB 2.0):
- Some MCUs have internal termination (STM32, ESP32) — no series resistors needed
- If needed: 27 ohm series on each line, placed close to the MCU
- USB 2.0 HS: typically no series resistors

**ESD protection**:
- Place USBLC6-2SC6 (or similar) as close to the USB connector as possible
- Route D+/D- through the ESD IC, not around it

**Connector placement**:
- USB connector on board edge
- Ground the connector shield: 1M + 4.7nF to ground
- Place ESD protection within 5mm of the connector

**Common mistakes**:
- Not having a ground plane under USB traces (impedance goes wrong)
- Very different D+/D- trace lengths (causes signal skew)
- Routing USB near a crystal oscillator (crosstalk)
- Forgetting CC pull-down resistors on USB-C (host won't provide power)""",
        },
        {
            "q": "What is a good ground plane strategy for a mixed-signal PCB?",
            "a": """Mixed-signal PCBs (analog + digital) need careful grounding to prevent digital noise from corrupting analog signals.

**2-layer board strategy**:
- Use a **single, unbroken ground plane** on the bottom layer
- Route all signals on the top layer
- Keep analog and digital circuits in separate physical areas
- Don't split the ground plane (this is a myth — a single solid plane is better for most designs)
- Return currents follow the path of least impedance, which is directly under the signal trace

**4-layer board strategy** (better, recommended for ADC/DAC designs):
```
Layer 1: Signal (top) — components, routing
Layer 2: Ground plane (solid, unbroken)
Layer 3: Power plane (VCC, 3.3V, etc.)
Layer 4: Signal (bottom) — routing overflow
```

**Key rules**:
1. **Never split the ground plane under the MCU**: The MCU's internal analog and digital grounds connect to the same pin. A split creates a current loop antenna.

2. **Separate by region, not by plane split**:
   - Place analog components (ADC, op-amps, sensors) in one area of the board
   - Place digital components (MCU, USB, clock) in another area
   - Place switching regulators in a third area (corner, away from analog)

3. **Filter between analog and digital power**, not ground:
   - VDDA from VDD through a ferrite bead + capacitors
   - One ground plane, NOT separate analog/digital grounds

4. **Star-point connection** (if you must split ground):
   - Connect analog and digital grounds at ONE point, directly under the MCU
   - This is the only scenario where a ground split makes sense

5. **Component placement order** (by priority):
   - Switching regulator first (corner of board, inductor placement)
   - MCU in the center
   - Crystal close to MCU
   - Decoupling caps next to MCU pins
   - Analog sensors away from switching noise
   - Connectors on board edges

6. **Via stitching**: Add ground vias around the board perimeter and in open areas to connect top and bottom ground fills.""",
        },
    ]
    return [msg(p["q"], p["a"]) for p in pcb_qa]


def generate_application_qa(components: list[Component]) -> list[dict]:
    """Generate application-specific Q&A (what's needed to use component X)."""
    pairs = []

    for comp in components:
        if not comp.notes:
            continue

        q = f"What should I know before using the {comp.name} in my design?"
        a = f"Important design considerations for the {comp.name} ({comp.description}):\n\n"
        a += comp.notes + "\n"

        if comp.specs.get("supply_voltage"):
            a += f"\nSupply voltage: {comp.specs['supply_voltage']}"
        if comp.specs.get("operating_temp"):
            a += f"\nOperating temperature: {comp.specs['operating_temp']}"
        if comp.alternatives:
            a += f"\nAlternatives to consider: {', '.join(comp.alternatives[:3])}"
        if comp.lcsc:
            a += f"\nLCSC part number: {comp.lcsc}"

        pairs.append(msg(q, a))

    return pairs


# ---------------------------------------------------------------------------
# High-volume combinatorial generators
# ---------------------------------------------------------------------------


def generate_use_case_qa(components: list[Component]) -> list[dict]:
    """Generate 'Can I use X for Y?' and 'Which component for Y?' Q&A pairs."""
    pairs = []

    use_cases_by_category = {
        "mcu": [
            ("a battery-powered sensor node", lambda c: "low" in c.specs.get("supply_current", "").lower() or "ultra" in c.description.lower() or "M0" in c.specs.get("core", "")),
            ("a motor control application", lambda c: "can" in " ".join(c.interfaces).lower() or "motor" in c.description.lower() or "pwm" in c.description.lower() or "fdcan" in " ".join(c.interfaces).lower()),
            ("a USB HID device (keyboard/mouse)", lambda c: "usb" in " ".join(c.interfaces).lower()),
            ("a Wi-Fi IoT project", lambda c: "wi-fi" in " ".join(c.interfaces).lower() or "wifi" in " ".join(c.interfaces).lower()),
            ("a Bluetooth Low Energy wearable", lambda c: "ble" in " ".join(c.interfaces).lower() or "bluetooth" in " ".join(c.interfaces).lower()),
            ("real-time audio processing", lambda c: "i2s" in " ".join(c.interfaces).lower() or "dsp" in c.description.lower()),
            ("a beginner Arduino-style project", lambda c: "avr" in c.specs.get("core", "").lower() or "arduino" in c.description.lower() or c.name == "ATmega328P-AU"),
            ("running FreeRTOS with a TCP/IP stack", lambda c: int(c.specs.get("flash", "0").split()[0]) >= 128 if c.specs.get("flash", "").split() else False),
            ("a CAN bus automotive application", lambda c: "can" in " ".join(c.interfaces).lower()),
            ("high-speed data acquisition", lambda c: "adc" in c.description.lower() or int(c.specs.get("max_frequency", "0").split()[0]) >= 100 if c.specs.get("max_frequency", "").split() else False),
        ],
        "regulator": [
            ("powering an ESP32 from a Li-Ion battery (3.0-4.2V)", lambda c: "3.3" in c.specs.get("output_voltage", "") and "ldo" in c.description.lower()),
            ("a 12V to 5V conversion at 2A", lambda c: "12" in c.specs.get("input_voltage", "") or "28" in c.specs.get("input_voltage", "") or "36" in c.specs.get("input_voltage", "")),
            ("a noise-sensitive analog circuit", lambda c: "noise" in c.description.lower() or "noise" in str(c.specs)),
            ("a battery-powered device needing minimal quiescent current", lambda c: "ua" in c.specs.get("quiescent_current", "").lower()),
            ("powering 3.3V logic from a 5V USB supply", lambda c: "3.3" in c.specs.get("output_voltage", "")),
        ],
        "opamp": [
            ("an audio preamplifier circuit", lambda c: "audio" in c.description.lower() or float(c.specs.get("gbw", "0").split()[0]) >= 10 if c.specs.get("gbw", "").split() else False),
            ("a precision voltage reference buffer", lambda c: "low" in c.specs.get("input_offset_voltage", "").lower() or "precision" in c.description.lower()),
            ("a battery-powered sensor signal conditioning", lambda c: "rail" in c.description.lower() or "rrio" in c.description.lower() or "cmos" in c.description.lower()),
            ("an active filter at 100 kHz", lambda c: float(c.specs.get("gbw", "0").split()[0]) >= 1 if c.specs.get("gbw", "").split() else False),
        ],
        "mosfet": [
            ("switching a 12V LED strip from a 3.3V MCU", lambda c: "n-channel" in c.specs.get("type", "").lower() and float(c.specs.get("vgs_threshold", "99").split("V")[0].split()[-1].replace("(", "")) <= 2.0 if c.specs.get("vgs_threshold") else False),
            ("reverse polarity protection", lambda c: "p-channel" in c.specs.get("type", "").lower()),
            ("a high-current motor driver H-bridge", lambda c: "n-channel" in c.specs.get("type", "").lower() and "to-220" in c.specs.get("package", "").lower()),
            ("a logic-level load switch", lambda c: "logic" in c.specs.get("type", "").lower() or "logic" in c.description.lower()),
            ("level shifting with a discrete MOSFET", lambda c: c.name in ("BSS138", "2N7002")),
        ],
        "sensor": [
            ("a weather station (temperature, humidity, pressure)", lambda c: "bme" in c.name.lower() or ("temperature" in c.description.lower() and "humidity" in c.description.lower())),
            ("a drone flight controller (IMU)", lambda c: "accelero" in c.description.lower() or "gyro" in c.description.lower() or "imu" in c.description.lower()),
            ("measuring battery current consumption", lambda c: "current" in c.description.lower() or "ina" in c.name.lower()),
            ("a precision temperature measurement (+/-0.1C)", lambda c: "tmp117" in c.name.lower() or "0.1" in c.specs.get("temperature_accuracy", "")),
            ("a digital scale / load cell project", lambda c: "load" in c.description.lower() or "hx711" in c.name.lower()),
            ("measuring light levels for auto-brightness", lambda c: "light" in c.description.lower() or "lux" in c.description.lower()),
            ("a contactless rotary encoder (knob)", lambda c: "rotary" in c.description.lower() or "magnetic" in c.description.lower()),
        ],
    }

    for comp in components:
        cat_uses = use_cases_by_category.get(comp.category, [])
        for use_case, matcher in cat_uses:
            try:
                is_suitable = matcher(comp)
            except (ValueError, IndexError, TypeError, AttributeError):
                is_suitable = False

            q = f"Can I use the {comp.name} for {use_case}?"
            if is_suitable:
                a = f"Yes, the {comp.name} is suitable for {use_case}. "
                a += f"It is a {comp.description}."
                key_specs = [f"- {k.replace('_', ' ').title()}: {v}" for k, v in list(comp.specs.items())[:5]]
                if key_specs:
                    a += "\n\nRelevant specs:\n" + "\n".join(key_specs)
                if comp.lcsc:
                    a += f"\n\nLCSC: {comp.lcsc}"
            else:
                a = f"The {comp.name} may not be the best choice for {use_case}. "
                a += f"It is a {comp.description}. "
                # Find a better alternative from the same category
                better = [c for c in components if c.category == comp.category and c.name != comp.name]
                for alt in better:
                    try:
                        if matcher(alt):
                            a += f"Consider the {alt.name} instead ({alt.description})."
                            break
                    except (ValueError, IndexError, TypeError, AttributeError):
                        continue
                else:
                    a += "Check the specific requirements against the component's datasheet."

            pairs.append(msg(q, a))

    return pairs


def generate_interface_qa(components: list[Component]) -> list[dict]:
    """Generate Q&A about connecting components together."""
    pairs = []

    # For each sensor/peripheral with I2C, generate connection questions
    for comp in components:
        if "I2C" in comp.interfaces and comp.category in ("sensor", "memory", "display", "logic", "timer"):
            for mcu in components:
                if mcu.category != "mcu":
                    continue
                if "I2C" not in mcu.interfaces:
                    continue
                # Only generate for a subset to avoid explosion
                if hash(comp.name + mcu.name) % 3 != 0:
                    continue

                q = f"How do I connect the {comp.name} to the {mcu.name} via I2C?"
                a = f"To connect the {comp.name} to the {mcu.name} via I2C:\n\n"
                a += f"**Wiring**:\n"
                a += f"- {comp.name} SDA -> {mcu.name} SDA pin\n"
                a += f"- {comp.name} SCL -> {mcu.name} SCL pin\n"
                a += f"- {comp.name} VCC -> {comp.specs.get('supply_voltage', '3.3V').split('to')[-1].strip().split('(')[0].strip() if 'to' in comp.specs.get('supply_voltage', '') else '3.3V'}\n"
                a += f"- {comp.name} GND -> GND\n"
                a += f"- Pull-up resistors: 4.7K to VCC on SDA and SCL\n\n"
                if comp.specs.get("i2c_address"):
                    a += f"**I2C Address**: {comp.specs['i2c_address']}\n"
                mcu_voltage = mcu.specs.get("supply_voltage", "")
                comp_voltage = comp.specs.get("supply_voltage", "")
                if mcu_voltage and comp_voltage:
                    a += f"\n**Voltage compatibility**: {mcu.name} operates at {mcu_voltage}, {comp.name} operates at {comp_voltage}. "
                    if "5" in mcu_voltage and "3.6" in comp_voltage:
                        a += "You may need a level shifter if the MCU I/O is 5V."
                    else:
                        a += "These should be directly compatible at 3.3V."

                pairs.append(msg(q, a))

        # SPI connections
        if "SPI" in comp.interfaces and comp.category in ("sensor", "memory", "display", "rf"):
            for mcu in components:
                if mcu.category != "mcu":
                    continue
                if "SPI" not in mcu.interfaces:
                    continue
                if hash(comp.name + mcu.name + "spi") % 5 != 0:
                    continue

                q = f"How do I connect the {comp.name} to the {mcu.name} via SPI?"
                a = f"To connect the {comp.name} to the {mcu.name} via SPI:\n\n"
                a += f"**Wiring**:\n"
                a += f"- {comp.name} MOSI (SDI/DIN) -> {mcu.name} SPI MOSI\n"
                a += f"- {comp.name} MISO (SDO/DOUT) -> {mcu.name} SPI MISO\n"
                a += f"- {comp.name} SCK (SCLK) -> {mcu.name} SPI SCK\n"
                a += f"- {comp.name} CS (NSS/SS) -> {mcu.name} GPIO (any available)\n"
                a += f"- {comp.name} VCC -> 3.3V\n"
                a += f"- {comp.name} GND -> GND\n\n"
                a += f"**Notes**: Configure the {mcu.name} SPI peripheral for the correct mode (check {comp.name} datasheet for CPOL/CPHA)."
                if comp.specs.get("spi_clock"):
                    a += f" Maximum SPI clock: {comp.specs['spi_clock']}."
                pairs.append(msg(q, a))

    return pairs


def generate_power_budget_qa(components: list[Component]) -> list[dict]:
    """Generate power budget / consumption Q&A pairs."""
    pairs = []

    for comp in components:
        current = comp.specs.get("supply_current") or comp.specs.get("quiescent_current")
        voltage = comp.specs.get("supply_voltage", "")
        if not current:
            continue

        q = f"What is the power consumption of the {comp.name}?"
        a = f"The {comp.name} ({comp.description}) has the following power characteristics:\n\n"
        a += f"- Supply voltage: {voltage}\n" if voltage else ""
        a += f"- Current consumption: {current}\n"
        if comp.specs.get("quiescent_current"):
            a += f"- Quiescent/standby current: {comp.specs['quiescent_current']}\n"

        # Estimate power
        a += f"\nThis information is important for battery life calculations and power supply sizing."
        if comp.lcsc:
            a += f"\nLCSC: {comp.lcsc}"

        pairs.append(msg(q, a))

        # Sleep mode question for MCUs
        if comp.category == "mcu" and comp.specs.get("supply_current"):
            q2 = f"What is the sleep mode current of the {comp.name}?"
            a2 = f"The {comp.name} current consumption details: {comp.specs.get('supply_current', 'Check datasheet')}. "
            a2 += f"Supply voltage range: {voltage}. "
            a2 += "For battery life estimation: Battery_capacity_mAh / sleep_current_mA = hours of standby. "
            a2 += "A CR2032 (220 mAh) at 1 uA sleep current would last ~25 years theoretically (limited by self-discharge to ~5-8 years)."
            pairs.append(msg(q2, a2))

    return pairs


def generate_package_qa(components: list[Component]) -> list[dict]:
    """Generate package selection and soldering Q&A."""
    pairs = []

    soldering_difficulty = {
        "SOT-23": ("easy", "Beginner-friendly SMD. 3 large pads, 0.95mm pitch. Hand-solderable with a standard iron and flux."),
        "SOT-23-5": ("easy", "Very manageable SMD. 5 pads, 0.95mm pitch. Use flux and drag-soldering technique."),
        "SOT-23-6": ("easy", "Similar to SOT-23-5. 6 pads, 0.95mm pitch."),
        "SOIC-8": ("easy", "Great for beginners. Large pads, 1.27mm pitch. Easy drag soldering."),
        "SOIC-16": ("easy", "Easy SMD, just more pins than SOIC-8. 1.27mm pitch. Drag solder with flux."),
        "TSSOP": ("medium", "Moderate difficulty. 0.65mm pitch requires flux, fine tip, and steady hand. Magnification helps."),
        "LQFP-48": ("medium", "Moderate. 0.5mm pitch, requires flux, thin solder, and patience. Check for bridges with magnification."),
        "LQFP-64": ("medium", "Same as LQFP-48 but more pins. Drag soldering technique with plenty of flux."),
        "QFN": ("hard", "Challenging. Pads are under the chip — requires hot air or reflow. Exposed pad needs solder paste and proper thermal connection."),
        "DFN": ("hard", "Similar to QFN. Pads hidden underneath. Hot air or oven reflow required."),
        "WLCSP": ("very hard", "Wafer-level chip-scale package. BGA-style balls, requires reflow oven. Not hand-solderable."),
        "BGA": ("very hard", "Requires reflow oven or professional hot air setup. X-ray inspection recommended."),
        "0402": ("medium", "Small (1.0 x 0.5mm). Needs fine tweezers, flux, and magnification. Not for beginners."),
        "0603": ("easy", "Good SMD starting point (1.6 x 0.8mm). Hand-solderable with a fine tip."),
        "0805": ("easy", "Easy to hand-solder (2.0 x 1.25mm). Great for prototyping."),
        "DIP": ("very easy", "Through-hole, perfect for breadboards and beginners. Just insert and solder."),
        "TO-220": ("very easy", "Large through-hole power package. Easy to solder, supports heatsinks."),
    }

    for comp in components:
        if not comp.packages:
            continue

        for pkg in comp.packages:
            # Match against known packages
            pkg_key = None
            for key in soldering_difficulty:
                if key.lower() in pkg.lower():
                    pkg_key = key
                    break

            if pkg_key:
                difficulty, tips = soldering_difficulty[pkg_key]
                q = f"How hard is it to hand-solder the {comp.name} in {pkg} package?"
                a = f"The {comp.name} in {pkg} package is **{difficulty}** to hand-solder. {tips}"
                if len(comp.packages) > 1:
                    a += f"\n\nAlternative packages for {comp.name}: {', '.join(p for p in comp.packages if p != pkg)}."
                pairs.append(msg(q, a))

    return pairs


def generate_quick_spec_qa(components: list[Component]) -> list[dict]:
    """Generate many short, factoid-style Q&A pairs — the high-volume multiplier."""
    pairs = []

    # Category-specific question templates that produce natural questions
    templates_by_spec: dict[str, list[tuple[str, str]]] = {
        "supply_voltage": [
            ("What voltage does the {name} run on?", "The {name} operates at {val}."),
            ("What is the VCC range for the {name}?", "The {name} VCC range is {val}."),
            ("Can the {name} run at 3.3V?", "The {name} supply voltage range is {val}. {yesno33}"),
        ],
        "max_frequency": [
            ("What is the {name} clock speed?", "The {name} runs at up to {val}."),
            ("How fast is the {name}?", "The {name} maximum clock frequency is {val}."),
        ],
        "flash": [
            ("How much flash does the {name} have?", "The {name} has {val} of flash memory."),
            ("What is the program memory size of the {name}?", "The {name} has {val} of program flash."),
        ],
        "sram": [
            ("How much RAM does the {name} have?", "The {name} has {val} of SRAM."),
        ],
        "core": [
            ("What CPU core does the {name} use?", "The {name} uses a {val} core."),
            ("Is the {name} ARM-based?", "The {name} core is: {val}. {arm_yesno}"),
        ],
        "gpio_count": [
            ("How many GPIOs does the {name} have?", "The {name} has {val} GPIO pins."),
            ("How many I/O pins on the {name}?", "The {name} provides {val} general-purpose I/O pins."),
        ],
        "uart": [
            ("How many UARTs does the {name} have?", "The {name} has {val} UART(s)."),
        ],
        "spi": [
            ("How many SPI interfaces does the {name} have?", "The {name} has {val} SPI interface(s)."),
        ],
        "i2c": [
            ("How many I2C buses does the {name} have?", "The {name} has {val} I2C bus(es)."),
        ],
        "output_voltage": [
            ("What voltage does the {name} output?", "The {name} outputs {val}."),
            ("What is the {name} output voltage?", "The {name} provides an output voltage of {val}."),
        ],
        "max_output_current": [
            ("How much current can the {name} supply?", "The {name} can supply up to {val}."),
            ("What is the {name} current rating?", "The {name} is rated for {val} output current."),
        ],
        "dropout_voltage": [
            ("What is the dropout of the {name}?", "The {name} has a dropout voltage of {val}."),
        ],
        "rds_on": [
            ("What is the Rds(on) of the {name}?", "The {name} Rds(on) is {val}."),
            ("What is the on-resistance of the {name}?", "The on-state resistance of the {name} is {val}."),
        ],
        "vds_max": [
            ("What is the max Vds of the {name}?", "The {name} maximum drain-source voltage is {val}."),
            ("What voltage can the {name} handle?", "The {name} is rated for Vds(max) of {val}."),
        ],
        "id_continuous": [
            ("What current can the {name} handle?", "The {name} continuous drain current is {val}."),
        ],
        "vgs_threshold": [
            ("What is the gate threshold of the {name}?", "The {name} Vgs(th) is {val}."),
            ("At what voltage does the {name} turn on?", "The {name} gate threshold voltage is {val}."),
        ],
        "gbw": [
            ("What is the {name} bandwidth?", "The {name} gain-bandwidth product (GBW) is {val}."),
        ],
        "i2c_address": [
            ("What I2C address does the {name} use?", "The {name} I2C address is {val}."),
            ("What address should I use for the {name}?", "Set the I2C address to {val} for the {name}."),
        ],
        "temperature_accuracy": [
            ("How accurate is the {name}?", "The {name} temperature accuracy is {val}."),
        ],
        "resolution": [
            ("What resolution does the {name} have?", "The {name} resolution is {val}."),
        ],
        "humidity_accuracy": [
            ("How accurate is the {name} humidity reading?", "The {name} humidity accuracy is {val}."),
        ],
        "switching_frequency": [
            ("What frequency does the {name} switch at?", "The {name} switching frequency is {val}."),
        ],
        "efficiency": [
            ("How efficient is the {name}?", "The {name} efficiency is {val}."),
        ],
        "channels": [
            ("How many channels does the {name} have?", "The {name} has {val} channel(s)."),
        ],
        "operating_temp": [
            ("What is the operating temperature range of the {name}?", "The {name} operates from {val}."),
        ],
        "package": [
            ("What package is the {name}?", "The {name} comes in {val} package."),
        ],
    }

    for comp in components:
        for spec_key, templates in templates_by_spec.items():
            if spec_key not in comp.specs:
                continue
            val = comp.specs[spec_key]

            for q_tmpl, a_tmpl in templates:
                q = q_tmpl.format(name=comp.name)

                # Handle special placeholders
                extra = {}
                if "{yesno33}" in a_tmpl:
                    can_33 = any(x in val for x in ["3.3", "3.0", "2.", "1."])
                    extra["yesno33"] = "Yes, 3.3V is within range." if can_33 else "Check if 3.3V falls within the specified range."
                if "{arm_yesno}" in a_tmpl:
                    is_arm = "arm" in val.lower() or "cortex" in val.lower()
                    extra["arm_yesno"] = "Yes, it is ARM-based." if is_arm else "No, it is not ARM-based."

                a = a_tmpl.format(name=comp.name, val=val, **extra)
                if comp.lcsc:
                    a += f" (LCSC: {comp.lcsc})"

                pairs.append(msg(q, a))

    return pairs


def generate_category_recommendation_qa(components: list[Component]) -> list[dict]:
    """Generate category-level recommendation Q&A (e.g., 'best MCU for low power')."""
    pairs = []

    recommendations = [
        # MCU recommendations
        {
            "q": "What is the cheapest ARM Cortex-M MCU available on LCSC?",
            "filter": lambda c: c.category == "mcu" and "cortex" in c.specs.get("core", "").lower(),
            "answer_prefix": "Among common ARM Cortex-M MCUs on LCSC, the cheapest options include",
        },
        {
            "q": "Which MCU has the most GPIO pins?",
            "filter": lambda c: c.category == "mcu" and c.specs.get("gpio_count"),
            "answer_prefix": "Comparing MCU GPIO counts from our database",
            "sort_key": lambda c: int(c.specs.get("gpio_count", "0")),
            "sort_reverse": True,
        },
        {
            "q": "Which MCU has the most flash memory?",
            "filter": lambda c: c.category == "mcu" and c.specs.get("flash"),
            "answer_prefix": "Comparing MCU flash sizes from our database",
            "sort_key": lambda c: int(c.specs.get("flash", "0 KB").split()[0]) * (1024 if "MB" in c.specs.get("flash", "") else 1),
            "sort_reverse": True,
        },
        {
            "q": "What are the best low-power MCU options?",
            "filter": lambda c: c.category == "mcu" and ("ultra-low" in c.description.lower() or "low-power" in c.description.lower() or "uA/MHz" in c.specs.get("supply_current", "")),
            "answer_prefix": "The best low-power MCU options from our database include",
        },
        {
            "q": "Which MCUs support Wi-Fi and Bluetooth?",
            "filter": lambda c: c.category == "mcu" and ("Wi-Fi" in " ".join(c.interfaces) or "wifi" in " ".join(c.interfaces).lower()) and ("BLE" in " ".join(c.interfaces) or "Bluetooth" in " ".join(c.interfaces)),
            "answer_prefix": "MCUs with both Wi-Fi and Bluetooth support",
        },
        {
            "q": "Which MCUs support CAN bus?",
            "filter": lambda c: c.category == "mcu" and ("CAN" in " ".join(c.interfaces) or "FDCAN" in " ".join(c.interfaces)),
            "answer_prefix": "MCUs with CAN bus support",
        },
        {
            "q": "Which MCUs have USB support?",
            "filter": lambda c: c.category == "mcu" and ("USB" in " ".join(c.interfaces) or "usb" in c.specs.get("usb", "")),
            "answer_prefix": "MCUs with USB support",
        },
        {
            "q": "Which MCUs are RISC-V based?",
            "filter": lambda c: c.category == "mcu" and "risc-v" in c.specs.get("core", "").lower(),
            "answer_prefix": "RISC-V MCUs in our database",
        },
        # Regulator recommendations
        {
            "q": "What are the best LDO regulators for battery-powered projects?",
            "filter": lambda c: c.category == "regulator" and "ldo" in c.description.lower() and c.specs.get("quiescent_current"),
            "answer_prefix": "For battery-powered projects, you want low quiescent current. The best LDOs include",
        },
        {
            "q": "What are the best step-down (buck) converters?",
            "filter": lambda c: c.category == "regulator" and ("step-down" in c.description.lower() or "buck" in c.description.lower()),
            "answer_prefix": "The best step-down converters from our database",
        },
        {
            "q": "Which regulators have the highest efficiency?",
            "filter": lambda c: c.category == "regulator" and c.specs.get("efficiency"),
            "answer_prefix": "The most efficient voltage regulators include",
        },
        # Sensor recommendations
        {
            "q": "What are the best I2C sensors for environmental monitoring?",
            "filter": lambda c: c.category == "sensor" and "I2C" in c.interfaces and ("temperature" in c.description.lower() or "humidity" in c.description.lower() or "pressure" in c.description.lower()),
            "answer_prefix": "The best I2C environmental sensors include",
        },
        {
            "q": "What are the most accurate temperature sensors available?",
            "filter": lambda c: c.category == "sensor" and c.specs.get("temperature_accuracy"),
            "answer_prefix": "The most accurate temperature sensors in our database",
        },
        {
            "q": "What current sensors are available for power monitoring?",
            "filter": lambda c: c.category == "sensor" and "current" in c.description.lower(),
            "answer_prefix": "Current sensors for power monitoring include",
        },
        # MOSFET recommendations
        {
            "q": "What are the best logic-level MOSFETs for 3.3V MCUs?",
            "filter": lambda c: c.category == "mosfet" and ("logic" in c.specs.get("type", "").lower() or float(c.specs.get("vgs_threshold", "99").split("V")[0].split(",")[0].split()[-1].replace("(", "").replace(")", "")) <= 2.0 if c.specs.get("vgs_threshold") else False),
            "answer_prefix": "Logic-level MOSFETs suitable for 3.3V MCU control include",
        },
        {
            "q": "What are the best P-channel MOSFETs for load switching?",
            "filter": lambda c: c.category == "mosfet" and "p-channel" in c.specs.get("type", "").lower(),
            "answer_prefix": "P-channel MOSFETs for load switching include",
        },
        # Op-amp recommendations
        {
            "q": "What are the best low-noise op-amps for audio?",
            "filter": lambda c: c.category == "opamp" and ("audio" in c.description.lower() or "low-noise" in c.description.lower() or c.specs.get("noise")),
            "answer_prefix": "The best low-noise op-amps for audio applications include",
        },
        {
            "q": "What are the best rail-to-rail op-amps?",
            "filter": lambda c: c.category == "opamp" and ("rail" in c.description.lower() or "rrio" in c.description.lower()),
            "answer_prefix": "Rail-to-rail op-amps from our database include",
        },
    ]

    for rec in recommendations:
        matching = [c for c in components if rec["filter"](c)]
        if not matching:
            continue

        if rec.get("sort_key"):
            try:
                matching.sort(key=rec["sort_key"], reverse=rec.get("sort_reverse", False))
            except (ValueError, TypeError):
                pass

        a = rec["answer_prefix"] + ":\n\n"
        for c in matching[:8]:  # Top 8
            a += f"- **{c.name}**: {c.description}"
            if c.lcsc:
                a += f" (LCSC: {c.lcsc})"
            a += "\n"

        pairs.append(msg(rec["q"], a.strip()))

    return pairs


def generate_alternative_qa(components: list[Component]) -> list[dict]:
    """Generate detailed cross-reference / alternative component Q&A."""
    pairs = []

    for comp in components:
        if not comp.alternatives:
            continue

        # Standard alternatives question
        q = f"What can I use instead of the {comp.name}?"
        a = f"Alternatives to the {comp.name} ({comp.description}):\n\n"
        for alt in comp.alternatives:
            a += f"- {alt}\n"
        a += f"\nWhen selecting an alternative, verify pin compatibility, electrical specifications, and package match for your specific application."
        if comp.lcsc:
            a += f"\n\nOriginal part LCSC: {comp.lcsc}"
        pairs.append(msg(q, a))

        # Pin-compatible question
        q2 = f"Is there a drop-in replacement for the {comp.name}?"
        a2 = f"For the {comp.name} ({comp.specs.get('package', comp.packages[0] if comp.packages else 'check datasheet')}), potential replacements include:\n\n"
        for alt in comp.alternatives:
            a2 += f"- {alt}\n"
        a2 += f"\n**Important**: Always verify pin-for-pin compatibility, especially for power, ground, and NC pins. Check the alternative's datasheet for any differences in:\n"
        a2 += "- Package footprint and pin assignment\n"
        a2 += "- Electrical specifications (voltage, current ratings)\n"
        a2 += "- Timing parameters\n"
        a2 += "- Enable/shutdown pin polarity"
        pairs.append(msg(q2, a2))

    return pairs


def generate_lcsc_sourcing_qa(components: list[Component]) -> list[dict]:
    """Generate LCSC/sourcing Q&A from component database."""
    pairs = []

    for comp in components:
        if not comp.lcsc:
            continue

        q = f"What is the LCSC part number for the {comp.name}?"
        a = f"The LCSC part number for the {comp.name} is **{comp.lcsc}**.\n\n"
        a += f"Description: {comp.description}\n"
        if comp.specs.get("package"):
            a += f"Package: {comp.specs['package']}\n"
        elif comp.packages:
            a += f"Package: {comp.packages[0]}\n"
        a += f"Manufacturer: {comp.manufacturer}\n"
        a += f"\nSearch on LCSC: https://www.lcsc.com/search?q={comp.lcsc}"
        pairs.append(msg(q, a))

        # Also generate a reverse lookup
        q2 = f"What component is LCSC part number {comp.lcsc}?"
        a2 = f"LCSC {comp.lcsc} is the **{comp.name}** — {comp.description}."
        a2 += f"\nManufacturer: {comp.manufacturer}"
        if comp.specs.get("package"):
            a2 += f"\nPackage: {comp.specs['package']}"
        pairs.append(msg(q2, a2))

    return pairs


def generate_comparison_detailed_qa(components: list[Component]) -> list[dict]:
    """Generate detailed head-to-head comparison Q&A pairs."""
    pairs = []

    by_category: dict[str, list[Component]] = {}
    for c in components:
        by_category.setdefault(c.category, []).append(c)

    comparison_templates = [
        "What is the difference between the {a} and the {b}?",
        "{a} vs {b} — which one should I choose?",
        "Should I use the {a} or the {b} for my project?",
    ]

    for cat, comps in by_category.items():
        if len(comps) < 2:
            continue
        for i, ca in enumerate(comps):
            for cb in comps[i+1:]:
                # Only generate for a subset to avoid explosion
                if hash(ca.name + cb.name) % 2 != 0:
                    continue

                tmpl = comparison_templates[hash(ca.name + cb.name) % len(comparison_templates)]
                q = tmpl.format(a=ca.name, b=cb.name)

                a = f"**{ca.name}** vs **{cb.name}**:\n\n"
                a += f"| Feature | {ca.name} | {cb.name} |\n"
                a += f"|---------|{'---' * len(ca.name)}|{'---' * len(cb.name)}|\n"

                # Compare common specs
                all_keys = set(ca.specs.keys()) | set(cb.specs.keys())
                important_keys = ["core", "max_frequency", "flash", "sram", "supply_voltage",
                                  "output_voltage", "max_output_current", "dropout_voltage",
                                  "rds_on", "vds_max", "id_continuous", "gbw", "channels",
                                  "resolution", "temperature_accuracy", "switching_frequency",
                                  "efficiency", "package"]
                shown = 0
                for key in important_keys:
                    if key in all_keys and shown < 8:
                        va = ca.specs.get(key, "N/A")
                        vb = cb.specs.get(key, "N/A")
                        if va != vb:  # Only show differences
                            a += f"| {key.replace('_', ' ').title()} | {va} | {vb} |\n"
                            shown += 1

                a += f"\n{ca.name}: {ca.description}\n{cb.name}: {cb.description}"
                pairs.append(msg(q, a))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CATEGORY_GENERATORS = {
    "specs": ("Component Specs Q&A", lambda comps, jitx: generate_spec_qa(comps)),
    "selection": ("Component Selection Q&A", lambda comps, jitx: generate_selection_qa()),
    "crossref": ("Cross-Reference Q&A", lambda comps, jitx: generate_crossref_qa()),
    "bom": ("BOM/Sourcing Q&A", lambda comps, jitx: generate_bom_qa()),
    "datasheet": ("Datasheet Reading Q&A", lambda comps, jitx: generate_datasheet_qa()),
    "jitx": ("JITX Component Q&A", lambda comps, jitx: generate_jitx_qa(jitx)),
    "comparison": ("Component Comparison Q&A", lambda comps, jitx: generate_comparison_qa(comps)),
    "pinout": ("Pinout Q&A", lambda comps, jitx: generate_pinout_qa(comps)),
    "application": ("Application Notes Q&A", lambda comps, jitx: generate_application_qa(comps)),
    "parametric": ("Parametric/Rephrased Q&A", lambda comps, jitx: generate_parametric_qa(comps)),
    "design_pattern": ("Design Pattern Q&A", lambda comps, jitx: generate_design_pattern_qa()),
    "troubleshooting": ("Troubleshooting Q&A", lambda comps, jitx: generate_troubleshooting_qa()),
    "passive_value": ("Passive Value Q&A", lambda comps, jitx: generate_passive_value_qa()),
    "pcb_design": ("PCB Design Q&A", lambda comps, jitx: generate_pcb_design_qa()),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate electronic components Q&A training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(CATEGORY_GENERATORS.keys()),
        help=f"Comma-separated categories to generate. Available: {', '.join(CATEGORY_GENERATORS.keys())}",
    )
    parser.add_argument(
        "--jitx-path",
        type=Path,
        default=Path("/tmp/jitx-odb"),
        help="Path to JITX open-components-database clone (default: /tmp/jitx-odb/)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics to stderr instead of generating data",
    )
    parser.add_argument(
        "--list-components",
        action="store_true",
        help="List all built-in components and exit",
    )
    args = parser.parse_args()

    if args.list_components:
        by_cat: dict[str, list[str]] = {}
        for c in COMPONENTS:
            by_cat.setdefault(c.category, []).append(c.name)
        for cat, names in sorted(by_cat.items()):
            print(f"\n=== {cat.upper()} ({len(names)}) ===", file=sys.stderr)
            for n in names:
                print(f"  {n}", file=sys.stderr)
        print(f"\nTotal: {len(COMPONENTS)} built-in components", file=sys.stderr)
        return

    # Parse categories
    requested = [c.strip() for c in args.categories.split(",")]
    for cat in requested:
        if cat not in CATEGORY_GENERATORS:
            logger.error("Unknown category: %s. Available: %s", cat, ", ".join(CATEGORY_GENERATORS.keys()))
            sys.exit(1)

    # Load JITX components
    jitx_components = load_jitx_components(args.jitx_path)

    # Generate
    total = 0
    stats: dict[str, int] = {}

    for cat in requested:
        label, gen_fn = CATEGORY_GENERATORS[cat]
        pairs = gen_fn(COMPONENTS, jitx_components)
        stats[label] = len(pairs)
        total += len(pairs)

        if not args.stats:
            for pair in pairs:
                print(json.dumps(pair, ensure_ascii=False))

    # Stats output
    logger.info("=== Generation Statistics ===")
    for label, count in stats.items():
        logger.info("  %-30s %5d pairs", label, count)
    logger.info("  %-30s %5d pairs", "TOTAL", total)
    logger.info("Built-in components: %d", len(COMPONENTS))
    logger.info("JITX components: %d (with pins)", len(jitx_components))


if __name__ == "__main__":
    main()
