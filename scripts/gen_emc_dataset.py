#!/usr/bin/env python3
"""Generate EMC/EMI training Q&A pairs.

Covers standards, shielding, filtering, PCB layout, conducted/radiated
emissions, ESD protection, pre-compliance testing, and cable shielding.

Output: JSONL to stdout.
"""
from __future__ import annotations

import json
import random
import sys

random.seed(44)

DOMAIN = "emc"


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


# ---------------------------------------------------------------------------
# Data banks
# ---------------------------------------------------------------------------

FREQUENCIES = ["100kHz", "1MHz", "10MHz", "30MHz", "100MHz", "200MHz", "300MHz", "500MHz", "1GHz", "3GHz", "6GHz"]
SHIELD_MATERIALS = ["aluminum", "copper", "mu-metal", "steel", "tin-plated steel", "nickel-silver", "conductive fabric"]
FERRITE_MATERIALS = ["MnZn (manganese-zinc)", "NiZn (nickel-zinc)"]
TVS_PARTS = ["SMBJ5.0A", "SMAJ15A", "PESD5V0S1BA", "TPD4E05U06", "SP0503BAHT", "PRTR5V0U2X"]
VARISTORS = ["ERZE14A471", "V14E250P", "MOV-10D471K"]
CMC_PARTS = ["WE-CNSW", "ACM2012", "DLW21SN", "ACT45B", "WE-SL5"]
FILTER_CAPS = ["100pF", "220pF", "470pF", "1nF", "2.2nF", "4.7nF", "10nF", "100nF"]
FERRITE_BEADS = ["BLM18PG121", "BLM18AG601", "BLM15HD102", "MMZ1608B601", "MPZ2012S601"]
PCB_LAYERS = ["2-layer", "4-layer", "6-layer", "8-layer"]
CLOCK_FREQS = ["8MHz", "16MHz", "25MHz", "48MHz", "50MHz", "100MHz", "133MHz", "200MHz"]
INTERFACES = ["USB 2.0", "USB 3.0", "HDMI", "Ethernet 100BASE-TX", "Ethernet 1000BASE-T", "SPI", "I2C", "CAN", "RS-485", "LVDS", "PCIe"]
CONNECTOR_TYPES = ["USB-C", "USB-A", "RJ45", "HDMI Type-A", "D-Sub 9", "SMA", "BNC", "screw terminal"]
PRODUCTS = ["industrial controller", "IoT gateway", "motor drive", "LED driver", "medical device", "automotive ECU", "consumer electronics", "telecom equipment"]
IEC_TESTS = [
    ("IEC 61000-4-2", "ESD immunity", "Electrostatic discharge"),
    ("IEC 61000-4-3", "Radiated immunity", "Radiated RF electromagnetic field"),
    ("IEC 61000-4-4", "EFT/Burst immunity", "Electrical fast transient/burst"),
    ("IEC 61000-4-5", "Surge immunity", "Surge (lightning)"),
    ("IEC 61000-4-6", "Conducted immunity", "Conducted RF disturbances"),
    ("IEC 61000-4-8", "Magnetic field immunity", "Power frequency magnetic field"),
    ("IEC 61000-4-11", "Voltage dip immunity", "Voltage dips/interruptions"),
]

EMISSION_STANDARDS = [
    ("CISPR 32", "Multimedia equipment emissions", "Class A (commercial) / Class B (residential)"),
    ("EN 55032", "European harmonized version of CISPR 32", "Radiated + conducted limits"),
    ("FCC Part 15", "US unintentional radiators", "Class A (commercial) / Class B (residential)"),
    ("CISPR 25", "Vehicle component emissions", "5 classes, limits from 150kHz to 2.5GHz"),
    ("MIL-STD-461G", "Military EMC requirements", "RE102, CE102 etc."),
    ("CISPR 11", "Industrial, scientific, medical (ISM) equipment", "Group 1/2, Class A/B"),
]

# ---------------------------------------------------------------------------
# Standards
# ---------------------------------------------------------------------------

def gen_standards():
    """EMC standards questions."""
    for std, desc, detail in EMISSION_STANDARDS:
        emit(
            f"What is {std} and what does it cover?",
            f"""{std}: {desc}

**Scope**: {detail}

{"**CISPR 32** (EN 55032 in Europe) defines emission limits for multimedia equipment including:" if std == "CISPR 32" else ""}
{"- **Conducted emissions**: 150 kHz to 30 MHz, measured on AC mains with a LISN" if std in ("CISPR 32", "EN 55032") else ""}
{"- **Radiated emissions**: 30 MHz to 6 GHz, measured at 3m or 10m distance" if std in ("CISPR 32", "EN 55032") else ""}
{"" if std not in ("CISPR 32", "EN 55032") else "- Class A: commercial/industrial (higher limits, easier to pass)"}
{"" if std not in ("CISPR 32", "EN 55032") else "- Class B: residential (stricter limits, ~10 dB lower than Class A)"}

{"**FCC Part 15** applies to all unintentional radiators sold in the US:" if std == "FCC Part 15" else ""}
{"- Subpart B: covers digital devices" if std == "FCC Part 15" else ""}
{"- Class A: commercial (less stringent)" if std == "FCC Part 15" else ""}
{"- Class B: residential (more stringent, ~6-10 dB lower limits)" if std == "FCC Part 15" else ""}
{"- Radiated: 30 MHz to 40 GHz" if std == "FCC Part 15" else ""}
{"- Conducted: 150 kHz to 30 MHz" if std == "FCC Part 15" else ""}

{"**MIL-STD-461G** is the US military EMC standard:" if std == "MIL-STD-461G" else ""}
{"- RE102: Radiated emissions 10 kHz to 18 GHz" if std == "MIL-STD-461G" else ""}
{"- CE102: Conducted emissions 10 kHz to 10 MHz" if std == "MIL-STD-461G" else ""}
{"- RS103: Radiated susceptibility" if std == "MIL-STD-461G" else ""}
{"- CS101/CS114/CS115/CS116: Conducted susceptibility" if std == "MIL-STD-461G" else ""}

Compliance is mandatory for market access in the respective region."""
        )

    for std, desc, test_type in IEC_TESTS:
        for product in random.sample(PRODUCTS, 3):
            emit(
                f"What are the {std} ({test_type}) requirements for a {product}?",
                f"""{std} — {desc} ({test_type})

**Test levels for {product}:**

{"| Level | Contact discharge | Air discharge |" if std == "IEC 61000-4-2" else "| Level | Test voltage |"}
{"| 1 | 2 kV | 2 kV |" if std == "IEC 61000-4-2" else "| 1 | 0.5 kV |" if std == "IEC 61000-4-5" else "| 1 | Low |"}
{"| 2 | 4 kV | 4 kV |" if std == "IEC 61000-4-2" else "| 2 | 1 kV |" if std == "IEC 61000-4-5" else "| 2 | Medium |"}
{"| 3 | 6 kV | 8 kV |" if std == "IEC 61000-4-2" else "| 3 | 2 kV |" if std == "IEC 61000-4-5" else "| 3 | High |"}
{"| 4 | 8 kV | 15 kV |" if std == "IEC 61000-4-2" else "| 4 | 4 kV |" if std == "IEC 61000-4-5" else "| 4 | Very high |"}

**For {product}**, typical requirement is {"Level 3 (6 kV contact, 8 kV air)" if std == "IEC 61000-4-2" else "Level 3"} with performance criterion {"B (temporary loss of function, self-recoverable)" if "industrial" in product or "motor" in product else "A (normal performance during and after test)"}.

**Performance criteria:**
- **A**: Normal performance within specification limits
- **B**: Temporary degradation, self-recoverable
- **C**: Temporary loss of function, operator intervention needed
- **D**: Loss of function not recoverable (not normally acceptable)

{"Key protection measures: TVS diodes on all external I/O, proper PCB grounding, shielded enclosure." if std == "IEC 61000-4-2" else "Ensure proper filtering and grounding for compliance."}"""
            )


def gen_shielding():
    """Shielding effectiveness questions."""
    for material in SHIELD_MATERIALS:
        for freq in random.sample(FREQUENCIES, 3):
            emit(
                f"What is the shielding effectiveness of {material} at {freq}?",
                f"""Shielding effectiveness (SE) of {material} at {freq}:

**Shielding effectiveness** is composed of three components:
SE (dB) = A (absorption loss) + R (reflection loss) + B (re-reflection correction)

For {material}:
{"- Excellent for electric fields (high conductivity → high reflection loss)" if material in ("copper", "aluminum") else "- Excellent for magnetic fields below 100 kHz (high permeability)" if material == "mu-metal" else "- Good general-purpose shielding"}
{"- At " + freq + ": SE depends on thickness, but typically 60-100 dB for 1mm sheet" if material in ("copper", "aluminum") else "- At " + freq + ": SE varies with permeability and frequency"}

**Absorption loss** (per skin depth):
- A = 8.686 * t / delta (dB)
- delta = 1/sqrt(pi*f*mu*sigma) (skin depth)
- {"delta ~ 0.066mm at 1MHz for copper" if material == "copper" else "delta ~ 0.084mm at 1MHz for aluminum" if material == "aluminum" else "delta varies with alloy composition"}

**Reflection loss**:
- R = 20*log10(Z_wave / (4*Z_shield)) (dB)
- Higher for electric fields (Z_wave high), lower for magnetic fields

**Design guidelines:**
- Use {material} sheet thickness >= 2 * skin depth at lowest frequency of concern
- Ensure continuous electrical contact at all seams
- {"Add EMI gaskets at enclosure joints" if material in ("aluminum", "steel") else ""}
- Apertures (ventilation, connectors) are the weakest points — keep apertures < lambda/20"""
            )

    # Aperture shielding
    for _ in range(15):
        aperture_mm = random.choice([1, 2, 3, 5, 10, 15, 20, 30, 50])
        freq = random.choice(FREQUENCIES)
        emit(
            f"How does a {aperture_mm}mm aperture affect shielding effectiveness at {freq}?",
            f"""A {aperture_mm}mm aperture degrades shielding at {freq}:

**Rule of thumb**: An aperture reduces SE when its largest dimension approaches lambda/2.

At {freq}:
- Wavelength lambda = c/f
- The aperture becomes significant when: aperture >= lambda/20

**SE reduction from aperture:**
SE_aperture = 20*log10(lambda / (2 * L_aperture)) dB

Where L_aperture = {aperture_mm}mm = {aperture_mm/1000}m.

**Mitigation strategies:**
1. **Waveguide-beyond-cutoff**: Extend aperture depth to >= 3x diameter (honeycomb vents)
2. **Multiple small apertures**: Replace one large hole with many small ones (SE improves by 20*log10(1/sqrt(n)) if spaced > lambda/2)
3. **Conductive mesh**: Wire mesh or perforated sheet (opening < lambda/20)
4. **EMI gaskets**: Conductive elastomer or finger stock at joints
5. **Conductive tape**: Copper or aluminum tape over slots

For {"a " + str(aperture_mm) + "mm aperture, this is critical above ~" + str(int(300000/(20*aperture_mm))) + " MHz" if aperture_mm <= 10 else "large apertures, shielding is compromised at lower frequencies — consider honeycomb panels"}."""
        )


def gen_filtering():
    """EMC filtering questions."""
    # Common-mode chokes
    for cmc in CMC_PARTS:
        for iface in random.sample(INTERFACES, 3):
            emit(
                f"How do I select a common-mode choke ({cmc}) for {iface} EMI filtering?",
                f"""Common-mode choke selection for {iface} using {cmc}:

**Selection criteria:**

1. **Impedance at noise frequency**: Choose impedance >= 100 ohm at the problematic frequency
   - {"USB 2.0: 90 ohm differential impedance, CMC impedance 90-120 ohm at 100 MHz" if iface == "USB 2.0" else "Select based on the noise frequency spectrum"}

2. **Rated current**: Must exceed maximum signal current
   - {"USB 2.0: >= 500 mA" if "USB" in iface else ">= rated interface current"}

3. **DC resistance**: Low DCR to minimize signal attenuation
   - Typical: < 0.5 ohm per winding

4. **Insertion loss**: Check differential-mode insertion loss at signal frequency
   - Must not attenuate the wanted signal: {"< 1 dB at 480 MHz for USB 2.0" if "USB 2.0" in iface else "< 1 dB at signal frequency"}

5. **Impedance curve**:
   - MnZn ferrite: High impedance at 1-30 MHz (good for conducted emissions)
   - NiZn ferrite: High impedance at 30-500 MHz (good for radiated)

**Circuit placement:**
```
Signal source ──── CMC ──── Connector/Cable
                 ┌─┤├─┐
                 │     │
                GND   GND
                 │     │
                 └─────┘
         (optional Y-caps to chassis GND)
```

Add Y-capacitors ({random.choice(FILTER_CAPS)}) from each line to chassis ground after the CMC for best CM rejection."""
            )

    # Ferrite beads
    for fb in FERRITE_BEADS:
        emit(
            f"How do I use ferrite bead {fb} for EMI filtering on a power rail?",
            f"""Ferrite bead {fb} for power rail filtering:

**Ferrite bead characteristics:**
- Acts as a frequency-dependent resistor (NOT an inductor at high frequency)
- Below resonance: mostly inductive (L)
- At resonance: maximum impedance (R)
- Above resonance: capacitive (C due to parasitic)

**Power rail filtering circuit:**
```
VIN ──── FB ──── VOUT
         {fb}
              ├── C1 (bulk, 10-100uF)
              ├── C2 (ceramic, 100nF)
              └── C3 (ceramic, 1nF for HF)
```

**Selection criteria for {fb}:**
1. **Impedance at noise frequency**: Target >= 100 ohm at problematic frequency
2. **DC resistance**: Low DCR to minimize voltage drop (< 0.5 ohm typical)
3. **Rated current**: Must handle max load current without saturation
4. **DC bias**: Impedance drops with DC current — check impedance vs. bias curves

**Common mistake**: Using a ferrite bead without sufficient output capacitance. The bead needs a low-impedance load (capacitor) to be effective.

**Do NOT use ferrite beads on:**
- High-speed digital signal lines (they distort edges)
- Power rails with fast transient loads (can cause voltage spikes)
- Analog reference voltages (noise modulation)"""
        )

    # Pi filters
    for _ in range(20):
        c1 = random.choice(FILTER_CAPS)
        c2 = random.choice(FILTER_CAPS)
        fb = random.choice(FERRITE_BEADS)
        product = random.choice(PRODUCTS)
        emit(
            f"Design a pi-filter for EMI suppression on the power input of a {product}.",
            f"""Pi-filter design for {product} power input:

```
AC Mains ──┬── L/FB ──┬── To DC/DC or LDO
           │          │
          C1         C2
         {c1}       {c2}
           │          │
          GND        GND
```

**Component selection:**

1. **C1 (input side)**: {c1} X2/Y2 safety-rated capacitor
   - X-cap (line-to-line): Suppresses differential-mode noise
   - Y-cap (line-to-ground): Suppresses common-mode noise
   - Must be safety-rated (X2 for line-line, Y2 for line-ground)

2. **L/FB (series element)**: {fb} or inductor
   - Common-mode choke for CM noise
   - Differential-mode inductor for DM noise
   - Impedance > 100 ohm at noise frequency

3. **C2 (output side)**: {c2}
   - Provides low impedance for filtered supply
   - Decouples downstream switching noise

**Safety requirements:**
- X-capacitors: Self-healing film type, max 100nF without bleeder resistor
- Y-capacitors: Max leakage current per IEC 60950-1:
  - Class I equipment: 3.5 mA
  - Class II (double insulated): 0.25 mA
  - Medical (BF applied part): 0.1 mA

**Attenuation**: A pi-filter provides ~40 dB/decade above cutoff, vs ~20 dB/decade for a simple L-C filter."""
        )


def gen_pcb_layout():
    """PCB layout for EMC questions."""
    for layers in PCB_LAYERS:
        emit(
            f"What is the recommended {layers} PCB stackup for EMC compliance?",
            f"""Recommended {layers} PCB stackup for EMC:

{"**2-layer stackup:**" if layers == "2-layer" else ""}
{"```" if layers == "2-layer" else ""}
{"Layer 1 (Top):    Signal + Power" if layers == "2-layer" else ""}
{"Layer 2 (Bottom): Ground plane (as continuous as possible)" if layers == "2-layer" else ""}
{"```" if layers == "2-layer" else ""}
{"EMC tips for 2-layer:" if layers == "2-layer" else ""}
{"- Keep ground plane as continuous as possible (>80% fill)" if layers == "2-layer" else ""}
{"- Route all signals over unbroken ground" if layers == "2-layer" else ""}
{"- Use ground pour on top layer around signals" if layers == "2-layer" else ""}

{"**4-layer stackup (recommended for most designs):**" if layers == "4-layer" else ""}
{"```" if layers == "4-layer" else ""}
{"Layer 1 (Top):    Signal (components)" if layers == "4-layer" else ""}
{"Layer 2:          Ground plane (solid, no splits)" if layers == "4-layer" else ""}
{"Layer 3:          Power plane" if layers == "4-layer" else ""}
{"Layer 4 (Bottom): Signal" if layers == "4-layer" else ""}
{"```" if layers == "4-layer" else ""}
{"EMC tips for 4-layer:" if layers == "4-layer" else ""}
{"- Signal layers adjacent to ground plane = good return path" if layers == "4-layer" else ""}
{"- Thin dielectric between L1-L2 (tight coupling)" if layers == "4-layer" else ""}
{"- Ground and power planes form a distributed capacitor" if layers == "4-layer" else ""}

{"**6-layer stackup:**" if layers == "6-layer" else ""}
{"```" if layers == "6-layer" else ""}
{"Layer 1: Signal" if layers == "6-layer" else ""}
{"Layer 2: Ground (reference for L1)" if layers == "6-layer" else ""}
{"Layer 3: Signal (inner)" if layers == "6-layer" else ""}
{"Layer 4: Power" if layers == "6-layer" else ""}
{"Layer 5: Ground (reference for L6)" if layers == "6-layer" else ""}
{"Layer 6: Signal" if layers == "6-layer" else ""}
{"```" if layers == "6-layer" else ""}

{"**8-layer stackup:**" if layers == "8-layer" else ""}
{"```" if layers == "8-layer" else ""}
{"Layer 1: Signal" if layers == "8-layer" else ""}
{"Layer 2: Ground" if layers == "8-layer" else ""}
{"Layer 3: Signal" if layers == "8-layer" else ""}
{"Layer 4: Power" if layers == "8-layer" else ""}
{"Layer 5: Ground" if layers == "8-layer" else ""}
{"Layer 6: Signal" if layers == "8-layer" else ""}
{"Layer 7: Ground" if layers == "8-layer" else ""}
{"Layer 8: Signal" if layers == "8-layer" else ""}
{"```" if layers == "8-layer" else ""}

**Universal EMC stackup rules:**
1. Every signal layer must be adjacent to a solid reference plane
2. Never route high-speed signals over splits in reference planes
3. Keep signal-to-reference-plane spacing as thin as possible (3-5 mil)
4. Use via stitching around board edges (lambda/20 spacing)"""
        )

    # Ground plane and return path
    topics = [
        {
            "q": "Why is a solid ground plane critical for EMC, and what happens when you split it?",
            "a": """A solid ground plane is the single most important PCB feature for EMC:

**Why ground planes matter:**
1. **Low-impedance return path**: Current returns directly under the signal trace (path of least inductance)
2. **Shielding**: Acts as a Faraday cage for internal layers
3. **Controlled impedance**: Reference for transmission line behavior
4. **Reduced loop area**: Signal + return form a minimal loop (less radiation)

**What happens with a split ground plane:**
- Return current is forced to detour around the split
- Creates a large current loop = excellent antenna
- Loop radiates proportional to: E ~ f^2 * A * I (area * current * frequency squared)
- A 1cm detour at 100 MHz can cause 10-20 dB increase in emissions

**Common mistakes that split the ground:**
1. Routing traces across the split (creates a slot antenna)
2. Placing analog/digital split under mixed-signal ICs
3. Large via anti-pads creating gaps
4. Power plane cutouts under high-speed signals

**Solutions:**
- Use a single, unbroken ground plane
- If split is necessary (mixed-signal), bridge with a single connection under the ADC
- Stitch planes with vias at lambda/20 spacing
- Route sensitive signals away from plane edges"""
        },
        {
            "q": "How do I use via stitching for EMC on a PCB?",
            "a": """Via stitching connects ground planes on multiple layers and forms a virtual shield:

**Placement rules:**
1. **Board perimeter**: Ring of vias around entire board edge
   - Spacing: lambda/20 at highest frequency of concern
   - At 1 GHz: lambda = 300mm, spacing <= 15mm
   - At 3 GHz: lambda = 100mm, spacing <= 5mm

2. **Around sensitive areas**: Create a via fence around:
   - Clock oscillators and PLL circuits
   - Switching power supplies
   - RF sections
   - High-speed I/O connectors

3. **Between ground planes**: Connect all ground layers together
   - Via spacing: 2-5mm typical
   - Use at least 4 vias per ground-to-ground connection

**Via fence design:**
```
Via  Via  Via  Via  Via  Via
 |    |    |    |    |    |
==================== Ground plane 1
==================== Ground plane 2
 |    |    |    |    |    |
Via  Via  Via  Via  Via  Via
```

**Effectiveness:**
- A via fence with lambda/20 spacing provides 20-40 dB isolation
- Smaller spacing = better isolation but more vias (cost)
- Via diameter: 0.3-0.5mm typical, smaller for denser designs

**Do NOT:**
- Leave large areas without stitching vias
- Place stitching vias through signal routing channels
- Forget stitching near connectors and cable entry points"""
        },
        {
            "q": "How should I route high-speed clock signals for EMC on a PCB?",
            "a": """Clock routing is critical because clocks are periodic signals with strong harmonics:

**Fundamental rules:**

1. **Shortest possible trace length**:
   - Place clock source close to the load
   - Every mm of trace is an antenna

2. **Impedance-controlled routing**:
   - Match trace impedance to source/load
   - Typical: 50 ohm single-ended, 100 ohm differential
   - Use manufacturer stackup calculator for correct trace width

3. **Route on inner layers**:
   - Between two ground planes = shielded stripline
   - 20-30 dB less radiation than microstrip (outer layer)

4. **Series termination**:
   ```
   Clock IC ── R(33-47 ohm) ── Trace ── Load
   ```
   - Reduces ringing and high-frequency harmonics
   - R value: Z0 - Rout (driver output impedance)

5. **Guard traces** (for very sensitive clocks):
   ```
   GND trace ─── via ─── via ─── via
   Clock trace
   GND trace ─── via ─── via ─── via
   ```

6. **Spread-spectrum clocking (SSC)**:
   - Reduces peak emissions by 6-10 dB
   - Spreads clock energy over a bandwidth (0.5-1% modulation)
   - Most modern MCUs support SSC on USB, PCIe clocks

7. **No stubs**: Clock traces must not have branches (T-junctions create reflections)

8. **Decoupling**: 100nF + 10nF at clock IC power pins, placed within 3mm"""
        },
    ]
    for t in topics:
        emit(t["q"], t["a"])

    # Decoupling capacitor placement
    for clock_freq in CLOCK_FREQS:
        for layers in random.sample(PCB_LAYERS, 2):
            emit(
                f"How should I place decoupling capacitors for a {clock_freq} MCU on a {layers} PCB?",
                f"""Decoupling capacitor placement for {clock_freq} MCU on {layers} board:

**Capacitor values** (use multiple values for broadband filtering):
```
100uF (bulk)    → Power entry, within 20mm of IC
10uF (mid)      → Near VDD pin group, within 10mm
100nF (bypass)  → One per VDD pin, within 3mm
10nF (HF)       → Critical VDD pins (PLL, USB), within 2mm
{"1nF (VHF)     → For clocks above 100 MHz, within 1mm" if int(clock_freq.replace("MHz","")) >= 100 else ""}
```

**Placement priority:**
1. VCAP/PLL power pin (most critical)
2. Core VDD pins
3. I/O VDD pins
4. Analog VDD (separate from digital)

**Via connection** ({"critical for " + layers if layers == "2-layer" else "optimized for " + layers}):
{"- 2-layer: Use wide traces (not vias) to connect cap pad directly to ground pour" if layers == "2-layer" else "- 4/6/8-layer: Use at least 2 vias per cap pad to ground plane"}
- Via-in-pad (filled and plated) is ideal for 0402 caps
- Place cap BETWEEN IC pin and via to ground (not beyond the via)

**Layout (cross-section):**
```
IC Pin ── Cap ── Via to GND plane
  Not:
IC Pin ── Via ── Cap ── Via to GND plane  (BAD: via inductance before cap)
```

**Self-resonant frequency**: Each cap value is effective near its SRF:
- 100nF / 0402: SRF ~ 100 MHz
- 10nF / 0402: SRF ~ 300 MHz
- 1nF / 0402: SRF ~ 1 GHz

For a {clock_freq} clock, the 3rd and 5th harmonics ({int(clock_freq.replace("MHz",""))*3} MHz, {int(clock_freq.replace("MHz",""))*5} MHz) are often the problematic frequencies."""
            )


def gen_conducted_radiated():
    """Conducted vs radiated emissions."""
    topics = [
        {
            "q": "What is the difference between conducted and radiated emissions in EMC?",
            "a": """Conducted and radiated emissions are the two main categories of unintentional electromagnetic emissions:

**Conducted emissions** (150 kHz - 30 MHz):
- Noise currents that flow on power cables, signal cables, and ground connections
- Measured using a Line Impedance Stabilization Network (LISN)
- Two modes:
  - **Differential mode (DM)**: Current flows on L and returns on N (loop)
  - **Common mode (CM)**: Current flows on both L and N, returns via ground
- Filtered with: EMI filters, X-caps (DM), Y-caps (CM), common-mode chokes

**Radiated emissions** (30 MHz - 6 GHz):
- Electromagnetic fields radiated from the product, cables, and PCB
- Measured with antennas in an anechoic chamber or OATS
- Sources:
  - PCB traces acting as antennas
  - Cables as antennas (most common source)
  - Enclosure slots and apertures
- Controlled with: shielding, cable filtering, PCB layout, spread-spectrum clocking

**Relationship:**
- Below ~30 MHz: Mostly conducted (cables too short to radiate efficiently)
- 30-200 MHz: Transition zone (both conducted and radiated)
- Above ~200 MHz: Mostly radiated (PCB traces and apertures)

**Key insight**: The #1 source of radiated emissions is usually the CABLES, not the PCB itself. Cables act as antennas for common-mode currents. Filtering at cable entry points is critical."""
        },
        {
            "q": "How do I measure conducted emissions with a LISN?",
            "a": """LISN (Line Impedance Stabilization Network) measurement setup:

**What a LISN does:**
1. Provides a standardized 50 ohm/50 uH impedance to the device under test (DUT)
2. Isolates the DUT from mains noise
3. Couples RF noise from the DUT to the spectrum analyzer

**Test setup:**
```
AC Mains → LISN → DUT
              ↓
         RF output → Spectrum Analyzer / EMI Receiver
```

**LISN standards:**
- CISPR 16-1-2 (50 ohm/50 uH, V-network)
- MIL-STD-461G (10 uF LISN for CE102)

**Measurement procedure:**
1. Connect LISN between mains and DUT
2. Connect LISN RF port to spectrum analyzer (50 ohm cable)
3. Measure Line (L) and Neutral (N) separately
4. DUT must be operating in worst-case emissions mode
5. Ground plane: 2m x 2m minimum, DUT 40cm above

**Frequency range:** 150 kHz to 30 MHz (CISPR), 10 kHz to 10 MHz (MIL-STD)

**Limits (CISPR 32 Class B):**
| Frequency | Quasi-peak (dBuV) | Average (dBuV) |
|-----------|-------------------|----------------|
| 0.15-0.5 MHz | 66-56 | 56-46 |
| 0.5-5 MHz | 56 | 46 |
| 5-30 MHz | 60 | 50 |

**Tips:**
- Use quasi-peak AND average detectors (both must pass)
- Test both L and N — one may be worse
- Peak detector for pre-scan (fast), then QP for compliance"""
        },
    ]
    for t in topics:
        emit(t["q"], t["a"])

    # Specific emission problems
    for product in PRODUCTS:
        for freq in random.sample(FREQUENCIES[:6], 2):
            emit(
                f"My {product} fails conducted emissions at {freq}. What are common causes and fixes?",
                f"""Conducted emissions failure at {freq} for {product}:

**Common causes at {freq}:**
{"- Switching power supply fundamental or harmonics (buck/boost at 100-500 kHz)" if "kHz" in freq or freq == "1MHz" else "- High-frequency switching transients or clock harmonics"}
{"- Differential-mode noise dominant below 1 MHz" if "kHz" in freq else "- Common-mode noise typically dominant above 1 MHz"}
- Poor input filter design or missing filter components
- Ground loop between DUT and measurement setup

**Diagnostic steps:**
1. Identify DM vs CM noise:
   - Measure L and N separately
   - DM: L and N are equal and opposite in phase
   - CM: L and N are equal and in phase
   - Or use a CM/DM separator network

2. Correlate with switching frequency:
   - Emissions at fsw and harmonics → DM from power stage
   - Broadband noise → CM from fast edges

**Fixes for differential-mode noise:**
- Add/increase X-capacitor (across L-N): {random.choice(["100nF", "220nF", "470nF"])} X2-rated
- Add series inductor (DM choke): 1-10 mH
- Slow down switching edges (gate resistor)
- Use spread-spectrum modulation on PWM

**Fixes for common-mode noise:**
- Add Y-capacitors (L-to-GND and N-to-GND): {random.choice(["1nF", "2.2nF", "4.7nF"])} Y2-rated
- Add common-mode choke before Y-caps
- Improve PCB grounding (wider traces, more vias)
- Add CM ferrite on cable

**Filter topology:**
```
L ──┬── CMC ──┬── DM inductor ──┬── To DUT
    X-cap     Y-cap              C-bulk
N ──┘── CMC ──┘──────────────── ┘
```"""
            )


def gen_esd_protection():
    """ESD protection design questions."""
    for tvs in TVS_PARTS:
        for iface in random.sample(INTERFACES, 2):
            emit(
                f"How do I protect {iface} against ESD using {tvs}?",
                f"""ESD protection for {iface} using {tvs}:

**Placement:**
```
Connector ──── {tvs} ──── Trace to IC
               │
              GND (chassis or signal ground)
```

**Critical design rules:**

1. **Place TVS as close to connector as possible** (within 5mm)
   - ESD current must not flow through sensitive traces

2. **Route ESD current path away from signal:**
   ```
   Connector pin → TVS → Chassis GND (shortest path)
   NOT through: Signal trace → IC → Ground → TVS
   ```

3. **Low-impedance ground connection:**
   - Multiple vias to ground plane
   - Wide trace (>= 0.5mm) to ground
   - {"Connect to chassis ground for IEC 61000-4-2 compliance" if "USB" in iface or "HDMI" in iface else "Connect to signal ground plane"}

4. **Series resistance** (optional, for signal conditioning):
   ```
   Connector ── R(33ohm) ── TVS ── IC
   ```
   Limits peak current into IC during ESD event

**{tvs} specifications:**
{"- Bidirectional TVS, low capacitance (<0.5pF for high-speed)" if "PESD" in tvs or "TPD" in tvs else "- Unidirectional TVS, higher clamping capability"}
{"- Multi-channel (2 or 4 lines in one package)" if "TPD" in tvs else "- Single channel"}

**{iface} specific considerations:**
{"- USB 2.0: Max 10pF capacitance on data lines" if iface == "USB 2.0" else ""}
{"- USB 3.0: Max 0.5pF capacitance on SuperSpeed lanes" if iface == "USB 3.0" else ""}
{"- HDMI: Use multi-channel TVS array for all data lanes" if iface == "HDMI" else ""}
{"- Ethernet: Use integrated magnetics with ESD protection or external TVS" if "Ethernet" in iface else ""}
{"- CAN: TVS rated for bus voltage (5V typical)" if iface == "CAN" else ""}

**IEC 61000-4-2 compliance targets:**
- Contact discharge: >= 6 kV (Level 3)
- Air discharge: >= 8 kV (Level 3)
- {tvs} clamp voltage must be below IC absolute maximum rating"""
            )


def gen_precompliance():
    """Pre-compliance testing questions."""
    topics = [
        {
            "q": "What equipment do I need for EMC pre-compliance testing?",
            "a": """Essential EMC pre-compliance test equipment:

**Minimum setup (~$2,000-5,000):**
1. **Spectrum analyzer with tracking generator**: $1,000-3,000
   - Range: 9 kHz to 3 GHz minimum
   - RBW: 9 kHz (quasi-peak), 200 Hz (CISPR Band A)
   - Examples: Siglent SSA3021X Plus, Rigol DSA815-TG

2. **Near-field probe set**: $200-500
   - H-field probes (loop): 3 sizes for different frequencies
   - E-field probe (monopole): for electric field
   - Example: Langer EMV-Technik RF-R set, Beehive 100C

3. **LISN (50 ohm/50 uH)**: $500-1,500
   - For conducted emissions measurement
   - Example: Tekbox TBLC08, Com-Power LI-150A

**Enhanced setup (~$5,000-15,000):**
4. **EMC pre-compliance software**: $500-2,000
   - Limit lines, quasi-peak detector emulation
   - Example: Gauss Instruments TDEMI, Tekbox EMCview

5. **Current probe**: $300-800
   - Clamp-on RF current probe for cable emissions
   - Example: Fischer F-33-1, Tekbox TBCP1

6. **ESD simulator**: $1,000-3,000
   - IEC 61000-4-2 compliant
   - Example: NoiseKen ESS-2000, EM TEST Dito

7. **Biconical + Log-periodic antennas**: $1,000-3,000
   - For semi-anechoic radiated measurements
   - 30 MHz - 1 GHz (biconical), 200 MHz - 6 GHz (LPDA)

**Test environment:**
- Conducted: Ground plane (copper sheet, 1m x 1m minimum)
- Radiated: Open area or semi-shielded room (not required for pre-compliance)
- Near-field: Any bench, probe close to PCB/cables"""
        },
        {
            "q": "How do I use near-field probes for EMC debugging?",
            "a": """Near-field probes identify EMI sources directly on the PCB:

**H-field (magnetic) probes:**
- Detect current flow (loops, traces, ICs)
- Shielded loop design rejects electric fields
- Three sizes for different resolution:
  - Large (25mm): Survey, find general area
  - Medium (10mm): Narrow down to component group
  - Small (3mm): Pinpoint individual traces/pins

**E-field (electric) probes:**
- Detect voltage (high-impedance nodes, heatsinks)
- Monopole or short dipole design
- Useful for finding radiating structures

**Measurement procedure:**
1. Connect probe to spectrum analyzer (50 ohm input)
2. Set span to cover frequency range of interest
3. Set RBW = 100 kHz for fast scanning
4. Move probe slowly over the PCB surface (~1-3mm height)
5. Note locations of peak emissions

**Interpreting results:**
- **Strong H-field near a trace**: High current flow, check return path
- **Strong H-field at IC**: Clock or switching noise source
- **Strong E-field at connector**: Cable is acting as antenna
- **Strong E-field at heatsink**: Heatsink radiating (add bypass cap to mounting)

**Tips:**
- Compare with limit line: probe factor + cable loss + antenna factor
- Pre-compliance correlation: near-field is ~20-30 dB above far-field
- Document positions with photos for before/after comparison
- Test at maximum clock speed / maximum load for worst case"""
        },
        {
            "q": "How do I perform a pre-compliance radiated emissions test?",
            "a": """Pre-compliance radiated emissions measurement:

**Setup:**
```
DUT (on table, 80cm height)
    ↕ 3m distance
Receiving antenna → Cable → Spectrum Analyzer
(on tripod, 1-4m height sweep)
```

**Procedure:**
1. **DUT configuration**: Maximum clock speed, all interfaces active, worst-case mode
2. **Antenna height**: Scan from 1m to 4m (find maximum)
3. **DUT rotation**: 0, 90, 180, 270 degrees (find maximum)
4. **Antenna polarization**: Both horizontal and vertical
5. **Frequency range**: 30 MHz to 1 GHz (biconical + LPDA)

**Spectrum analyzer settings:**
- RBW: 120 kHz (CISPR Band C/D)
- Detector: Peak for pre-scan, then quasi-peak for compliance
- Video bandwidth: >= 3x RBW
- Sweep time: Auto or manual (ensure sufficient dwell time)

**Distance correction:**
Measurements at 3m need correction for 10m limits:
- E_10m = E_3m - 20*log10(10/3) = E_3m - 10.5 dB

**CISPR 32 Class B limits (at 10m):**
| Frequency | QP Limit (dBuV/m) |
|-----------|--------------------|
| 30-230 MHz | 30 |
| 230-1000 MHz | 37 |

**Margin**: Aim for 6 dB margin below the limit for confidence.

**Common findings:**
- Emissions at clock harmonics → better decoupling, spread spectrum
- Broadband hump → switching noise, cable common-mode current
- Specific frequency peaks → identify source with near-field probe"""
        },
    ]
    for t in topics:
        emit(t["q"], t["a"])


def gen_cable_shielding():
    """Cable shielding questions."""
    cable_types = [
        ("braid shield", "85-95% coverage typical, good flexibility", "Use for most applications, 360-degree backshell termination"),
        ("foil shield", "100% coverage, thin and lightweight", "Good for static installations, must have drain wire"),
        ("braid + foil (double shield)", "95%+ coverage, best performance", "Required for USB 3.0, HDMI, and high-speed digital"),
        ("spiral shield", "60-80% coverage, very flexible", "Only for audio and low-frequency applications"),
        ("corrugated tube", "100% coverage, rigid", "EMI-tight conduit for cable bundles"),
    ]

    for shield_type, coverage, usage in cable_types:
        emit(
            f"When should I use {shield_type} cable shielding for EMC?",
            f"""{shield_type.title()} cable shielding:

**Coverage**: {coverage}
**Best for**: {usage}

**Shielding effectiveness:**
{"- Braid: 40-90 dB depending on coverage and frequency" if "braid" in shield_type and "foil" not in shield_type else ""}
{"- Foil: 50-100+ dB at high frequencies (solid barrier)" if "foil" in shield_type and "braid" not in shield_type else ""}
{"- Braid + Foil: 60-100+ dB across full spectrum" if "double" in shield_type else ""}
{"- Spiral: 20-40 dB (gaps between turns)" if "spiral" in shield_type else ""}
{"- Corrugated tube: 60-80 dB (continuous conductor)" if "corrugated" in shield_type else ""}

**Transfer impedance (Zt):**
- Key metric: lower Zt = better shielding
- {"Braid: ~10-50 mOhm/m below 10 MHz, rises above 100 MHz (due to braid apertures)" if "braid" in shield_type else "Foil: very low Zt at high frequencies" if "foil" in shield_type else "Zt depends on construction quality"}

**Termination is critical:**
- 360-degree termination (backshell) provides 20-40 dB more SE than pigtail
- **Never use pigtail connections** for EMC-critical cables
- Pigtail > 25mm = almost useless above 100 MHz
- Connector shell must make circumferential contact with shield

**Grounding:**
- Signal cables: Shield grounded at both ends (for RF)
- Low-frequency analog: Shield grounded at one end (to avoid ground loops)
- Mixed: Ground both ends with capacitor at one end (blocks DC loops, passes RF)"""
        )

    # Specific cable recommendations per interface
    for iface in INTERFACES:
        emit(
            f"What cable shielding is required for {iface} to pass EMC testing?",
            f"""Cable shielding requirements for {iface}:

{"**USB 2.0**: Braid + foil (double shield), 28AWG data pair with drain wire, 90 ohm impedance. Shield grounded at both ends through connector shell." if iface == "USB 2.0" else ""}
{"**USB 3.0**: Double-shielded with individual pair shielding for SuperSpeed lanes. Braid (85%+) over foil. Critical: 360-degree backshell termination." if iface == "USB 3.0" else ""}
{"**HDMI**: Triple-shielded (inner foil + braid + outer foil). Individual foil on each TMDS pair. 100 ohm differential impedance." if iface == "HDMI" else ""}
{"**Ethernet 100BASE-TX**: Unshielded (UTP Cat5e) acceptable for Class B. For industrial: STP/FTP with shielded RJ45 connectors." if "100BASE" in iface else ""}
{"**Ethernet 1000BASE-T**: Cat5e UTP minimum. For EMC: Cat6 STP with shielded connectors, drain wire connected to chassis ground." if "1000BASE" in iface else ""}
{"**SPI**: Not typically cabled externally. If cabled: use shielded ribbon cable with ground wires between signal wires. Keep cable < 30cm." if iface == "SPI" else ""}
{"**I2C**: Not designed for cabling. If external: shielded cable, max 1m, add bus buffer/repeater. Pull-ups on both ends." if iface == "I2C" else ""}
{"**CAN**: Shielded twisted pair (STP) for industrial/automotive. Shield grounded at one end (or both ends through capacitor). 120 ohm termination at each end." if iface == "CAN" else ""}
{"**RS-485**: Shielded twisted pair, shield grounded at one end. For long runs (>100m): use foil+braid. Termination 120 ohm at each end." if iface == "RS-485" else ""}
{"**LVDS**: Tightly coupled differential pair, shielded cable. 100 ohm impedance. Keep pair skew < 10ps. Shield grounded at both ends." if iface == "LVDS" else ""}
{"**PCIe**: Internal only (board-to-board). External cabling (OCuLink): double-shielded, individually shielded pairs, 85 ohm impedance." if iface == "PCIe" else ""}

**Common to all:** The cable shield must terminate with 360-degree contact at the connector — pigtail termination degrades shielding by 20-40 dB above 30 MHz."""
        )


def gen_emc_design_rules():
    """General EMC design rules and best practices."""
    rules = [
        {
            "q": "What are the top 10 PCB design rules for EMC compliance?",
            "a": """Top 10 PCB EMC design rules:

**1. Unbroken ground plane**
Never route signals across ground plane splits. Every signal needs a low-impedance return path directly underneath.

**2. Decoupling capacitors close to IC pins**
100nF within 3mm of each VDD pin. Use via-in-pad for 0402 caps. Multiple values (100nF + 10nF + 1nF) for broadband filtering.

**3. Short, fat power traces**
Power distribution: wide traces or planes. Every mOhm of power trace impedance creates voltage noise.

**4. Minimize loop areas**
Signal + return path form the radiating loop. Tight coupling to reference plane minimizes loop area.

**5. Route clocks on inner layers**
Stripline routing between ground planes provides shielding. Use series termination resistors.

**6. 3W rule for trace spacing**
Space traces >= 3x trace width apart to reduce crosstalk. Critical for: clock, reset, analog signals.

**7. Guard rings around oscillators**
Via-stitched ground ring around clock crystals and oscillators. Fill with ground copper.

**8. Filter at connectors**
All I/O lines filtered at the connector: TVS for ESD, ferrite bead or CMC for EMI. Filter before the signal enters the board.

**9. No right-angle traces**
Use 45-degree bends or arcs. Right angles cause impedance discontinuities and reflections at high frequency.

**10. Component placement**
- Group by function (digital, analog, power)
- Place noisy components (switchers, clocks) away from sensitive (ADC, radio)
- I/O connectors on one edge of the board
- Power entry near I/O connectors (filter noise before it spreads)"""
        },
        {
            "q": "How do I design a PCB for both conducted and radiated emissions compliance?",
            "a": """Comprehensive PCB EMC design approach:

**Conducted emissions (150 kHz - 30 MHz):**

1. **Power supply filtering:**
   - Pi-filter at power input (C-L-C)
   - X-cap: 100nF-1uF across L-N (differential mode)
   - Y-caps: 1nF-4.7nF to chassis ground (common mode)
   - Common-mode choke: >1 mH for 150 kHz-1 MHz

2. **Switching power supply layout:**
   - Minimize hot loop area (switch node)
   - Input capacitor close to FET and diode
   - Snubber across switch (RC: 10 ohm + 1nF typical)
   - Boot strap layout: short traces
   - Keep feedback trace away from inductor

3. **Ground strategy:**
   - Single-point connection between power ground and signal ground
   - Star ground for analog sections

**Radiated emissions (30 MHz - 6 GHz):**

1. **PCB stackup:**
   - 4-layer minimum for any product with clocks > 25 MHz
   - Signal-Ground-Power-Signal

2. **I/O filtering:**
   - Common-mode choke on all external cables
   - Ferrite beads on I/O power pins
   - TVS diodes for ESD

3. **Clock management:**
   - Spread-spectrum clocking (SSC): -6 to -10 dB peak reduction
   - Series termination: 33-47 ohm near source
   - Avoid clock traces near board edges

4. **Cable/connector treatment:**
   - Shield all external cables
   - 360-degree shield termination
   - Filter at cable entry point

5. **Enclosure:**
   - Conductive gaskets at seams
   - Waveguide-beyond-cutoff ventilation
   - Conductive coating on plastic enclosures"""
        },
    ]
    for r in rules:
        emit(r["q"], r["a"])


def gen_parametric_emc():
    """Generate parametric variations for EMC topics."""
    # Product-specific EMC design
    for product in PRODUCTS:
        for std_name, std_desc, _ in random.sample(EMISSION_STANDARDS, 2):
            emit(
                f"What EMC considerations are specific to designing a {product} for {std_name} compliance?",
                f"""EMC design considerations for {product} ({std_name}):

**{std_name}** ({std_desc}):

**Product-specific noise sources for {product}:**
{"- Switching power supply (SMPS): Primary noise source 150 kHz-30 MHz" if "motor" in product or "LED" in product or "power" in product else "- Digital clocks and processors: Harmonics from MHz to GHz"}
{"- Motor PWM switching: Large dI/dt causes conducted and radiated emissions" if "motor" in product else ""}
{"- LED driver switching: Fast edges on high-current paths" if "LED" in product else ""}
{"- High-speed digital interfaces (USB, Ethernet): Clock harmonics above 30 MHz" if "IoT" in product or "consumer" in product or "industrial" in product else ""}
{"- Patient-connected circuits: Strict leakage current limits" if "medical" in product else ""}
{"- Automotive bus transceivers (CAN, LIN): Conducted emissions on bus cables" if "automotive" in product else ""}

**Key mitigation strategies:**
1. {"Input filter: Pi-filter with X2/Y2 caps and CM choke" if "motor" in product or "LED" in product else "Decoupling: 100nF per VDD pin, additional HF caps near high-speed ICs"}
2. {"Shielded enclosure: Aluminum or steel, gaskets at seams" if "industrial" in product or "medical" in product else "PCB ground pour with via stitching"}
3. {"Cable filtering: CM choke + TVS at every I/O connector" if "industrial" in product else "I/O filtering: TVS diodes and ferrite beads at connectors"}
4. {"Spread-spectrum clocking on all non-critical clocks" if "consumer" in product or "IoT" in product else "Slew rate control on high-current switching"}

**{std_name} test plan:**
- Conducted emissions: LISN measurement, 150 kHz-30 MHz
- Radiated emissions: 30 MHz-1 GHz ({"6 GHz for " + std_name if std_name == "FCC Part 15" else "1 GHz minimum"})
{"- Immunity: IEC 61000-4-2 (ESD), -4-3 (radiated), -4-4 (EFT), -4-5 (surge)" if "industrial" in product or "medical" in product else "- Immunity: per product-specific standard"}"""
            )

    # Connector-specific EMC
    for conn in CONNECTOR_TYPES:
        emit(
            f"What EMC filtering and shielding is needed for a {conn} connector?",
            f"""EMC design for {conn} connector:

**Shield/ground connection:**
{"- USB-C: Metal shell connected to chassis ground via low-impedance path" if conn == "USB-C" else ""}
{"- USB-A: Shield pins (1,4) to chassis ground, data shield to signal ground via 1nF cap" if conn == "USB-A" else ""}
{"- RJ45: Integrated magnetics with CM choke, shield to chassis ground" if conn == "RJ45" else ""}
{"- HDMI: Metal shell to chassis ground, individual pair shielding" if conn == "HDMI Type-A" else ""}
{"- D-Sub: Backshell with 360-degree shield termination" if conn == "D-Sub 9" else ""}
{"- SMA: Connector ground directly to ground plane, continuous ground around connector" if conn == "SMA" else ""}
{"- BNC: Ground sleeve to chassis, bulkhead mount preferred" if conn == "BNC" else ""}
{"- Screw terminal: No inherent shielding — add external CM choke and TVS" if conn == "screw terminal" else ""}

**Filtering components at connector:**
```
{conn}
  Pin 1 ── TVS to GND ── Ferrite bead ── Signal trace
  Pin 2 ── TVS to GND ── Ferrite bead ── Signal trace
  Shield ── Low-Z to chassis GND (multiple vias)
```

**PCB layout near {conn}:**
- Place connector at board edge
- Ground pour under and around connector
- Via stitching around connector footprint
- Filter components between connector and first via to inner layers
- Keep filtered and unfiltered sides separate (don't route unfiltered signals past filter)

**Common mistakes:**
- Long traces from connector to filter components (defeats filtering)
- Missing ground connection on connector shell
- Routing high-speed signals under connector (coupling)"""
        )

    # Additional parametric EMC questions
    for _ in range(100):
        freq = random.choice(FREQUENCIES)
        product = random.choice(PRODUCTS)
        material = random.choice(SHIELD_MATERIALS)
        layers = random.choice(PCB_LAYERS)

        topic = random.choice([
            "crosstalk",
            "ground bounce",
            "power integrity",
            "signal integrity vs EMC",
            "thermal management and EMC",
        ])

        if topic == "crosstalk":
            emit(
                f"How do I minimize crosstalk on a {layers} PCB for a {product}?",
                f"""Crosstalk reduction on {layers} PCB for {product}:

**Types:**
- **Near-end (NEXT)**: Noise at the near end of victim trace (backward crosstalk)
- **Far-end (FEXT)**: Noise at the far end of victim trace (forward crosstalk)
- In stripline (inner layers): FEXT cancels, only NEXT remains
- In microstrip (outer layers): Both NEXT and FEXT are present

**Design rules:**
1. **3W rule**: Space traces center-to-center >= 3x trace width
   - Reduces crosstalk by ~70%
   - For critical signals: 5W spacing

2. **Ground reference**: Both aggressor and victim must reference the same ground plane
   - Different reference planes = massive crosstalk

3. **Trace length**: Crosstalk couples over parallel length
   - Minimize parallel run length
   - Stagger routing (different layers or offset)

4. **Guard traces**: Ground trace between sensitive signals
   - Via-stitched to ground plane every lambda/20
   - Without vias, guard traces can make crosstalk WORSE

5. **Layer assignment**:
   - Route sensitive signals on inner layers (stripline)
   - Keep high-speed clocks away from analog signals
   - Use different layers for orthogonal routing

**For {product}:**
{"- Keep motor drive PWM signals away from sensor analog inputs" if "motor" in product else "- Keep switching regulator traces away from ADC inputs" if "medical" in product or "industrial" in product else "- Separate digital and analog signal groups"}"""
            )
        elif topic == "ground bounce":
            emit(
                f"How do I prevent ground bounce in a {product} design for EMC?",
                f"""Ground bounce prevention for {product}:

**What is ground bounce:**
Simultaneous switching of multiple I/O pins causes transient voltage on the ground plane due to parasitic inductance in the ground path (package pins, vias, traces).

**Impact on EMC:**
- Ground bounce modulates all signals referenced to that ground
- Creates common-mode noise on I/O cables
- Typical amplitude: 100-500 mV at 100-500 MHz

**Prevention:**

1. **Multiple ground pins/vias:**
   - Use all available ground pins on ICs
   - Place decoupling caps between VDD and GND pins (not far-away ground)

2. **Low-inductance ground path:**
   - Wide traces or planes
   - Via arrays (multiple vias in parallel)
   - Minimize ground path length from IC to bulk ground

3. **Reduce simultaneous switching:**
   - Stagger output enable signals
   - Use slower slew-rate drivers where speed allows
   - Limit number of simultaneously switching outputs

4. **I/O driver strength:**
   - Use minimum drive strength that meets timing
   - {"4-8 mA drive for most GPIO" if "IoT" in product or "industrial" in product else "Programmable drive strength where available"}

5. **Decoupling:**
   - 100nF + 10nF per VDD/GND pin pair
   - Place within 2mm of pins
   - Via-in-pad for best results"""
            )
        elif topic == "power integrity":
            emit(
                f"How does power integrity affect EMC in a {product}?",
                f"""Power integrity and EMC relationship for {product}:

**Connection between PI and EMC:**
- Noisy power rails → noisy signals → more emissions
- Power plane resonances create peaks in emission spectrum
- Inadequate decoupling = voltage droop = signal jitter = wider spectrum

**Power integrity for EMC:**

1. **Target impedance:**
   Z_target = deltaV_allowed / deltaI_max
   - {"For digital ICs: Z < 50 mOhm from DC to 1 GHz" if "IoT" in product or "consumer" in product else "For power electronics: Z < 100 mOhm from DC to switching frequency harmonics"}

2. **Capacitor strategy:**
   | Frequency range | Capacitor | Purpose |
   |----------------|-----------|---------|
   | DC - 10 kHz | Bulk (100-1000 uF) | Energy reservoir |
   | 10 kHz - 10 MHz | Bypass (1-10 uF ceramic) | Transient response |
   | 10 MHz - 500 MHz | Decoupling (100nF ceramic) | IC switching noise |
   | 500 MHz - 2 GHz | HF (1-10nF, 0402) | Package resonance |
   | > 2 GHz | Plane capacitance | Interplane coupling |

3. **Plane pair design:**
   - Thin dielectric between power and ground planes (3-5 mil)
   - Provides distributed capacitance: C = epsilon * A / d

4. **Anti-resonance:**
   - Multiple capacitor values prevent anti-resonance peaks
   - Add ESR (ferrite bead) to dampen resonances"""
            )
        else:
            emit(
                f"How do I balance signal integrity and EMC in a {product} design?",
                f"""Signal integrity vs EMC tradeoffs for {product}:

**The fundamental tension:**
- SI wants: fast edges (clean signals, meet timing)
- EMC wants: slow edges (less harmonic content, less radiation)

**Finding the balance:**

1. **Use the minimum edge rate that meets timing:**
   - Calculate required rise time: Tr = 0.35 / BW_signal
   - {"USB 2.0: Tr ~ 4-20 ns (don't make faster)" if "IoT" in product or "consumer" in product else "CAN: Tr ~ 50-200 ns (plenty of margin)"}
   - Add series termination resistors to slow edges

2. **Controlled impedance:**
   - Benefits both SI and EMC
   - Reduces reflections (SI) and radiation (EMC)
   - {"50 ohm single-ended, 100 ohm differential typical" if "consumer" in product else "90 ohm USB, 100 ohm Ethernet, 120 ohm CAN"}

3. **Termination strategy:**
   - Series termination: Good for SI and EMC (reduces ringing and radiation)
   - Parallel termination: Best SI but wastes power
   - AC termination: Compromise (RC to ground)

4. **Return path continuity:**
   - Critical for both SI and EMC
   - Signal crossing reference plane boundary = impedance discontinuity (SI) = large loop (EMC)

5. **Differential signaling:**
   - Inherently good EMC (fields cancel in far field)
   - Good SI (common-mode rejection)
   - Used in: {"USB, HDMI, Ethernet, LVDS" if "consumer" in product else "CAN, RS-485, LVDS"}"""
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gen_standards()
    gen_shielding()
    gen_filtering()
    gen_pcb_layout()
    gen_conducted_radiated()
    gen_esd_protection()
    gen_precompliance()
    gen_cable_shielding()
    gen_emc_design_rules()
    gen_parametric_emc()


if __name__ == "__main__":
    main()
