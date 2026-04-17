#!/usr/bin/env python3
"""Generate SPICE simulation training Q&A pairs.

Covers ngspice syntax, circuit analysis, directives, behavioral sources,
convergence issues, and complete netlist examples.

Output: JSONL to stdout.
"""
from __future__ import annotations

import json
import random
import sys

random.seed(43)

DOMAIN = "spice"


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

RESISTORS = ["100", "220", "470", "1k", "2.2k", "4.7k", "10k", "22k", "47k", "100k", "1Meg"]
CAPACITORS = ["10p", "22p", "47p", "100p", "1n", "10n", "100n", "1u", "10u", "47u", "100u", "470u"]
INDUCTORS = ["1u", "10u", "22u", "47u", "100u", "220u", "470u", "1m", "10m", "100m"]
VOLTAGES = ["1.8", "2.5", "3.3", "5", "9", "12", "15", "24", "48"]
FREQUENCIES = ["100", "1k", "10k", "100k", "1Meg", "10Meg"]
MOSFETS = ["IRF540N", "IRFP260N", "BSS138", "AO3400A", "IRF9540N", "SI2301", "IRF3205", "2N7002"]
BJTS = ["2N2222", "2N3904", "2N3906", "BC547", "BC557", "BD139", "TIP31C", "TIP32C"]
OPAMPS = ["LM741", "LM358", "TL072", "OPA2134", "AD8605", "MCP6002", "LMV321", "NE5532"]
DIODES = ["1N4148", "1N4007", "1N5819", "BAT54", "BAS16", "UF4007", "MBR1045"]
ZENERS = ["BZX55C3V3", "BZX55C5V1", "1N4733A", "1N4742A", "1N4744A"]
REGULATORS_IC = ["LM317", "LM7805", "LM7812", "LM1117", "LM2596"]

ANALYSIS_TYPES = [
    (".tran", "transient", "time-domain simulation"),
    (".ac", "AC sweep", "frequency-domain response"),
    (".dc", "DC sweep", "operating point vs. parameter"),
    (".op", "operating point", "DC bias point"),
    (".noise", "noise analysis", "noise spectral density"),
    (".tf", "transfer function", "small-signal gain and impedance"),
]

FILTER_TYPES = ["low-pass", "high-pass", "band-pass", "band-stop", "notch"]

# ---------------------------------------------------------------------------
# Complete netlist generators
# ---------------------------------------------------------------------------

def gen_voltage_divider():
    """Voltage divider netlists."""
    for _ in range(30):
        vin = random.choice(VOLTAGES)
        r1 = random.choice(RESISTORS)
        r2 = random.choice(RESISTORS)

        emit(
            f"Write an ngspice netlist for a voltage divider with Vin={vin}V, R1={r1} and R2={r2}.",
            f"""Complete ngspice netlist for a voltage divider:

```spice
* Voltage Divider — Vin={vin}V, R1={r1}, R2={r2}
.title Voltage Divider

V1 in 0 DC {vin}
R1 in out {r1}
R2 out 0 {r2}

.control
op
print v(out)
print v(out)/v(in)
echo "Theoretical Vout = {vin} * {r2} / ({r1} + {r2})"
.endc

.end
```

The output voltage follows the divider equation: Vout = Vin * R2/(R1+R2).

Run with: `ngspice voltage_divider.cir`

To sweep the input voltage:
```spice
.dc V1 0 {vin} 0.1
.control
run
plot v(out)
.endc
```"""
        )


def gen_rc_filters():
    """RC filter netlists."""
    for ftype in ["low-pass", "high-pass"]:
        for _ in range(25):
            r = random.choice(RESISTORS)
            c = random.choice(CAPACITORS)
            vin = random.choice(["1", "3.3", "5"])

            if ftype == "low-pass":
                netlist = f"""```spice
* RC {ftype} filter — R={r}, C={c}
.title RC Low-Pass Filter

V1 in 0 AC 1 DC 0
R1 in out {r}
C1 out 0 {c}

.ac dec 100 1 100Meg
.control
run
plot vdb(out) title 'Magnitude [dB]'
plot vp(out)*180/pi title 'Phase [degrees]'
let fc = 1/(2*pi*{r}*{c})
echo "Cutoff frequency: " fc "Hz"
.endc

.end
```"""
            else:
                netlist = f"""```spice
* RC {ftype} filter — R={r}, C={c}
.title RC High-Pass Filter

V1 in 0 AC 1 DC 0
C1 in out {c}
R1 out 0 {r}

.ac dec 100 1 100Meg
.control
run
plot vdb(out) title 'Magnitude [dB]'
plot vp(out)*180/pi title 'Phase [degrees]'
let fc = 1/(2*pi*{r}*{c})
echo "Cutoff frequency: " fc "Hz"
.endc

.end
```"""

            emit(
                f"Write an ngspice netlist for an RC {ftype} filter with R={r} and C={c}.",
                f"""Complete ngspice netlist for an RC {ftype} filter:

{netlist}

The -3dB cutoff frequency is: fc = 1/(2*pi*R*C).
- At fc, the gain is -3dB (0.707x) and phase is -45 degrees (low-pass) or +45 degrees (high-pass).
- Below fc: {"passes signal" if ftype == "low-pass" else "attenuates signal"} ({"0 dB" if ftype == "low-pass" else "-20 dB/dec"})
- Above fc: {"attenuates at -20 dB/decade" if ftype == "low-pass" else "passes signal (0 dB)"}"""
            )


def gen_rlc_filters():
    """RLC filter netlists."""
    for _ in range(30):
        r = random.choice(RESISTORS)
        l = random.choice(INDUCTORS)
        c = random.choice(CAPACITORS)

        emit(
            f"Write an ngspice netlist for a series RLC band-pass filter with R={r}, L={l}H, C={c}F.",
            f"""Complete ngspice netlist for a series RLC band-pass filter:

```spice
* Series RLC Band-Pass Filter — R={r}, L={l}, C={c}
.title RLC Band-Pass Filter

V1 in 0 AC 1 DC 0
R1 in n1 {r}
L1 n1 n2 {l}
C1 n2 out {c}
Rload out 0 1Meg

.ac dec 200 1 100Meg
.control
run
plot vdb(out) title 'Magnitude [dB]'
plot vp(out)*180/pi title 'Phase [degrees]'
let f0 = 1/(2*pi*sqrt({l}*{c}))
let Q = 1/{r} * sqrt({l}/{c})
echo "Resonant frequency:" f0 "Hz"
echo "Quality factor Q:" Q
.endc

.end
```

Key parameters:
- **Resonant frequency**: f0 = 1/(2*pi*sqrt(L*C))
- **Quality factor**: Q = (1/R)*sqrt(L/C) — higher Q means narrower bandwidth
- **Bandwidth**: BW = f0/Q
- At resonance, impedance is minimum (equals R) and current is maximum."""
        )


def gen_opamp_circuits():
    """Op-amp circuit netlists."""
    circuits = [
        {
            "name": "inverting amplifier",
            "q": "Write an ngspice netlist for an inverting op-amp amplifier with gain = -{gain}.",
            "netlist": lambda gain, rf, ri, opamp, vcc: f"""```spice
* Inverting Op-Amp Amplifier — Gain = -{gain}
.title Inverting Amplifier

* Power supply
Vcc vcc 0 DC {vcc}
Vee vee 0 DC -{vcc}

* Input signal
Vin in 0 AC 1 SIN(0 0.1 1k)

* Op-amp circuit
Ri in inv {ri}
Rf inv out {rf}
X1 0 inv vcc vee out {opamp}

.include {opamp}.lib

.tran 10u 5m
.ac dec 100 1 10Meg

.control
run
plot v(out) v(in) title 'Transient Response'
ac
plot vdb(out) title 'Frequency Response [dB]'
.endc

.end
```""",
        },
        {
            "name": "non-inverting amplifier",
            "q": "Write an ngspice netlist for a non-inverting op-amp amplifier with gain = {gain_ni}.",
            "netlist": lambda gain, rf, ri, opamp, vcc: f"""```spice
* Non-Inverting Op-Amp Amplifier — Gain = {gain}
.title Non-Inverting Amplifier

Vcc vcc 0 DC {vcc}
Vee vee 0 DC -{vcc}

Vin in 0 AC 1 SIN(0 0.1 1k)

R1 inv 0 {ri}
Rf inv out {rf}
X1 in inv vcc vee out {opamp}

.include {opamp}.lib

.tran 10u 5m

.control
run
plot v(out) v(in) title 'Transient Response'
echo "Gain = 1 + Rf/R1 = 1 + {rf}/{ri} = {gain}"
.endc

.end
```""",
        },
        {
            "name": "summing amplifier",
            "q": "Write an ngspice netlist for a summing amplifier with two inputs.",
            "netlist": lambda gain, rf, ri, opamp, vcc: f"""```spice
* Summing Amplifier — Two Inputs
.title Summing Amplifier

Vcc vcc 0 DC {vcc}
Vee vee 0 DC -{vcc}

V1 in1 0 SIN(0 0.5 1k)
V2 in2 0 SIN(0 0.3 3k)

R1 in1 inv {ri}
R2 in2 inv {ri}
Rf inv out {rf}
X1 0 inv vcc vee out {opamp}

.include {opamp}.lib

.tran 10u 5m

.control
run
plot v(out) v(in1) v(in2) title 'Summing Amplifier'
echo "Vout = -(Rf/R1)*V1 + (Rf/R2)*V2"
.endc

.end
```""",
        },
        {
            "name": "differential amplifier",
            "q": "Write an ngspice netlist for a differential amplifier.",
            "netlist": lambda gain, rf, ri, opamp, vcc: f"""```spice
* Differential Amplifier — Gain = Rf/R1 = {rf}/{ri}
.title Differential Amplifier

Vcc vcc 0 DC {vcc}
Vee vee 0 DC -{vcc}

V1 in1 0 SIN(0 0.5 1k)
V2 in2 0 SIN(0 0.3 1k 0 0 90)

R1 in1 inv {ri}
R2 in2 noninv {ri}
Rf inv out {rf}
R3 noninv 0 {rf}
X1 noninv inv vcc vee out {opamp}

.include {opamp}.lib

.tran 10u 5m

.control
run
plot v(out) v(in1) v(in2) v(in1)-v(in2) title 'Differential Amplifier'
.endc

.end
```""",
        },
        {
            "name": "integrator",
            "q": "Write an ngspice netlist for an op-amp integrator.",
            "netlist": lambda gain, rf, ri, opamp, vcc: f"""```spice
* Op-Amp Integrator
.title Integrator

Vcc vcc 0 DC {vcc}
Vee vee 0 DC -{vcc}

Vin in 0 PULSE(0 1 0 1n 1n 0.5m 1m)

R1 in inv {ri}
C1 inv out 100n
Rf inv out 1Meg
X1 0 inv vcc vee out {opamp}

.include {opamp}.lib

.tran 1u 5m

.control
run
plot v(out) v(in) title 'Integrator Response'
.endc

.end
```""",
        },
    ]

    for circ in circuits:
        for _ in range(15):
            gain = random.randint(2, 20)
            ri_val = random.choice(["1k", "2.2k", "4.7k", "10k"])
            rf_val = f"{int(float(ri_val.replace('k',''))*gain)}k"
            opamp = random.choice(OPAMPS)
            vcc = random.choice(["5", "12", "15"])

            q = circ["q"].format(gain=gain, gain_ni=gain+1)
            a = circ["netlist"](gain, rf_val, ri_val, opamp, vcc)

            emit(q, f"""Complete ngspice netlist for an {circ['name']}:

{a}

{"Gain = -Rf/Ri = -" + rf_val + "/" + ri_val + " = -" + str(gain) if "inverting" in circ["name"] and "non" not in circ["name"] else "Gain = 1 + Rf/Ri = 1 + " + rf_val + "/" + ri_val + " = " + str(gain+1) if "non-inverting" in circ["name"] else "Configuration: " + circ["name"]}

Op-amp model: {opamp}. The `.include` directive loads the SPICE model file.
Supply voltage: +/-{vcc}V (dual supply).""")


def gen_bjt_amplifier():
    """BJT amplifier netlists."""
    configs = [
        ("common-emitter", "CE"),
        ("common-collector", "CC (emitter follower)"),
        ("common-base", "CB"),
    ]
    for config_name, config_short in configs:
        for _ in range(20):
            bjt = random.choice(BJTS[:4])  # NPN
            vcc = random.choice(["5", "9", "12"])
            rc = random.choice(["1k", "2.2k", "4.7k"])
            re = random.choice(["100", "220", "470", "1k"])
            r1 = random.choice(["22k", "33k", "47k", "68k"])
            r2 = random.choice(["4.7k", "6.8k", "10k", "15k"])
            c_in = random.choice(["1u", "10u"])
            c_out = random.choice(["1u", "10u"])

            if config_name == "common-emitter":
                netlist = f"""```spice
* {config_short} Amplifier with {bjt}
.title Common-Emitter Amplifier

Vcc vcc 0 DC {vcc}
Vin in 0 AC 1 SIN(0 10m 1k)

* Bias network
R1 vcc base {r1}
R2 base 0 {r2}

* Amplifier
Cin in base {c_in}
Q1 collector base emitter {bjt}
Rc vcc collector {rc}
Re emitter 0 {re}
Ce emitter 0 100u
Cout collector out {c_out}
Rload out 0 10k

.model {bjt} NPN(BF=200 IS=1e-14 VAF=100 CJC=5p CJE=10p TF=0.3n)

.tran 10u 10m
.ac dec 100 10 10Meg

.control
run
plot v(out) v(in) title 'CE Amplifier Transient'
ac
plot vdb(out) title 'Frequency Response [dB]'
.endc

.end
```"""
            elif config_name == "common-collector":
                netlist = f"""```spice
* {config_short} with {bjt}
.title Emitter Follower

Vcc vcc 0 DC {vcc}
Vin in 0 AC 1 SIN(0 0.5 1k)

R1 vcc base {r1}
R2 base 0 {r2}
Cin in base {c_in}
Q1 vcc base emitter {bjt}
Re emitter 0 {re}
Cout emitter out {c_out}
Rload out 0 10k

.model {bjt} NPN(BF=200 IS=1e-14 VAF=100)

.tran 10u 10m

.control
run
plot v(out) v(in) title 'Emitter Follower'
echo "Gain ~ 1, Low output impedance"
.endc

.end
```"""
            else:
                netlist = f"""```spice
* {config_short} Amplifier with {bjt}
.title Common-Base Amplifier

Vcc vcc 0 DC {vcc}
Vin in 0 AC 1 SIN(0 10m 1k)

R1 vcc base {r1}
R2 base 0 {r2}
Cin in emitter {c_in}
Re emitter 0 {re}
Q1 collector base emitter {bjt}
Rc vcc collector {rc}
Cout collector out {c_out}
Rload out 0 10k
Cb base 0 100u

.model {bjt} NPN(BF=200 IS=1e-14 VAF=100)

.tran 10u 10m
.ac dec 100 10 100Meg

.control
run
plot v(out) v(in) title 'CB Amplifier'
ac
plot vdb(out) title 'CB Frequency Response'
.endc

.end
```"""

            emit(
                f"Write an ngspice netlist for a {config_name} BJT amplifier using {bjt} with Vcc={vcc}V.",
                f"""Complete ngspice netlist for a {config_name} ({config_short}) amplifier:

{netlist}

Bias point design:
- R1={r1}, R2={r2} set the base voltage via voltage divider
- Rc={rc} sets the collector load
- Re={re} provides thermal stability (bypassed by Ce for AC gain)
- Coupling capacitors Cin={c_in}, Cout={c_out} block DC

{"AC voltage gain: Av ~ -Rc/re (with Ce bypass) where re = 26mV/Ic" if config_name == "common-emitter" else "Voltage gain ~ 1 (unity), very low output impedance" if config_name == "common-collector" else "Current gain ~ 1, good high-frequency response"}"""
            )


def gen_mosfet_circuits():
    """MOSFET switch and driver netlists."""
    for _ in range(30):
        mosfet = random.choice(MOSFETS[:4])  # N-channel
        vdd = random.choice(["5", "12", "24"])
        rload = random.choice(["10", "22", "47", "100"])
        rgate = random.choice(["100", "220", "470"])
        freq = random.choice(["1k", "10k", "100k"])

        emit(
            f"Write an ngspice netlist for a MOSFET switch using {mosfet} driving a {rload} ohm load at {vdd}V.",
            f"""Complete ngspice netlist for an N-channel MOSFET switch:

```spice
* MOSFET Switch — {mosfet}, {vdd}V, {rload} ohm load
.title MOSFET Switch

Vdd vdd 0 DC {vdd}
Vpwm gate_in 0 PULSE(0 {vdd} 0 10n 10n {{1/{freq}/2}} {{1/{freq}}})

Rgate gate_in gate {rgate}
M1 drain gate 0 0 {mosfet}_MODEL W=1 L=1

Rload vdd drain {rload}

.model {mosfet}_MODEL NMOS(LEVEL=1 VTO=2 KP=20m LAMBDA=0.01 CBD=50p CBS=50p)

.tran 1u {{5/{freq}}}

.control
run
plot v(gate) v(drain) title 'MOSFET Switch'
plot i(Vdd) title 'Load Current'
let P_load = v(vdd,drain) * i(Rload)
plot P_load title 'Load Power'
.endc

.end
```

Key parameters:
- **Rgate={rgate}**: Limits gate charge current, controls switching speed
- **VTO** (threshold): Voltage at which MOSFET turns on (~2V for logic-level)
- **Switching frequency**: {freq}Hz
- **Load current**: ~{vdd}/{rload} = {float(vdd)/float(rload):.1f}A when on

For a P-channel version, swap source/drain connections and invert the gate drive."""
        )


def gen_buck_converter():
    """Buck converter netlists."""
    for _ in range(25):
        vin = random.choice(["12", "24", "48"])
        vout = random.choice(["3.3", "5"])
        freq = random.choice(["100k", "200k", "500k"])
        l = random.choice(["10u", "22u", "47u"])
        c = random.choice(["47u", "100u", "220u"])

        duty = float(vout) / float(vin)

        emit(
            f"Write an ngspice netlist for a buck converter: Vin={vin}V to Vout={vout}V at {freq}Hz switching frequency.",
            f"""Complete ngspice netlist for a buck converter:

```spice
* Buck Converter — {vin}V to {vout}V, fsw={freq}Hz
.title Buck Converter

Vin vin 0 DC {vin}

* PWM gate drive (duty cycle ~ {duty:.2f})
Vpwm gate 0 PULSE(0 10 0 10n 10n {{{duty}/{freq}}} {{1/{freq}}})

* Power stage
S1 vin sw gate 0 SMOD
.model SMOD SW(VT=5 RON=0.05 ROFF=1Meg)

D1 0 sw DMOD
.model DMOD D(IS=1e-14 RS=0.01 BV=100 CJO=100p TT=10n)

L1 sw out {l} IC=0
C1 out 0 {c} IC={vout}
Rload out 0 {{({vout}*{vout})/10}}

* Parasitic ESR
Resr out_esr out 0.01

.tran 100n 1m UIC

.control
set maxstep=50n
run
plot v(out) title 'Output Voltage'
plot v(sw) title 'Switch Node'
plot i(L1) title 'Inductor Current'
let ripple = maximum(v(out)) - minimum(v(out))
echo "Output ripple:" ripple "V"
.endc

.end
```

Design equations:
- **Duty cycle**: D = Vout/Vin = {vout}/{vin} = {duty:.2f}
- **Inductor ripple**: deltaI = (Vin-Vout)*D/(L*fsw)
- **Output ripple**: deltaV = deltaI/(8*C*fsw)
- **L={l}**: Sized for continuous conduction mode (CCM)
- **C={c}**: Sized for acceptable output ripple

The switch S1 and diode D1 model a basic asynchronous buck. For synchronous buck, replace D1 with a second switch."""
        )


def gen_hbridge():
    """H-bridge motor driver netlists."""
    for _ in range(15):
        vdd = random.choice(["12", "24"])
        mosfet_n = random.choice(MOSFETS[:4])
        freq = random.choice(["10k", "20k"])

        emit(
            f"Write an ngspice netlist for an H-bridge motor driver at {vdd}V.",
            f"""Complete ngspice netlist for an H-bridge motor driver:

```spice
* H-Bridge Motor Driver — {vdd}V
.title H-Bridge DC Motor Driver

Vdd vdd 0 DC {vdd}

* Gate drive signals (complementary PWM with dead time)
Vg1 g_hs1 0 PULSE(0 10 0 10n 10n 45u 100u)
Vg2 g_ls2 0 PULSE(0 10 0 10n 10n 45u 100u)
Vg3 g_hs2 0 PULSE(0 10 50u 10n 10n 45u 100u)
Vg4 g_ls1 0 PULSE(0 10 50u 10n 10n 45u 100u)

* High-side MOSFETs (P-channel simplified as switches)
S1 vdd motor_a g_hs1 0 SMOD
S3 vdd motor_b g_hs2 0 SMOD

* Low-side MOSFETs
S2 motor_a 0 g_ls1 0 SMOD
S4 motor_b 0 g_ls2 0 SMOD

.model SMOD SW(VT=5 RON=0.05 ROFF=1Meg)

* Motor model (simplified: R + L + back-EMF)
Rmotor motor_a motor_mid 1
Lmotor motor_mid motor_bemf 1m
Vbemf motor_bemf motor_b DC 0

* Freewheeling diodes
D1 0 motor_a DMOD
D2 motor_a vdd DMOD
D3 0 motor_b DMOD
D4 motor_b vdd DMOD
.model DMOD D(IS=1e-14 RS=0.01)

.tran 1u 1m

.control
run
plot v(motor_a) v(motor_b) title 'Bridge Outputs'
plot v(motor_a,motor_b) title 'Motor Voltage'
plot i(Lmotor) title 'Motor Current'
.endc

.end
```

H-bridge operation:
- **Forward**: S1+S4 ON, S2+S3 OFF (motor_a=Vdd, motor_b=GND)
- **Reverse**: S2+S3 ON, S1+S4 OFF (motor_a=GND, motor_b=Vdd)
- **Brake**: S2+S4 ON (both low-side, motor shorted)
- **Coast**: All OFF (motor freewheels through diodes)
- **Dead time**: 5us gap prevents shoot-through (both high and low side on simultaneously)"""
        )


def gen_analysis_types():
    """Analysis type explanations."""
    analyses = [
        {
            "q": "Explain the .tran (transient) analysis in ngspice with an example.",
            "a": """`.tran` performs time-domain simulation — it solves the circuit equations at each time step.

Syntax:
```spice
.tran TSTEP TSTOP [TSTART [TMAX]] [UIC]
```

Parameters:
- **TSTEP**: Suggested time step (ngspice auto-adjusts)
- **TSTOP**: End time
- **TSTART**: Start saving data (default: 0)
- **TMAX**: Maximum time step (controls accuracy)
- **UIC**: Use Initial Conditions (skip DC operating point)

Example — RC charging:
```spice
.title RC Transient
V1 in 0 PULSE(0 5 1m 1n 1n 10m 20m)
R1 in out 10k
C1 out 0 1u IC=0

.tran 10u 50m UIC
.control
run
plot v(out) v(in) title 'RC Charging'
let tau = 10k * 1u
echo "Time constant tau =" tau "seconds"
.endc
.end
```

The time constant tau = R*C = 10k * 1u = 10ms. The capacitor reaches 63.2% of Vin after one tau, 99.3% after 5*tau."""
        },
        {
            "q": "Explain AC analysis (.ac) in ngspice with an example.",
            "a": """`.ac` computes the small-signal frequency response by linearizing the circuit around its DC operating point.

Syntax:
```spice
.ac DEC|OCT|LIN NPOINTS FSTART FSTOP
```

Sweep types:
- **DEC**: Logarithmic, N points per decade
- **OCT**: Logarithmic, N points per octave
- **LIN**: Linear, N total points

Example — Low-pass filter:
```spice
.title AC Analysis Example
V1 in 0 AC 1
R1 in out 10k
C1 out 0 10n

.ac dec 100 1 100Meg
.control
run
plot vdb(out) title 'Magnitude [dB]'
plot 180/pi*vp(out) title 'Phase [degrees]'
let f3dB = 1/(2*pi*10k*10n)
echo "-3dB frequency:" f3dB "Hz"
meas ac f3db WHEN vdb(out)=-3 FALL=1
.endc
.end
```

Key functions:
- `vdb(node)`: Magnitude in dB = 20*log10(|V|)
- `vp(node)`: Phase in radians
- `vm(node)`: Magnitude (linear)

Important: All AC sources must have `AC` value. Use `AC 1` for unity gain reference."""
        },
        {
            "q": "Explain DC sweep (.dc) analysis in ngspice.",
            "a": """`.dc` sweeps a source value and computes the DC operating point at each step.

Syntax:
```spice
.dc SRCNAME VSTART VSTOP VINCR [SRC2 START2 STOP2 INCR2]
```

Example — Diode I-V curve:
```spice
.title Diode I-V Characteristic
V1 anode 0 DC 0
D1 anode cathode DMOD
R1 cathode 0 1
.model DMOD D(IS=1e-14 N=1.05 RS=0.5)

.dc V1 -1 1 0.01
.control
run
plot -i(V1) title 'Diode Current vs Voltage'
plot log(abs(i(V1))) title 'Log(I) vs V'
.endc
.end
```

Nested sweep (2D):
```spice
* MOSFET output characteristics
.dc Vds 0 10 0.1 Vgs 1 5 1
```
This sweeps Vds from 0 to 10V while stepping Vgs from 1V to 5V in 1V increments.

Use cases:
- Transfer characteristics (Vout vs Vin)
- Device I-V curves
- Sensitivity analysis
- Finding bias points"""
        },
        {
            "q": "How do I use .op (operating point) analysis in ngspice?",
            "a": """`.op` computes the DC operating (bias) point of the circuit with all capacitors open and inductors shorted.

```spice
.title Operating Point Analysis
Vcc vcc 0 DC 12
R1 vcc base 47k
R2 base 0 10k
Q1 collector base emitter 2N2222
Rc vcc collector 2.2k
Re emitter 0 470

.model 2N2222 NPN(BF=200 IS=1e-14 VAF=100)

.op
.control
run
print all
print @Q1[ic]
print @Q1[ib]
print @Q1[vbe]
print @Q1[vce]
print @Q1[gm]
echo "Bias point analysis complete"
.endc
.end
```

`.op` output includes:
- All node voltages
- All branch currents
- Device parameters (accessed via `@device[parameter]`):
  - `@Q1[ic]`: Collector current
  - `@Q1[gm]`: Transconductance
  - `@Q1[vbe]`: Base-emitter voltage

This is essential before AC analysis — ngspice automatically runs `.op` to find the linearization point."""
        },
        {
            "q": "How do I run Monte Carlo analysis in ngspice?",
            "a": """Monte Carlo analysis uses `.control` scripting to run multiple simulations with randomized component values:

```spice
.title Monte Carlo Analysis — RC Filter
.param R_nom = 10k
.param C_nom = 10n

V1 in 0 AC 1
R1 in out {R_nom}
C1 out 0 {C_nom}

.ac dec 50 100 10Meg

.control
let num_runs = 100
let tolerance = 0.1

let run = 0
while run < num_runs
    alter R1 = {R_nom} * (1 + tolerance * sunif(0))
    alter C1 = {C_nom} * (1 + tolerance * sunif(0))
    ac dec 50 100 10Meg
    let run = run + 1
end

plot db(v(out)) title 'Monte Carlo AC Response (100 runs, 10% tolerance)'
.endc
.end
```

Key functions:
- `sunif(seed)`: Uniform random [-1, 1]
- `sgauss(seed)`: Gaussian random (mean=0, sigma=1)
- `unif(nominal, reltol)`: Uniform around nominal
- `gauss(nominal, reltol, sigma)`: Gaussian around nominal

For systematic worst-case analysis:
```spice
.control
foreach R_val 9k 10k 11k
    foreach C_val 9n 10n 11n
        alter R1 = $R_val
        alter C1 = $C_val
        ac dec 50 100 10Meg
    end
end
.endc
```"""
        },
        {
            "q": "How do I use .noise analysis in ngspice?",
            "a": """`.noise` computes the noise spectral density at an output node referenced to an input source:

```spice
.title Noise Analysis — CE Amplifier
Vcc vcc 0 DC 12
Vin in 0 AC 1

R1 vcc base 47k
R2 base 0 10k
Cin in base 10u
Q1 collector base emitter 2N3904
Rc vcc collector 4.7k
Re emitter 0 1k
Ce emitter 0 100u
Cout collector out 10u
Rload out 0 10k

.model 2N3904 NPN(BF=300 IS=6.7e-15 VAF=100 KF=1e-16 AF=1)

.noise v(out) Vin dec 50 10 10Meg

.control
run
setplot noise1
plot onoise_spectrum title 'Output Noise Density [V/sqrt(Hz)]'
plot inoise_spectrum title 'Input-Referred Noise [V/sqrt(Hz)]'

setplot noise2
print onoise_total
print inoise_total
.endc
.end
```

Parameters:
- `v(out)`: Output node for noise measurement
- `Vin`: Input source for input-referred noise
- Sweep: same as .ac (dec/oct/lin points fstart fstop)

Noise sources modeled:
- **Thermal noise** (Johnson): 4kT*R
- **Shot noise**: 2qI
- **Flicker (1/f) noise**: KF * I^AF / f
- KF and AF in the .model statement control 1/f noise"""
        },
    ]
    for a in analyses:
        emit(a["q"], a["a"])


def gen_behavioral_sources():
    """B-source and dependent source examples."""
    sources = [
        {
            "q": "How do I use B (behavioral) sources in ngspice?",
            "a": """B sources define arbitrary voltage or current as mathematical expressions:

```spice
* Behavioral voltage source — full-wave rectifier
B1 out 0 V = abs(v(in))

* Behavioral current source — voltage-controlled current
B2 out 0 I = v(ctrl) * 1m

* Nonlinear function — soft limiter
B3 out 0 V = 5*tanh(v(in)/2)

* Conditional / piecewise
B4 out 0 V = v(in) > 2.5 ? 5 : 0

* Time-dependent
B5 out 0 V = sin(2*3.14159*1k*time)

* Complex expression — PID controller
B6 out 0 V = 1.0*v(error) + 0.1*idt(v(error)) + 0.01*ddt(v(error))
```

Available functions: `abs`, `sqrt`, `exp`, `log`, `log10`, `sin`, `cos`, `tan`, `atan`, `sinh`, `cosh`, `tanh`, `min`, `max`, `pow`, `sgn`, `floor`, `ceil`

Special variables:
- `time`: Current simulation time
- `temper`: Current temperature
- `hertz`: Current frequency (AC analysis)

Derivatives and integrals:
- `ddt(x)`: Time derivative of x
- `idt(x)`: Time integral of x
- `sdt(x)`: Like idt but with initial condition support"""
        },
        {
            "q": "Explain E, F, G, H dependent sources in SPICE.",
            "a": """SPICE has four linear dependent sources:

**E — Voltage-Controlled Voltage Source (VCVS)**:
```spice
E1 out+ out- ctrl+ ctrl- gain
E1 out 0 in 0 10    ; Vout = 10 * V(in)
```

**F — Current-Controlled Current Source (CCCS)**:
```spice
F1 out+ out- Vsense gain
Vsense n1 n2 DC 0   ; Zero-volt source to sense current
F1 out 0 Vsense 5   ; Iout = 5 * I(Vsense)
```

**G — Voltage-Controlled Current Source (VCCS)**:
```spice
G1 out+ out- ctrl+ ctrl- transconductance
G1 out 0 in 0 1m    ; Iout = 1mA/V * V(in)
```

**H — Current-Controlled Voltage Source (CCVS)**:
```spice
H1 out+ out- Vsense transresistance
Vsense n1 n2 DC 0
H1 out 0 Vsense 1k  ; Vout = 1kOhm * I(Vsense)
```

Example — ideal transformer (1:N):
```spice
* Ideal transformer, turns ratio 1:5
E1 sec+ sec- pri+ pri- 5
F1 pri+ pri- Vsec 5
Vsec sec+ sec_mid DC 0
```"""
        },
    ]
    for s in sources:
        emit(s["q"], s["a"])


def gen_convergence():
    """Convergence issues and fixes."""
    issues = [
        {
            "q": "My ngspice simulation fails with 'no convergence in DC operating point'. How do I fix it?",
            "a": """DC convergence failure is common with complex nonlinear circuits. Here are systematic fixes:

**1. Add RSHUNT to help convergence:**
```spice
.options RSHUNT=1e8
```
This adds a high-value resistor from every node to ground, preventing floating nodes.

**2. Relax convergence criteria:**
```spice
.options ABSTOL=1e-10 RELTOL=0.01 VNTOL=1e-4
.options ITL1=500 ITL2=200 ITL4=100
```
- `ABSTOL`: Absolute current tolerance
- `RELTOL`: Relative tolerance (default 0.001)
- `VNTOL`: Voltage tolerance
- `ITL1`: DC iteration limit (default 100)

**3. Use GMIN stepping:**
```spice
.options GMIN=1e-12
.options GMINSTEPS=100
```

**4. Add initial conditions:**
```spice
.ic v(node1)=5 v(node2)=3.3
.nodeset v(node1)=5 v(node2)=3.3
.tran 1u 10m UIC
```
`.nodeset` provides a starting guess; `.ic` forces the value.

**5. Source stepping (automatic in ngspice):**
```spice
.options SRCSTEPS=100
```

**6. Check for circuit issues:**
- Floating nodes (add pull-down resistors)
- Missing ground connection
- Inductor loops without resistance (add small series R)
- Capacitor cutsets (add small parallel R)"""
        },
        {
            "q": "How do I fix 'timestep too small' errors in ngspice transient analysis?",
            "a": """The 'timestep too small' error occurs when ngspice cannot reduce the timestep enough to maintain accuracy:

**1. Increase maximum iterations:**
```spice
.options ITL4=200    ; Default is 10
.options RELTOL=0.01 ; Relax from default 0.001
```

**2. Use the GEAR integration method:**
```spice
.options METHOD=GEAR  ; More stable than default TRAP for stiff circuits
.options MAXORD=3     ; Maximum order for GEAR (2-6)
```

**3. Limit timestep size explicitly:**
```spice
.tran 1u 10m 0 100n  ; Set TMAX=100n
```

**4. Smooth discontinuities:**
Replace ideal switches with smooth transitions:
```spice
* Instead of:
Vpwm gate 0 PULSE(0 5 0 0 0 5u 10u)
* Use finite rise/fall times:
Vpwm gate 0 PULSE(0 5 0 10n 10n 5u 10u)
```

**5. Add snubber circuits:**
```spice
* RC snubber across switches/diodes
Rs node1 snub 10
Cs snub node2 1n
```

**6. Check for very stiff circuits:**
- Wide time constant spread (fast switching + slow thermal)
- Ideal switches with zero transition time
- Very small parasitic values next to large ones"""
        },
        {
            "q": "What are the most important .options settings in ngspice for reliable simulation?",
            "a": """Key `.options` settings for ngspice:

```spice
* ---- Accuracy ----
.options RELTOL=0.001    ; Relative tolerance (default, tighten to 1e-4 for precision)
.options ABSTOL=1e-12    ; Absolute current tolerance (A)
.options VNTOL=1e-6      ; Absolute voltage tolerance (V)
.options CHGTOL=1e-14    ; Charge tolerance (C)

* ---- Convergence ----
.options ITL1=200        ; DC iteration limit (default 100)
.options ITL2=100        ; DC transfer curve iterations
.options ITL4=50         ; Transient timepoint iterations (default 10)
.options GMIN=1e-12      ; Minimum conductance (default 1e-12)
.options RSHUNT=1e8      ; Shunt resistance, every node to ground
.options GMINSTEPS=50    ; GMIN stepping iterations
.options SRCSTEPS=50     ; Source stepping iterations

* ---- Integration ----
.options METHOD=TRAP     ; TRAP (default) or GEAR
.options MAXORD=2        ; Max order for GEAR method

* ---- Speed ----
.options TRTOL=7         ; Transient error tolerance factor (default 7)
.options PIVTOL=1e-13    ; Pivot tolerance for matrix solver
.options TEMP=27         ; Simulation temperature (C)

* ---- Display ----
.options NOACCT          ; Suppress accounting statistics
.options NOPAGE          ; No page breaks in output
```

Start with defaults. Only change when you encounter convergence problems.
Recommended for power electronics:
```spice
.options ITL1=500 ITL4=100 METHOD=GEAR RELTOL=0.005 RSHUNT=1e8
```"""
        },
    ]
    for i in issues:
        emit(i["q"], i["a"])


def gen_subcircuit():
    """Subcircuit definitions."""
    subcircuits = [
        {
            "q": "How do I create and use .subckt in ngspice?",
            "a": """`.subckt` defines a reusable circuit block:

```spice
* Define a voltage regulator subcircuit
.subckt LDO_3V3 vin vout gnd
* LDO simplified model: dropout 0.3V, 100mA max
R1 vin n1 0.5
E1 n1 gnd vout gnd 1
D1 n1 vout DMOD
Rout vout gnd 1Meg
Vref ref gnd DC 3.3
E2 n2 gnd ref gnd 1
R2 n2 vout 10
.model DMOD D(IS=1e-14)
.ends LDO_3V3

* Use the subcircuit
Vin supply 0 DC 5
X1 supply output 0 LDO_3V3
Rload output 0 33

.op
.tran 1u 10m
.control
run
plot v(output) title 'LDO Output'
.endc
.end
```

Key rules:
- Pin order in `.subckt` defines the interface
- `X` prefix instantiates a subcircuit: `Xname pin1 pin2 ... subckt_name`
- Internal node names are local to the subcircuit
- Use `.include filename.lib` to load subcircuits from files
- Subcircuits can be nested"""
        },
        {
            "q": "How do I use .include and .lib to load SPICE models?",
            "a": """`.include` and `.lib` load external model files:

```spice
* .include — reads entire file
.include /path/to/models/2N2222.mod
.include ../lib/opamp_models.lib

* .lib — reads only specific section
.lib /path/to/models.lib TT    ; Load "TT" (typical) section
.lib /path/to/models.lib FF    ; Load "FF" (fast-fast) corner
```

Model file structure (`opamp_models.lib`):
```spice
* Op-Amp Model Library

.subckt LM741 noninv inv vcc vee out
* Simplified LM741 model
Rin noninv inv 2Meg
E1 mid 0 noninv inv 200k
Rout mid out 75
Vos noninv noninv_int DC 1m
.ends LM741

.subckt LM358 noninv inv vcc vee out
Rin noninv inv 1Meg
E1 mid 0 noninv inv 100k
Rout mid out 50
.ends LM358
```

Usage:
```spice
.include opamp_models.lib
X1 in+ in- vcc vee out LM741
```

Search paths in ngspice:
1. Current directory
2. Directories in SPICE_LIB_DIR environment variable
3. Absolute path"""
        },
    ]
    for s in subcircuits:
        emit(s["q"], s["a"])


def gen_param_directives():
    """Parameter and directive questions."""
    directives = [
        {
            "q": "How do I use .param for parameterized circuits in ngspice?",
            "a": """`.param` defines named parameters for reusable circuit templates:

```spice
.title Parameterized Low-Pass Filter
.param R_val = 10k
.param C_val = 10n
.param Vcc = 5

V1 in 0 AC 1 DC {Vcc/2}
R1 in out {R_val}
C1 out 0 {C_val}

.ac dec 100 1 100Meg
.control
run
let fc = 1/(2*pi*{R_val}*{C_val})
echo "Cutoff frequency:" fc "Hz"
plot vdb(out) title 'Parameterized Filter Response'

* Sweep parameter
foreach r 1k 4.7k 10k 47k 100k
    alter R1 = $r
    ac dec 100 1 100Meg
end
.endc
.end
```

Parameter expressions:
```spice
.param R1_val = 10k
.param R2_val = R1_val * 2      ; Expression
.param Gain = R2_val / R1_val    ; Dependent param
.param fc = 1/(2*3.14159*R1_val*C1_val)  ; Computed
```

Use curly braces `{}` to evaluate parameters in component values."""
        },
        {
            "q": "How do I use .measure (MEAS) in ngspice?",
            "a": """`.meas` automatically extracts measurements from simulation results:

```spice
.title Measurement Examples

V1 in 0 PULSE(0 5 0 10n 10n 5u 10u)
R1 in out 1k
C1 out 0 1n

.tran 1n 50u

* Transient measurements
.meas tran Vmax MAX v(out)
.meas tran Vmin MIN v(out)
.meas tran Vpp PP v(out)
.meas tran Vavg AVG v(out) FROM=10u TO=50u
.meas tran Vrms RMS v(out)
.meas tran Trise TRIG v(out) VAL=0.5 RISE=1 TARG v(out) VAL=4.5 RISE=1
.meas tran Tfall TRIG v(out) VAL=4.5 FALL=1 TARG v(out) VAL=0.5 FALL=1
.meas tran Tdelay TRIG v(in) VAL=2.5 RISE=1 TARG v(out) VAL=2.5 RISE=1
.meas tran V_at_10u FIND v(out) AT=10u

.control
run
echo "Results available in measurement variables"
.endc
.end
```

AC measurements:
```spice
.ac dec 100 1 100Meg
.meas ac BW TRIG vdb(out) VAL=-3 RISE=1 TARG vdb(out) VAL=-3 FALL=1
.meas ac GainDC FIND vdb(out) AT=100
.meas ac Phase_at_1k FIND vp(out) AT=1k
```

WHEN keyword:
```spice
.meas tran T50 WHEN v(out)=2.5 RISE=1
```"""
        },
    ]
    for d in directives:
        emit(d["q"], d["a"])


def gen_ltspice_vs_ngspice():
    """LTspice vs ngspice comparison."""
    comparisons = [
        {
            "q": "What are the main differences between LTspice and ngspice?",
            "a": """Key differences between LTspice and ngspice:

| Feature | LTspice | ngspice |
|---------|---------|---------|
| License | Freeware (Analog Devices) | BSD open-source |
| Platform | Windows, macOS | Windows, macOS, Linux |
| GUI | Built-in schematic editor | External (KiCad, gEDA, Xschem) |
| Scripting | Limited (.meas, .step) | Full scripting (.control/.endc) |
| Model library | Extensive (AD/LT parts) | Community models |
| XSPICE | No | Yes (code models, digital) |
| Verilog-A | Yes | Via OSDI plugin |
| Performance | Fast (optimized solver) | Good (BSIM models, KLU solver) |
| Netlisting | Internal + .asc | Standard SPICE netlist |

**Syntax differences:**

Parameter sweeping:
```spice
* LTspice:
.step param R 1k 100k 10k
* ngspice:
.control
foreach R_val 1k 10k 100k
    alter R1 = $R_val
    tran 1u 10m
end
.endc
```

Behavioral sources:
```spice
* LTspice: voltage-dependent
B1 out 0 V=if(v(in)>2.5, 5, 0)
* ngspice: same syntax works
B1 out 0 V = v(in) > 2.5 ? 5 : 0
```

Waveform math:
```spice
* LTspice: .meas and .four built-in
* ngspice: use .control scripts
.control
run
fourier 1k v(out)
.endc
```"""
        },
        {
            "q": "How do I convert an LTspice schematic to ngspice?",
            "a": """Steps to convert LTspice (.asc) to ngspice:

1. **Export netlist from LTspice**: View → SPICE Netlist (or press Ctrl+Shift+N)

2. **Fix LTspice-specific syntax**:

```spice
* LTspice uses:
.lib LTC1871.sub
* ngspice uses:
.include LTC1871.sub
```

3. **Convert .step to .control loops**:
```spice
* LTspice:
.step param R1 1k 10k 1k

* ngspice:
.control
foreach R1_val 1k 2k 3k 4k 5k 6k 7k 8k 9k 10k
    alter R1 = $R1_val
    tran 1u 10m
end
.endc
```

4. **Replace LTspice-only directives**:
```spice
* LTspice: .wave "output.wav" 16 44.1k v(out)
* ngspice: wrdata output.txt v(out)
```

5. **Model compatibility**: Most SPICE models work in both. Check for:
   - `Rser` parameter in capacitors (LTspice extension)
   - `Lser` in capacitors
   - `Ron/Roff` in switches

6. **KiCad integration**: KiCad exports ngspice-compatible netlists directly from the schematic editor."""
        },
    ]
    for c in comparisons:
        emit(c["q"], c["a"])


def gen_source_types():
    """Independent source definitions."""
    sources = [
        {
            "q": "Explain all ngspice independent voltage source waveforms (PULSE, SIN, PWL, EXP, SFFM).",
            "a": """ngspice independent voltage/current sources support these waveforms:

**1. PULSE — Square/rectangular/trapezoidal wave:**
```spice
V1 out 0 PULSE(V1 V2 TD TR TF PW PER)
* V1=initial, V2=peak, TD=delay, TR=rise, TF=fall, PW=pulse width, PER=period
V1 out 0 PULSE(0 5 0 10n 10n 500u 1m)  ; 1kHz, 50% duty, 5V
```

**2. SIN — Sinusoidal:**
```spice
V1 out 0 SIN(VO VA FREQ TD THETA PHASE)
* VO=offset, VA=amplitude, FREQ=Hz, TD=delay, THETA=damping, PHASE=degrees
V1 out 0 SIN(2.5 2.5 1k)               ; 1kHz, 0-5V sine
V1 out 0 SIN(0 1 10k 0 0 0) AC 1       ; With AC analysis magnitude
```

**3. PWL — Piecewise Linear:**
```spice
V1 out 0 PWL(T1 V1 T2 V2 T3 V3 ...)
V1 out 0 PWL(0 0 1m 5 2m 5 3m 0 4m -5 5m 0)  ; Custom waveform
```

**4. EXP — Exponential:**
```spice
V1 out 0 EXP(V1 V2 TD1 TAU1 TD2 TAU2)
V1 out 0 EXP(0 5 0 1m 5m 2m)           ; Exponential rise/fall
```

**5. SFFM — Single-Frequency FM:**
```spice
V1 out 0 SFFM(VO VA FC MDI FS)
* VO=offset, VA=amplitude, FC=carrier freq, MDI=modulation index, FS=signal freq
V1 out 0 SFFM(0 1 10k 5 100)           ; FM modulated
```

**Current sources** use the same syntax with `I` prefix:
```spice
I1 out 0 SIN(0 10m 1k)  ; 10mA peak sinusoidal current
```"""
        },
    ]
    for s in sources:
        emit(s["q"], s["a"])


def gen_xspice():
    """XSPICE/mixed-mode examples."""
    examples = [
        {
            "q": "How do I use XSPICE code models in ngspice for digital simulation?",
            "a": """XSPICE extends ngspice with digital gates, ADC/DAC bridges, and custom code models:

```spice
* Mixed analog-digital simulation
.title XSPICE Digital Example

* Analog input
V1 analog_in 0 SIN(0 2.5 1k)

* Analog-to-digital converter (1-bit comparator)
aADC [analog_in] [digital_in] adc_bridge1
.model adc_bridge1 adc_bridge(in_low=1.0 in_high=2.0)

* Digital inverter
aINV [digital_in] [digital_out] inv_gate
.model inv_gate d_inverter(rise_delay=10n fall_delay=8n)

* Digital-to-analog converter
aDAC [digital_out] [analog_out] dac_bridge1
.model dac_bridge1 dac_bridge(out_low=0.0 out_high=3.3)

Rload analog_out 0 1k

.tran 1u 5m

.control
run
plot v(analog_in) v(analog_out) title 'Mixed-Mode Simulation'
.endc
.end
```

Available XSPICE digital models:
- `d_inverter`, `d_and`, `d_nand`, `d_or`, `d_nor`, `d_xor`, `d_xnor`
- `d_buffer`, `d_tristate`, `d_pullup`, `d_pulldown`
- `d_dff` (D flip-flop), `d_jkff`, `d_tff`, `d_srlatch`
- `d_ram`, `d_rom`
- `adc_bridge`, `dac_bridge` (analog/digital converters)

XSPICE is unique to ngspice — LTspice does not support it."""
        },
    ]
    for e in examples:
        emit(e["q"], e["a"])


def gen_circuit_templates():
    """Additional complete circuit templates."""
    templates = [
        {
            "q": "Write an ngspice netlist for a Schmitt trigger using an op-amp.",
            "a": """Complete ngspice netlist for an inverting Schmitt trigger:

```spice
* Inverting Schmitt Trigger
.title Op-Amp Schmitt Trigger

Vcc vcc 0 DC 12
Vee vee 0 DC -12
Vin in 0 SIN(0 5 100)

* Op-amp (simplified comparator model)
.subckt OPAMP noninv inv vcc vee out
Rin noninv inv 10Meg
E1 mid 0 noninv inv 100k
R1 mid out 100
D1 out vcc DLIM
D2 vee out DLIM
.model DLIM D(IS=1e-14)
.ends OPAMP

R1 in inv 10k
R2 inv out 22k
X1 0 inv vcc vee out OPAMP

.tran 10u 50m

.control
run
plot v(in) v(out) title 'Schmitt Trigger'
let Vth_high = 12 * 10k / (10k + 22k)
let Vth_low = -12 * 10k / (10k + 22k)
echo "Upper threshold:" Vth_high "V"
echo "Lower threshold:" Vth_low "V"
echo "Hysteresis:" Vth_high - Vth_low "V"
.endc

.end
```

The hysteresis thresholds are:
- V_TH+ = +Vsat * R1/(R1+R2) = +12 * 10k/32k = 3.75V
- V_TH- = -Vsat * R1/(R1+R2) = -12 * 10k/32k = -3.75V
- Hysteresis = V_TH+ - V_TH- = 7.5V"""
        },
        {
            "q": "Write an ngspice netlist for a Wien bridge oscillator.",
            "a": """Complete ngspice netlist for a Wien bridge oscillator:

```spice
* Wien Bridge Oscillator
.title Wien Bridge Oscillator

Vcc vcc 0 DC 12
Vee vee 0 DC -12

* Wien bridge network (determines frequency)
.param R_wien = 10k
.param C_wien = 10n
* f_osc = 1/(2*pi*R*C) ~ 1.59 kHz

R1 out n1 {R_wien}
C1 n1 noninv {C_wien}
R2 noninv 0 {R_wien}
C2 noninv 0 {C_wien}

* Feedback network (gain = 3 at oscillation)
Rf out inv 20k
Rg inv 0 10k

* Amplitude limiting (prevents clipping)
D1 inv n_lim1 DMOD
D2 n_lim1 0 DMOD
R_lim inv 0 100k

.model DMOD D(IS=1e-14)

* Op-amp
.subckt OPAMP_FAST noninv inv vcc vee out
Rin noninv inv 1Meg
G1 0 mid noninv inv 100m
R1 mid 0 1Meg
C1 mid 0 10p
E1 out2 0 mid 0 1
Rout out2 out 100
.ends OPAMP_FAST

X1 noninv inv vcc vee out OPAMP_FAST

* Initial kick to start oscillation
Vkick noninv 0 PULSE(0 0.1 0 1n 1n 1u 0) DC 0

.tran 1u 10m UIC

.control
run
plot v(out) title 'Wien Bridge Oscillator Output'
fourier 1.59k v(out)
.endc

.end
```

Design: f = 1/(2*pi*R*C) = 1/(2*pi*10k*10n) ~ 1.59 kHz.
Gain must be exactly 3 for sustained oscillation (Rf/Rg + 1 = 3)."""
        },
        {
            "q": "Write an ngspice netlist for a voltage regulator using LM317.",
            "a": """Complete ngspice netlist for an LM317 adjustable voltage regulator:

```spice
* LM317 Voltage Regulator — 5V output
.title LM317 Regulator

.subckt LM317 in out adj
* Simplified LM317 model
* Vref = 1.25V between OUT and ADJ
Vref out adj DC 1.25
Rin in n1 1
E1 n1 0 out 0 1
Rout n1 out 0.5
Iq adj 0 DC 50u
.ends LM317

Vin supply 0 DC 12
X1 supply output adj LM317

* Output voltage: Vout = 1.25 * (1 + R2/R1) + Iq*R2
R1 output adj 240
R2 adj 0 720

* Input/output capacitors
Cin supply 0 100n
Cout output 0 10u
Cadj adj 0 10u

Rload output 0 50

.tran 10u 20m
.dc Vin 0 15 0.1

.control
tran 10u 20m
plot v(output) title 'Transient Output'
dc Vin 0 15 0.1
plot v(output) title 'Output vs Input (Dropout Test)'
echo "Vout = 1.25 * (1 + 720/240) = 5V"
.endc

.end
```

Design equations:
- Vout = 1.25V * (1 + R2/R1) + Iq*R2
- Vout = 1.25 * (1 + 720/240) + 50u*720 = 5.036V
- Dropout voltage: ~2V (Vin must be >= Vout + 2V)"""
        },
        {
            "q": "Write an ngspice netlist for a current sense amplifier.",
            "a": """Complete ngspice netlist for a high-side current sense amplifier:

```spice
* High-Side Current Sense Amplifier
.title Current Sense Amplifier

Vdd vdd 0 DC 12
Iload vdd_sense 0 DC 1

* Sense resistor (10 mohm)
Rsense vdd vdd_sense 10m

* Differential amplifier (gain = 100)
.subckt DIFF_AMP inp inn vcc out
R1 inp n_inv 100k
R2 inn n_ninv 100k
R3 n_inv out 10Meg
R4 n_ninv 0 10Meg
.subckt OPAMP2 noninv inv vcc vee out
Rin noninv inv 10Meg
E1 mid 0 noninv inv 200k
Rout mid out 75
.ends OPAMP2
X1 n_ninv n_inv vcc 0 out OPAMP2
.ends DIFF_AMP

X1 vdd vdd_sense vdd out DIFF_AMP

.dc Iload 0 5 0.1

.control
run
plot v(out) title 'Sense Voltage vs Load Current'
plot v(out)/10m/100 title 'Measured Current [A]'
echo "At 1A: Vsense = 10mV, Vout = 1V (gain=100)"
.endc

.end
```

Current sensing: Vsense = Iload * Rsense = 1A * 10mOhm = 10mV.
After amplification (gain=100): Vout = 10mV * 100 = 1V.
Scale factor: 1V/A."""
        },
    ]
    for t in templates:
        emit(t["q"], t["a"])


def gen_parametric_variations():
    """Generate parametric variations of simple circuits."""
    for _ in range(80):
        r1 = random.choice(RESISTORS)
        r2 = random.choice(RESISTORS)
        c1 = random.choice(CAPACITORS)
        vin = random.choice(VOLTAGES)
        freq = random.choice(FREQUENCIES)
        diode = random.choice(DIODES)

        circuit_type = random.choice([
            "half-wave rectifier",
            "full-wave bridge rectifier",
            "clamper circuit",
            "clipper circuit",
            "peak detector",
            "voltage doubler",
        ])

        if circuit_type == "half-wave rectifier":
            emit(
                f"Write an ngspice netlist for a half-wave rectifier with {diode}, Vin={vin}Vpk at {freq}Hz, filter C={c1}.",
                f"""```spice
* Half-Wave Rectifier
.title Half-Wave Rectifier

V1 in 0 SIN(0 {vin} {freq})
D1 in out {diode}_MOD
C1 out 0 {c1}
Rload out 0 {r1}

.model {diode}_MOD D(IS=1e-14 RS=0.5 BV=100 CJO=5p TT=5n)

.tran {{1/{freq}/100}} {{5/{freq}}}

.control
run
plot v(in) v(out) title 'Half-Wave Rectifier'
meas tran Vripple PP v(out) FROM={{3/{freq}}} TO={{5/{freq}}}
echo "Ripple voltage:" Vripple
.endc
.end
```

Output DC voltage: ~{vin}V - 0.7V (diode drop).
Ripple depends on C1={c1} and Rload={r1}."""
            )
        elif circuit_type == "full-wave bridge rectifier":
            emit(
                f"Write an ngspice netlist for a bridge rectifier with {diode} diodes, {vin}Vpk at {freq}Hz.",
                f"""```spice
* Full-Wave Bridge Rectifier
.title Bridge Rectifier

V1 in_p in_n SIN(0 {vin} {freq})

D1 in_p out_p {diode}_MOD
D2 in_n out_p {diode}_MOD
D3 out_n in_p {diode}_MOD
D4 out_n in_n {diode}_MOD

C1 out_p out_n {c1}
Rload out_p out_n {r1}

.model {diode}_MOD D(IS=1e-14 RS=0.5 BV=100)

.tran {{1/{freq}/100}} {{5/{freq}}}

.control
run
plot v(out_p,out_n) v(in_p,in_n) title 'Bridge Rectifier'
.endc
.end
```

DC output: ~{vin}V - 1.4V (two diode drops).
Full-wave has half the ripple of half-wave for the same filter cap."""
            )
        else:
            emit(
                f"Write an ngspice netlist for a {circuit_type} using {diode}.",
                f"""```spice
* {circuit_type.title()} Circuit
.title {circuit_type.title()}

V1 in 0 SIN(0 {vin} {freq})
D1 in out {diode}_MOD
R1 out 0 {r1}
C1 out 0 {c1}

.model {diode}_MOD D(IS=1e-14 RS=0.5 BV=100)

.tran {{1/{freq}/100}} {{10/{freq}}}

.control
run
plot v(in) v(out) title '{circuit_type.title()}'
.endc
.end
```

This {circuit_type} uses {diode} with Vin={vin}Vpk at {freq}Hz.
R1={r1} and C1={c1} set the time constant."""
            )


def gen_misc_topics():
    """Miscellaneous SPICE topics."""
    misc = [
        {
            "q": "How do I perform Fourier (FFT) analysis in ngspice?",
            "a": """Use the `.four` directive or `spec` command for FFT analysis:

```spice
.title FFT Analysis
V1 in 0 SIN(0 1 1k)
R1 in out 1k
C1 out 0 10n

.tran 1u 10m

* Built-in Fourier analysis at fundamental frequency
.four 1k v(out)

.control
run
* FFT using spec command
linearize v(out)
spec 0 100k 100 v(out)
plot mag(v(out)) title 'FFT Magnitude'
plot ph(v(out)) title 'FFT Phase'

* Or use fft directly
fft v(out)
plot mag(v(out)) title 'FFT Result'
.endc
.end
```

`.four freq node` computes THD and harmonics at the specified fundamental frequency.
`spec` computes a windowed FFT with specified frequency range and number of points.
`fft` computes a raw FFT of the time-domain data."""
        },
        {
            "q": "How do I simulate temperature effects in ngspice?",
            "a": """ngspice models temperature-dependent behavior:

```spice
.title Temperature Sweep
V1 in 0 DC 0
D1 in 0 DMOD
R1 in anode 100
Vd anode 0 DC 5

.model DMOD D(IS=1e-14 N=1 EG=1.11 XTI=3)

* Single temperature
.options TEMP=27

* Temperature sweep using .control
.control
let temp_list = ( -40 -20 0 25 50 75 100 125 )
foreach t $&temp_list
    set temp = $t
    op
    print v(in)
    echo "Temperature: $t C, Vf = " v(in)
end

* Or use .temp directive
.endc
.end
```

Temperature effects modeled:
- **Diode forward voltage**: decreases ~2mV/°C
- **BJT Vbe**: decreases ~2mV/°C
- **BJT beta**: increases with temperature
- **Resistor**: R(T) = R(Tnom) * (1 + TC1*(T-Tnom) + TC2*(T-Tnom)^2)
- **MOSFET threshold**: decreases with temperature

```spice
* Temperature-dependent resistor
R1 n1 n2 10k TC=0.004, 0.0001  ; TC1=0.4%/°C, TC2
```"""
        },
        {
            "q": "How do I export ngspice simulation data to a file?",
            "a": """Several methods to export simulation data:

```spice
.control
run

* Method 1: wrdata (space-separated columns)
wrdata output.txt v(out) v(in) i(V1)

* Method 2: write (raw binary format, for later loading)
write output.raw v(out) v(in)

* Method 3: print to file
set wr_singlescale
set wr_vecnames
option numdgt=7
wrdata results.csv v(out) v(in)

* Method 4: redirect print output
echo "time,v_out,v_in" > data.csv
print v(out) v(in) >> data.csv

* Method 5: ASCII raw file
set filetype=ascii
write output.raw v(out) v(in)

* Load data back
load output.raw
.endc
```

For Python post-processing:
```python
import numpy as np
# Read wrdata output
data = np.loadtxt('output.txt')
time = data[:, 0]
v_out = data[:, 1]
```

For MATLAB/Octave:
```matlab
data = load('output.txt');
plot(data(:,1), data(:,2));
```"""
        },
        {
            "q": "How do I simulate a transmission line in ngspice?",
            "a": """ngspice supports lossless and lossy transmission lines:

**Lossless transmission line (T element):**
```spice
.title Transmission Line Simulation
* Lossless line: Z0=50 ohm, delay=1ns

Vs source 0 PULSE(0 1 0 100p 100p 5n 20n)
Rs source in 50

T1 in 0 out 0 Z0=50 TD=1ns

Rload out 0 50

.tran 10p 20n

.control
run
plot v(in) v(out) title 'Matched Transmission Line'
.endc
.end
```

**Lossy transmission line (LTRA):**
```spice
* Lossy line with R, L, G, C per unit length
O1 in 0 out 0 LOSSYMOD
.model LOSSYMOD LTRA(R=5 L=250n G=0 C=100p LEN=0.5)
```

**Coupled transmission lines:**
```spice
* Coupled microstrip pair
U1 in1 0 out1 0 in2 0 out2 0 CPLMOD
.model CPLMOD CPL length=0.05
+ R=0 0 0 0
+ L=3.38e-7 5.09e-8 5.09e-8 3.38e-7
+ G=0 0 0 0
+ C=1.21e-10 -1.99e-11 -1.99e-11 1.21e-10
```

For impedance-matched termination (no reflections), Rload = Z0.
For open circuit (Rload=inf), the reflected wave doubles the voltage."""
        },
    ]
    for m in misc:
        emit(m["q"], m["a"])


def gen_power_supply_circuits():
    """Additional power supply circuits."""
    for _ in range(40):
        vin = random.choice(["12", "24", "48"])
        vout = random.choice(["3.3", "5", "9"])
        ripple_freq = random.choice(["100", "120"])

        emit(
            f"Write an ngspice netlist for a linear power supply: {vin}VAC to {vout}VDC.",
            f"""```spice
* Linear Power Supply: {vin}VAC → {vout}VDC
.title Linear Power Supply

* Transformer secondary (simplified)
V1 sec_p sec_n SIN(0 {float(vin)*1.414:.1f} 50)

* Bridge rectifier
D1 sec_p rect_p DMOD
D2 sec_n rect_p DMOD
D3 rect_n sec_p DMOD
D4 rect_n sec_n DMOD
.model DMOD D(IS=1e-14 RS=0.3 BV=100)

* Filter
C1 rect_p rect_n 1000u
C2 rect_p rect_n 100n

* Voltage regulator (simplified)
.subckt REG in out gnd
Vref out gnd DC {vout}
Rin in out 1
.ends REG

X1 rect_p reg_out rect_n REG
C3 reg_out rect_n 10u
Rload reg_out rect_n {{({vout}*{vout})/5}}

.tran 100u 100m

.control
run
plot v(sec_p,sec_n) title 'Transformer Secondary'
plot v(rect_p,rect_n) title 'Rectified (before regulator)'
plot v(reg_out,rect_n) title 'Regulated Output'
.endc
.end
```

Stages: AC → transformer → bridge rectifier → capacitor filter → linear regulator.
Unregulated DC: ~{float(vin)*1.414 - 1.4:.1f}V peak ({vin}V * 1.414 - 1.4V diode drops).
Regulated output: {vout}V DC."""
        )


def gen_spice_model_syntax():
    """SPICE model syntax and parameters."""
    models = [
        {
            "q": "Explain the SPICE .model syntax for diodes.",
            "a": """The SPICE diode model:

```spice
.model DNAME D(param1=val1 param2=val2 ...)
```

Key parameters:
| Parameter | Description | Default | Unit |
|-----------|-------------|---------|------|
| IS | Saturation current | 1e-14 | A |
| N | Emission coefficient | 1 | - |
| RS | Series resistance | 0 | Ohm |
| BV | Reverse breakdown voltage | inf | V |
| IBV | Current at breakdown | 1e-3 | A |
| CJO | Zero-bias junction capacitance | 0 | F |
| VJ | Junction potential | 1 | V |
| M | Grading coefficient | 0.5 | - |
| TT | Transit time | 0 | s |
| EG | Bandgap energy | 1.11 | eV |
| XTI | IS temperature exponent | 3 | - |

Example models:
```spice
* Signal diode
.model 1N4148 D(IS=2.52e-9 RS=0.568 N=1.752 BV=100 IBV=100u CJO=4p VJ=0.7 M=0.4 TT=6n)

* Power rectifier
.model 1N4007 D(IS=7.02e-9 RS=0.0341 N=1.8 BV=1000 IBV=5u CJO=18p VJ=0.7 M=0.38 TT=4u)

* Schottky
.model 1N5819 D(IS=3.15e-8 RS=0.042 N=1.06 BV=40 IBV=1m CJO=110p VJ=0.34 M=0.44 TT=5n)

* Zener (5.1V)
.model BZX55C5V1 D(IS=1e-14 RS=10 BV=5.1 IBV=5m CJO=50p)
```"""
        },
        {
            "q": "Explain the SPICE .model syntax for MOSFETs.",
            "a": """SPICE MOSFET model levels:

**Level 1 (Shichman-Hodges, basic):**
```spice
.model NMOS1 NMOS(LEVEL=1 VTO=1.5 KP=110u GAMMA=0.4 PHI=0.65
+ LAMBDA=0.04 CBD=10p CBS=10p CGSO=2p CGDO=2p)
```

| Parameter | Description | Unit |
|-----------|-------------|------|
| VTO | Threshold voltage | V |
| KP | Transconductance | A/V^2 |
| GAMMA | Body effect | V^0.5 |
| LAMBDA | Channel-length modulation | 1/V |
| CBD/CBS | Drain/Source junction cap | F |
| CGSO/CGDO | Gate overlap cap | F/m |

**Level 3 (semi-empirical, better for short channels):**
```spice
.model NMOS3 NMOS(LEVEL=3 VTO=0.7 UO=600 TOX=10n
+ THETA=0.1 ETA=0.05 KAPPA=0.2)
```

**BSIM models (Level 49/54, industry standard):**
```spice
.model NMOS_BSIM NMOS(LEVEL=54 VERSION=4.5
+ TNOM=27 TOXE=1.8e-9 TOXP=1.5e-9
+ VTH0=0.4 K1=0.5 K2=-0.1
+ VSAT=1.5e5 UA=2e-9 UB=2e-18
+ RDSW=200 PCLM=1.3)
```

For power MOSFETs, use Level 1 with appropriate parameters:
```spice
.model IRF540N NMOS(LEVEL=1 VTO=4 KP=20 RDS=0.077
+ CBD=1.5n CGS=1n CGD=0.5n)
```"""
        },
    ]
    for m in models:
        emit(m["q"], m["a"])


def gen_extra_spice():
    """Extra Q&A to reach target count."""
    topics = [
        ("How do I simulate a phase-locked loop (PLL) in ngspice?",
         """A simplified PLL simulation in ngspice:

```spice
* Phase-Locked Loop (PLL) — Behavioral Model
.title PLL Simulation

* Reference signal
Vref ref 0 PULSE(0 5 0 1n 1n 50u 100u)

* Phase detector (XOR type)
B_pd pd 0 V = (v(ref) > 2.5) != (v(vco_out) > 2.5) ? 5 : 0

* Loop filter (low-pass)
R1 pd lpf 10k
C1 lpf 0 100n
C2 lpf 0 1n

* VCO (voltage-controlled oscillator)
B_vco vco_out 0 V = 5 * sin(2*3.14159*(10k + 1k*v(lpf))*time)

.tran 1u 10m UIC

.control
run
plot v(ref) v(vco_out) title 'PLL Lock'
plot v(lpf) title 'Loop Filter (Control Voltage)'
.endc
.end
```

PLL components: Phase Detector → Loop Filter → VCO → feedback.
The VCO frequency tracks the reference frequency when locked."""),

        ("How do I simulate a crystal oscillator in ngspice?",
         """Crystal oscillator (Colpitts topology):

```spice
* Crystal Oscillator — 10 MHz
.title Crystal Oscillator

Vcc vcc 0 DC 5

* Crystal equivalent circuit (10 MHz)
.subckt XTAL p1 p2
Lm p1 n1 10.2m
Cm n1 n2 25f
Rm n2 p2 30
Cp p1 p2 5p
.ends XTAL

* Colpitts oscillator
X1 collector base XTAL
C1 base 0 33p
C2 collector 0 68p
Rb1 vcc base 100k
Rb2 base 0 47k
Rc vcc collector 1k
Re emitter 0 470
Ce emitter 0 100u
Q1 collector base emitter 2N3904

.model 2N3904 NPN(BF=300 IS=6.7e-15 CJC=3.6p CJE=4.5p TF=0.35n)

* Kick to start oscillation
Ic1 collector 0 PULSE(0 1m 0 1n 1n 10n 0)

.tran 10n 500u UIC

.control
run
plot v(collector) title 'Crystal Oscillator Output'
fourier 10Meg v(collector)
.endc
.end
```

Crystal parameters: Lm=10.2mH, Cm=25fF, Rm=30ohm, Cp=5pF model a 10 MHz quartz crystal.
Series resonance: fs = 1/(2*pi*sqrt(Lm*Cm))."""),

        ("How do I use .step equivalent in ngspice to sweep a component value?",
         """ngspice uses `.control` loops instead of LTspice's `.step`:

```spice
.title Component Sweep (ngspice)

V1 in 0 AC 1
R1 in out 10k
C1 out 0 10n

.ac dec 100 1 10Meg

.control
* Sweep resistor value
foreach R_val 1k 2.2k 4.7k 10k 22k 47k 100k
    alter R1 = $R_val
    ac dec 100 1 10Meg
end
plot db(v(out)) title 'R Sweep: Frequency Response'

* Sweep with let/while loop
let C_vals = ( 1n 10n 100n 1u )
let idx = 0
while idx < length(C_vals)
    alter C1 = C_vals[idx]
    ac dec 100 1 10Meg
    let idx = idx + 1
end
plot db(v(out)) title 'C Sweep: Frequency Response'
.endc
.end
```

The `alter` command changes a component value between simulation runs.
Each `ac` (or `tran`, `dc`) command creates a new plot."""),

        ("How do I create a .SPICE model for a new component?",
         """Creating custom SPICE models:

**1. Diode from datasheet:**
```spice
* Extract IS and N from Vf at two currents
* If=1mA → Vf=0.6V, If=100mA → Vf=0.85V
* N = (V2-V1) / (Vt * ln(I2/I1)) = 0.25/(0.026*4.6) = 2.09
* IS = I / exp(V/(N*Vt)) = 1m / exp(0.6/(2.09*0.026)) = 1.2e-9

.model MY_DIODE D(IS=1.2e-9 N=2.09 RS=0.5 CJO=10p VJ=0.7 BV=50 TT=5n)
```

**2. BJT from datasheet:**
```spice
* hFE=200 at Ic=10mA → BF=200
* Vbe=0.7V at Ic=10mA → IS=10m/exp(0.7/0.026)=2.7e-14
* Early voltage VA=100V (from output characteristics slope)

.model MY_BJT NPN(BF=200 IS=2.7e-14 VAF=100
+ CJC=5p CJE=10p TF=0.3n TR=30n
+ RB=10 RC=1 RE=0.5)
```

**3. Power MOSFET from datasheet:**
```spice
* Vth=3V, Rds(on)=0.1 ohm, Ids=20A
* KP = 2*Ids/(Vgs-Vth)^2 = 2*20/(10-3)^2 = 0.82

.model MY_MOSFET NMOS(LEVEL=1 VTO=3 KP=0.82
+ RDS=0.1 CBD=500p CGS=1n CGD=200p
+ IS=1e-14 RD=0.01 RS=0.01)
```

Validate by simulating known datasheet curves (I-V, transfer characteristics)."""),
    ]

    for q, a in topics:
        emit(q, a)

    # Generate more parametric variations
    for _ in range(200):
        circuit = random.choice([
            "RC low-pass", "RC high-pass", "RL low-pass", "RL high-pass",
            "voltage divider", "current mirror", "differential pair",
            "cascode amplifier", "push-pull output stage", "bootstrap circuit",
        ])
        r = random.choice(RESISTORS)
        c = random.choice(CAPACITORS)
        l = random.choice(INDUCTORS)
        v = random.choice(VOLTAGES)
        bjt = random.choice(BJTS)
        mosfet = random.choice(MOSFETS)

        emit(
            f"Write an ngspice netlist for a {circuit} with {'R=' + r + ', C=' + c if 'RC' in circuit else 'R=' + r + ', L=' + l + 'H' if 'RL' in circuit else 'Vcc=' + v + 'V'}.",
            f"""```spice
* {circuit.title()} Circuit
.title {circuit.title()}

{"V1 in 0 AC 1 SIN(0 1 1k)" if "pass" in circuit or "divider" in circuit else "Vcc vcc 0 DC " + v}
{"R1 in out " + r if "low" in circuit and "RC" in circuit else "R1 in n1 " + r}
{"C1 out 0 " + c if "RC" in circuit else "L1 n1 out " + l if "RL" in circuit else ""}
{"" if "pass" in circuit else "R2 out 0 " + random.choice(RESISTORS) if "divider" in circuit else "Q1 out n1 0 " + bjt if "mirror" in circuit or "pair" in circuit or "cascode" in circuit or "push" in circuit else ""}

{".model " + bjt + " NPN(BF=200 IS=1e-14 VAF=100)" if bjt in circuit or "mirror" in circuit or "pair" in circuit or "cascode" in circuit or "push" in circuit else ""}

.{"ac dec 100 1 100Meg" if "pass" in circuit else "tran 10u 10m" if "divider" not in circuit else "dc V1 0 " + v + " 0.1"}

.control
run
plot {"vdb(out) title 'Frequency Response'" if "pass" in circuit else "v(out) title '" + circuit.title() + " Response'"}
.endc
.end
```

{circuit.title()} circuit with {"R=" + r + ", C=" + c if "RC" in circuit else "R=" + r + ", L=" + l if "RL" in circuit else "supply=" + v + "V"}.
{"Cutoff frequency: fc = 1/(2*pi*R*C)" if "RC" in circuit else "Cutoff frequency: fc = R/(2*pi*L)" if "RL" in circuit else ""}"""
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gen_voltage_divider()
    gen_rc_filters()
    gen_rlc_filters()
    gen_opamp_circuits()
    gen_bjt_amplifier()
    gen_mosfet_circuits()
    gen_buck_converter()
    gen_hbridge()
    gen_analysis_types()
    gen_behavioral_sources()
    gen_convergence()
    gen_subcircuit()
    gen_param_directives()
    gen_ltspice_vs_ngspice()
    gen_source_types()
    gen_xspice()
    gen_circuit_templates()
    gen_parametric_variations()
    gen_misc_topics()
    gen_power_supply_circuits()
    gen_spice_model_syntax()
    gen_extra_spice()


if __name__ == "__main__":
    main()
