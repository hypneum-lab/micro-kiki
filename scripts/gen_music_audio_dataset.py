#!/usr/bin/env python3
"""Generate music/audio DSP training dataset.

Covers: MIDI protocol, audio DSP concepts, synthesis types, audio effects,
frameworks (JUCE, SuperCollider, Csound, Faust, PureData, Web Audio API),
audio file formats, real-time audio, music theory for programmers.

Target: ~1500 Q&A pairs.
Output: JSONL to stdout.

Usage::

    uv run python scripts/gen_music_audio_dataset.py > /tmp/gen-music-audio.jsonl
"""
from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def msg(user: str, assistant: str) -> dict:
    """Format a single Q&A pair as chat messages dict."""
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


# ---------------------------------------------------------------------------
# 1. MIDI Protocol
# ---------------------------------------------------------------------------

def gen_midi_protocol() -> list[dict]:
    pairs = []

    # Note On/Off
    pairs.append(msg(
        "What is a MIDI Note On message and what bytes does it consist of?",
        "A MIDI Note On message tells a synthesizer to start playing a note. It consists of 3 bytes:\n\n1. **Status byte**: `0x9n` where `n` is the channel (0-15), so `0x90` = Note On channel 1\n2. **Note number**: 0-127 (60 = Middle C, C4)\n3. **Velocity**: 1-127 (0 velocity is treated as Note Off by convention)\n\nExample: `0x90 0x3C 0x7F` = Note On, channel 1, Middle C (60), velocity 127 (max)."
    ))
    pairs.append(msg(
        "What is the difference between MIDI Note Off and Note On with velocity 0?",
        "Functionally they are equivalent — both silence a note. However:\n\n- **Note Off** (`0x8n`): Explicit 3-byte message with a release velocity (rarely used)\n- **Note On velocity 0** (`0x9n`, vel=0): Convention used because MIDI running status optimization works better when all note messages share the same status byte `0x9n`, reducing bandwidth\n\nMost modern MIDI implementations treat velocity-0 Note On as Note Off. The release velocity in true Note Off can control fade-out behavior on some synthesizers."
    ))
    pairs.append(msg(
        "How does MIDI running status work and why is it useful?",
        "Running status is a MIDI optimization where the status byte is omitted when consecutive messages share the same status.\n\n**How it works:**\n- First message: send full 3 bytes (status + data1 + data2)\n- Subsequent messages with same status: send only 2 bytes (data1 + data2)\n- The receiver remembers the last status byte\n\n**Example** — three Note On messages on channel 1:\n```\n0x90 0x3C 0x7F   # full message\n     0x3E 0x7F   # running status, saves 1 byte\n     0x40 0x7F   # running status\n```\n\n**Why useful:** Reduces MIDI bandwidth by ~33% for streams of same-type messages (e.g., rapid notes, CC automation). Critical for the original 31.25 kbaud MIDI serial link."
    ))

    # Control Change
    pairs.append(msg(
        "What is a MIDI Control Change (CC) message? List common CC numbers.",
        "A Control Change message (`0xBn`) adjusts a continuous parameter on a MIDI channel. Format: status + CC number (0-127) + value (0-127).\n\n**Common CC numbers:**\n| CC | Name | Usage |\n|----|------|-------|\n| 1 | Modulation Wheel | Vibrato, filter sweep |\n| 7 | Volume | Channel volume |\n| 10 | Pan | Stereo position |\n| 11 | Expression | Dynamic volume within a phrase |\n| 64 | Sustain Pedal | On (≥64) / Off (<64) |\n| 71 | Resonance | Filter Q |\n| 74 | Brightness | Filter cutoff |\n| 91 | Reverb Send | Effect level |\n| 93 | Chorus Send | Effect level |\n| 120 | All Sound Off | Emergency silence |\n| 123 | All Notes Off | Release all held notes |\n\nCC 0 and 32 are reserved for Bank Select (MSB/LSB) to switch between sound banks."
    ))
    pairs.append(msg(
        "What is the difference between MIDI CC 7 (Volume) and CC 11 (Expression)?",
        "Both control loudness but serve different roles:\n\n- **CC 7 (Volume)**: Sets the overall channel volume level. Think of it as the \"fader\" on a mixing console. Typically set once per track and rarely automated during performance.\n\n- **CC 11 (Expression)**: A sub-volume control for dynamic phrasing within a passage. It scales relative to the CC 7 level. Used for crescendos, decrescendos, and breath-like dynamics.\n\n**Relationship:** Final volume = CC7 × CC11 × velocity (roughly). A CC7 of 100 with CC11 of 64 gives about half the CC7 level. This two-stage design lets you set a mix balance (CC7) while still having per-phrase dynamics (CC11)."
    ))

    # Program Change
    pairs.append(msg(
        "How does MIDI Program Change work? What about Bank Select?",
        "**Program Change** (`0xCn`) selects a sound/patch on a channel. It's a 2-byte message: status + program number (0-127), giving 128 possible sounds per bank.\n\n**Bank Select** extends this to 16,384 banks:\n1. Send CC 0 (Bank Select MSB) with bank high byte\n2. Send CC 32 (Bank Select LSB) with bank low byte\n3. Send Program Change with program number\n\n**Example** — select program 5 in bank 2:\n```\n0xB0 0x00 0x00   # CC0 = 0 (MSB)\n0xB0 0x20 0x02   # CC32 = 2 (LSB)\n0xC0 0x05        # Program Change = 5\n```\n\n**General MIDI** (GM) defines a standard mapping: program 0 = Acoustic Grand Piano, program 24 = Nylon Guitar, program 73 = Flute, etc."
    ))

    # SysEx
    pairs.append(msg(
        "What are MIDI System Exclusive (SysEx) messages?",
        "SysEx messages are manufacturer-specific MIDI messages for device-specific parameters that standard MIDI doesn't cover.\n\n**Format:**\n```\n0xF0              # SysEx start\n<manufacturer ID>  # 1 or 3 bytes (e.g., 0x43 = Yamaha, 0x41 = Roland)\n<device ID>        # target device\n<data bytes>       # payload (all < 0x80, i.e., 7-bit values)\n0xF7              # SysEx end (EOX)\n```\n\n**Common uses:**\n- Bulk patch dump/restore\n- Firmware updates\n- Deep parameter editing (filter type, oscillator settings)\n- Universal SysEx (0x7E/0x7F): GM System On, Master Volume, tuning\n\n**Example** — GM System On:\n```\nF0 7E 7F 09 01 F7\n```\n\nSysEx can be arbitrarily long (kilobytes for patch dumps). Real-time messages can be interleaved within SysEx."
    ))

    # MPE
    pairs.append(msg(
        "What is MIDI Polyphonic Expression (MPE) and how does it work?",
        "MPE (MIDI Polyphonic Expression) enables per-note expression by assigning each note its own MIDI channel.\n\n**Problem MPE solves:** Standard MIDI applies pitch bend and CC to all notes on a channel simultaneously. You can't bend one note while holding another steady.\n\n**How MPE works:**\n1. A **Manager Channel** (ch 1 or 16) sends global controls\n2. **Member Channels** (ch 2-15) each handle one active note\n3. Each note gets independent: pitch bend, pressure (aftertouch), CC 74 (slide/timbre)\n\n**Zone configuration:**\n- Lower Zone: Manager = ch 1, Members = ch 2 upward\n- Upper Zone: Manager = ch 16, Members = ch 15 downward\n- Configured via MCM (MPE Configuration Message): CC 6 on the manager channel\n\n**Controllers used for MPE:**\n- Pitch Bend: per-note pitch (±48 semitones typical)\n- Channel Pressure: per-note pressure\n- CC 74: per-note slide (Y-axis on controllers like Roli Seaboard, Linnstrument)\n\nMPE is part of the MIDI 1.0 spec (adopted 2018) and natively supported in MIDI 2.0."
    ))

    # MIDI 2.0
    pairs.append(msg(
        "What are the key improvements in MIDI 2.0 over MIDI 1.0?",
        "MIDI 2.0 (ratified 2020) brings major improvements:\n\n**1. Higher Resolution:**\n- Velocity: 16-bit (65,536 levels) vs 7-bit (128)\n- Controllers: 32-bit values vs 7-bit\n- Pitch Bend: 32-bit vs 14-bit\n\n**2. Universal MIDI Packet (UMP):**\n- Fixed-size 32/64/96/128-bit packets (not variable-length byte streams)\n- Transport-agnostic (USB, network, Bluetooth, serial)\n\n**3. Per-Note Controllers:**\n- Built-in per-note pitch bend, CC, management — supersedes MPE\n- Each note can have independent expression without channel tricks\n\n**4. Bidirectional Protocol Negotiation:**\n- MIDI-CI (Capability Inquiry): devices negotiate features\n- Profile Configuration: standard behavior profiles (e.g., \"drawbar organ\")\n- Property Exchange: JSON-based parameter read/write\n\n**5. Backward Compatibility:**\n- MIDI 2.0 devices must also support MIDI 1.0\n- UMP can encapsulate MIDI 1.0 messages\n\n**6. Jitter Reduction Timestamps:**\n- Sub-millisecond timing for tight sync"
    ))

    # Channels and ports
    pairs.append(msg(
        "How many channels does MIDI support and what is the significance of channel 10?",
        "MIDI supports **16 channels** per port (cable/connection).\n\n**Channel 10** is special in General MIDI (GM): it's reserved for **percussion/drums**. Each note number triggers a different drum sound:\n- 35/36: Kick drum\n- 38: Snare\n- 42: Closed hi-hat\n- 46: Open hi-hat\n- 49: Crash cymbal\n- 51: Ride cymbal\n\nThis convention means any GM-compatible synth will play drums on channel 10 without manual patch selection.\n\n**Multiple ports:** To get more than 16 channels, use multiple MIDI ports. USB MIDI supports up to 16 virtual cables × 16 channels = 256 channels. In DAWs, each virtual instrument typically gets its own port."
    ))

    # Clock and sync
    pairs.append(msg(
        "How does MIDI Clock synchronization work?",
        "MIDI Clock provides tempo synchronization between devices using these real-time messages:\n\n**Messages:**\n- `0xF8` — **Timing Clock**: sent 24 times per quarter note (24 PPQN)\n- `0xFA` — **Start**: begin playback from the beginning\n- `0xFB` — **Continue**: resume from current position\n- `0xFC` — **Stop**: stop playback\n- `0xF2` — **Song Position Pointer**: 14-bit value indicating position in MIDI beats (1 beat = 6 clocks)\n\n**Calculating BPM from clock:**\n```\nBPM = 60,000,000 / (average_microseconds_between_clocks × 24)\n```\n\nAt 120 BPM: clock interval = 60/(120×24) = 20.83ms\n\n**Limitations:**\n- No sub-beat resolution (only 24 PPQN)\n- Jitter-sensitive on slow serial MIDI\n- No absolute time reference\n\n**Alternative:** MIDI Time Code (MTC) provides SMPTE-style absolute time for film/video sync."
    ))

    pairs.append(msg(
        "Write a Python function that parses raw MIDI bytes into human-readable messages.",
        """```python
def parse_midi_message(data: bytes) -> str:
    \"\"\"Parse raw MIDI bytes into a human-readable string.\"\"\"
    if not data:
        return "Empty message"

    status = data[0]
    msg_type = status & 0xF0
    channel = (status & 0x0F) + 1  # 1-based

    if msg_type == 0x80:
        note, vel = data[1], data[2]
        return f"Note Off  ch={channel} note={note} vel={vel}"
    elif msg_type == 0x90:
        note, vel = data[1], data[2]
        if vel == 0:
            return f"Note Off  ch={channel} note={note} (vel=0)"
        return f"Note On   ch={channel} note={note} vel={vel}"
    elif msg_type == 0xA0:
        return f"Poly Pressure ch={channel} note={data[1]} pressure={data[2]}"
    elif msg_type == 0xB0:
        return f"CC        ch={channel} cc={data[1]} val={data[2]}"
    elif msg_type == 0xC0:
        return f"Program Change ch={channel} program={data[1]}"
    elif msg_type == 0xD0:
        return f"Ch Pressure ch={channel} pressure={data[1]}"
    elif msg_type == 0xE0:
        bend = (data[2] << 7) | data[1]  # 14-bit, center=8192
        return f"Pitch Bend ch={channel} value={bend} (center=8192)"
    elif status == 0xF0:
        return f"SysEx ({len(data)} bytes)"
    elif status == 0xF8:
        return "Timing Clock"
    elif status == 0xFA:
        return "Start"
    elif status == 0xFB:
        return "Continue"
    elif status == 0xFC:
        return "Stop"
    else:
        return f"Unknown status=0x{status:02X}"
```

This parser handles all Channel Voice messages, plus common System Real-Time messages. For SysEx, it reports the length — full parsing requires manufacturer-specific knowledge."""
    ))

    # MIDI file format
    pairs.append(msg(
        "What is the structure of a Standard MIDI File (SMF)?",
        "A Standard MIDI File has two types of chunks:\n\n**1. Header Chunk (MThd):**\n```\n4D 54 68 64  # \"MThd\" magic\n00 00 00 06  # chunk length (always 6)\nFF FF        # format: 0 (single track), 1 (multi-track sync), 2 (multi-track async)\nNN NN        # number of tracks\nDD DD        # division (ticks per quarter note, typically 480 or 960)\n```\n\n**2. Track Chunks (MTrk):**\n```\n4D 54 72 6B  # \"MTrk\" magic\nLL LL LL LL  # chunk length in bytes\n<events>     # delta-time + MIDI events\n```\n\n**Delta time** uses variable-length encoding (1-4 bytes, MSB continuation bit). It represents ticks since the previous event.\n\n**Meta events** (0xFF) store non-MIDI data:\n- `FF 51 03`: Set Tempo (microseconds per quarter note)\n- `FF 58 04`: Time Signature\n- `FF 59 02`: Key Signature\n- `FF 03 len`: Track Name\n- `FF 2F 00`: End of Track (required)\n\n**Format 1** is most common: track 1 = tempo map, tracks 2+ = instrument parts."
    ))

    # Pitch bend range
    pairs.append(msg(
        "How is MIDI Pitch Bend encoded and what is the typical range?",
        "Pitch Bend is a 14-bit value transmitted as two 7-bit bytes:\n\n**Message format:** `0xEn LSB MSB`\n- Value range: 0 to 16383\n- Center (no bend): 8192 (0x2000)\n- Full down: 0\n- Full up: 16383\n\n**Calculation:**\n```python\n# Encode pitch bend\ndef encode_pitch_bend(semitones: float, range_semitones: int = 2) -> tuple[int, int]:\n    normalized = semitones / range_semitones  # -1.0 to 1.0\n    value = int(8192 + normalized * 8191)\n    value = max(0, min(16383, value))\n    return value & 0x7F, (value >> 7) & 0x7F  # LSB, MSB\n```\n\n**Default range:** ±2 semitones (GM standard). Changed via RPN 0 (Pitch Bend Sensitivity):\n```\nB0 65 00  # RPN MSB = 0\nB0 64 00  # RPN LSB = 0\nB0 06 0C  # Data Entry MSB = 12 (±12 semitones = 1 octave)\nB0 26 00  # Data Entry LSB = 0 (no fine tuning)\n```\n\nMPE typically uses ±48 semitones for per-note pitch slides."
    ))

    # MIDI over USB, Bluetooth, Network
    pairs.append(msg(
        "What are the different MIDI transport layers (DIN, USB, Bluetooth, Network)?",
        "MIDI can travel over several physical transports:\n\n**DIN-5 (Classic MIDI):**\n- 31.25 kbaud serial, optoisolated\n- 5-pin DIN connector (only 3 pins used)\n- ~1ms latency per 3-byte message\n- One-directional per cable (IN/OUT/THRU)\n\n**USB MIDI:**\n- USB bulk transfer, 64-byte packets\n- 4-byte event packets: cable number + MIDI bytes\n- Up to 16 virtual cables (256 channels)\n- Sub-millisecond latency, bus-powered\n- Class-compliant = driverless on macOS/Linux/iOS\n\n**Bluetooth MIDI (BLE-MIDI):**\n- BLE GATT service (UUID: 03B80E5A-...)\n- ~10-15ms latency (connection interval dependent)\n- Timestamps for jitter compensation\n- Range: ~10m, no pairing required on some devices\n\n**Network MIDI (RTP-MIDI / Apple MIDI):**\n- UDP/IP, supports LAN and WAN\n- Built into macOS (Audio MIDI Setup → Network)\n- RTP timestamps for jitter recovery\n- Session management via AppleMIDI protocol\n- Linux: `raveloxmidi`, Windows: `rtpMIDI`\n\n**MIDI 2.0 transports:** UMP (Universal MIDI Packet) is transport-agnostic — works over USB, network, or future transports."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 2. Audio DSP Concepts
# ---------------------------------------------------------------------------

def gen_audio_dsp() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "Explain sample rate, bit depth, and the Nyquist theorem in digital audio.",
        "**Sample Rate** is how many times per second the analog signal is measured (sampled). Common rates:\n- 44,100 Hz (CD quality)\n- 48,000 Hz (video/broadcast standard)\n- 96,000 Hz / 192,000 Hz (high-resolution audio)\n\n**Bit Depth** determines the precision of each sample — the number of possible amplitude levels:\n- 16-bit: 65,536 levels, ~96 dB dynamic range\n- 24-bit: 16,777,216 levels, ~144 dB dynamic range\n- 32-bit float: enormous range, used internally in DAWs\n\n**Nyquist Theorem:** To accurately capture a frequency `f`, the sample rate must be at least `2f`. The maximum representable frequency (Nyquist frequency) is `sample_rate / 2`.\n- At 44.1 kHz: Nyquist = 22,050 Hz (just above human hearing ~20 kHz)\n- At 48 kHz: Nyquist = 24,000 Hz\n\nIf a signal contains frequencies above the Nyquist frequency, **aliasing** occurs — those frequencies fold back as false lower frequencies. Anti-aliasing filters (low-pass) are applied before sampling to prevent this."
    ))

    pairs.append(msg(
        "What is aliasing in digital audio and how is it prevented?",
        "**Aliasing** occurs when a signal contains frequencies above the Nyquist frequency (sample_rate/2). These frequencies are \"mirrored\" back into the audible range as phantom frequencies.\n\n**Example:** Recording a 23 kHz sine wave at 44.1 kHz:\n- Nyquist = 22,050 Hz\n- 23,000 Hz > Nyquist → aliases to 44,100 - 23,000 = 21,100 Hz\n- You hear a false 21,100 Hz tone instead\n\n**Prevention:**\n1. **Anti-aliasing filter (ADC):** Analog low-pass filter before the converter cuts frequencies above Nyquist. Steep roll-off needed.\n2. **Oversampling:** Sample at a much higher rate (e.g., 8×), apply a gentle digital filter, then downsample. Relaxes analog filter requirements.\n3. **In DSP/synthesis:** When generating signals digitally (oscillators, distortion), use oversampling internally:\n   ```python\n   # Oversample 4x, process, then downsample\n   upsampled = scipy.signal.resample_poly(signal, up=4, down=1)\n   processed = apply_distortion(upsampled)  # generates harmonics\n   result = scipy.signal.resample_poly(processed, up=1, down=4)\n   ```\n4. **Band-limited oscillators:** Use BLIT, PolyBLEP, or wavetable methods that inherently avoid harmonics above Nyquist."
    ))

    pairs.append(msg(
        "What is dithering in audio and when should you use it?",
        "**Dithering** is the intentional addition of low-level noise when reducing bit depth. It converts quantization distortion (correlated, harsh) into uncorrelated noise (benign hiss).\n\n**Why it's needed:** When truncating from 24-bit to 16-bit, the lost bits create quantization error that correlates with the signal, producing audible distortion on quiet passages.\n\n**Types of dither:**\n- **RPDF** (Rectangular): Uniform random noise, ±0.5 LSB. Simple but leaves some distortion.\n- **TPDF** (Triangular): Sum of two RPDF sources, ±1 LSB. Eliminates distortion completely. Standard choice.\n- **Noise-shaped** (e.g., POW-R, Apogee UV22HR): Pushes dither noise energy into frequencies above ~15 kHz where the ear is less sensitive.\n\n**When to dither:**\n- Final mixdown: 32-bit float → 16-bit for CD\n- Any bit-depth reduction in the signal chain\n- **Do NOT** dither when staying in float or going to a higher bit depth\n- **Do NOT** dither multiple times — only at the final truncation step\n\n**Implementation:**\n```python\nimport numpy as np\ndef apply_tpdf_dither(samples_float: np.ndarray, target_bits: int = 16) -> np.ndarray:\n    scale = 2 ** (target_bits - 1)\n    dither = (np.random.uniform(-0.5, 0.5, len(samples_float))\n              + np.random.uniform(-0.5, 0.5, len(samples_float)))\n    return np.round(samples_float * scale + dither) / scale\n```"
    ))

    pairs.append(msg(
        "Explain the Fast Fourier Transform (FFT) and its use in audio processing.",
        "The **FFT** (Fast Fourier Transform) converts a time-domain signal into its frequency-domain representation in O(n log n) time.\n\n**Key concepts:**\n- Input: N time-domain samples\n- Output: N/2+1 complex frequency bins (for real input)\n- Frequency resolution: `sample_rate / N` Hz per bin\n- Example: 1024-sample FFT at 44.1 kHz → 43 Hz resolution, bins up to 22,050 Hz\n\n**Audio applications:**\n1. **Spectrum analysis:** Visualize frequency content (spectrograms, analyzers)\n2. **EQ:** Multiply frequency bins to boost/cut bands\n3. **Convolution reverb:** Convolve with impulse response in frequency domain (FFT multiply is faster than time-domain convolution for long IRs)\n4. **Pitch detection:** Find the dominant frequency peak\n5. **Noise reduction:** Identify and subtract noise spectrum\n6. **Time stretching:** Phase vocoder (STFT + phase manipulation)\n\n**Windowing** is critical — applying a window function (Hann, Blackman, Kaiser) before FFT prevents spectral leakage from non-periodic signal edges.\n\n**STFT** (Short-Time FFT): Apply FFT to overlapping windowed frames for time-varying spectral analysis:\n```python\nimport numpy as np\ndef stft(signal, fft_size=2048, hop_size=512, window='hann'):\n    win = np.hanning(fft_size)\n    frames = []\n    for i in range(0, len(signal) - fft_size, hop_size):\n        frame = signal[i:i+fft_size] * win\n        frames.append(np.fft.rfft(frame))\n    return np.array(frames)\n```"
    ))

    pairs.append(msg(
        "What is the difference between FIR and IIR filters in audio?",
        "**FIR (Finite Impulse Response):**\n- Output depends only on current and past input samples\n- Formula: `y[n] = b0*x[n] + b1*x[n-1] + ... + bN*x[n-N]`\n- Always stable (no feedback)\n- Can achieve perfect linear phase (no phase distortion)\n- Requires many taps (coefficients) for steep roll-off → higher latency and CPU\n- Typical: 64-512 taps\n\n**IIR (Infinite Impulse Response):**\n- Output depends on both input and previous output samples (feedback)\n- Formula: `y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1] - a2*y[n-2]`\n- Much fewer coefficients for same steepness (a biquad = 5 coefficients)\n- Non-linear phase (unless cascade of allpass correctors)\n- Can be unstable if poorly designed\n- Types: Butterworth (flat passband), Chebyshev (steep but ripple), Bessel (best phase)\n\n**In practice:**\n| Use case | Choose |\n|----------|--------|\n| Parametric EQ | IIR biquad (low CPU, real-time) |\n| Linear-phase mastering EQ | FIR |\n| Convolution reverb | FIR (impulse response) |\n| Anti-aliasing | IIR (low latency) |\n| Crossover filters | IIR Linkwitz-Riley (steep, flat sum) |\n\n**Biquad** (2nd-order IIR) is the workhorse of audio — cascaded biquads implement most EQs and filters."
    ))

    pairs.append(msg(
        "Write a biquad low-pass filter in Python for audio processing.",
        """```python
import numpy as np

class BiquadLowPass:
    \"\"\"2nd-order IIR low-pass filter (Robert Bristow-Johnson's cookbook).\"\"\"

    def __init__(self, sample_rate: float, cutoff: float, q: float = 0.707):
        w0 = 2.0 * np.pi * cutoff / sample_rate
        alpha = np.sin(w0) / (2.0 * q)

        b0 = (1.0 - np.cos(w0)) / 2.0
        b1 = 1.0 - np.cos(w0)
        b2 = (1.0 - np.cos(w0)) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha

        # Normalize
        self.b = np.array([b0/a0, b1/a0, b2/a0])
        self.a = np.array([1.0, a1/a0, a2/a0])

        # State (Direct Form II Transposed)
        self.z1 = 0.0
        self.z2 = 0.0

    def process_sample(self, x: float) -> float:
        y = self.b[0] * x + self.z1
        self.z1 = self.b[1] * x - self.a[1] * y + self.z2
        self.z2 = self.b[2] * x - self.a[2] * y
        return y

    def process_block(self, block: np.ndarray) -> np.ndarray:
        out = np.empty_like(block)
        for i in range(len(block)):
            out[i] = self.process_sample(block[i])
        return out

# Usage
lpf = BiquadLowPass(sample_rate=44100, cutoff=1000, q=0.707)
signal = np.random.randn(44100)  # 1 second of noise
filtered = lpf.process_block(signal)
```

The Q factor of 0.707 (1/sqrt(2)) gives a Butterworth response — maximally flat passband with no resonance peak. Higher Q values create a resonant bump at the cutoff frequency."""
    ))

    pairs.append(msg(
        "What is a circular buffer and why is it essential for real-time audio?",
        "A **circular buffer** (ring buffer) is a fixed-size array where write and read pointers wrap around, treating the array as if its ends are connected.\n\n**Why essential for audio:**\n1. **Delay lines:** Store past samples for delay effects, reverb, chorus\n2. **Audio I/O buffering:** Bridge between hardware callbacks and processing\n3. **Lock-free producer/consumer:** Audio thread writes, UI thread reads — no mutex needed\n4. **Constant memory:** No allocation/deallocation during real-time processing\n\n**Implementation:**\n```python\nimport numpy as np\n\nclass CircularBuffer:\n    def __init__(self, size: int):\n        self.buffer = np.zeros(size)\n        self.size = size\n        self.write_pos = 0\n\n    def write(self, sample: float):\n        self.buffer[self.write_pos] = sample\n        self.write_pos = (self.write_pos + 1) % self.size\n\n    def read(self, delay: int) -> float:\n        \"\"\"Read sample from `delay` samples ago.\"\"\"\n        pos = (self.write_pos - delay) % self.size\n        return self.buffer[pos]\n\n    def read_interpolated(self, delay: float) -> float:\n        \"\"\"Fractional delay with linear interpolation.\"\"\"\n        int_delay = int(delay)\n        frac = delay - int_delay\n        s0 = self.read(int_delay)\n        s1 = self.read(int_delay + 1)\n        return s0 + frac * (s1 - s0)\n```\n\n**Key properties:**\n- O(1) read/write\n- No dynamic allocation (real-time safe)\n- Power-of-2 sizes allow bitwise modulo: `pos & (size - 1)`\n- Fractional interpolation needed for modulated delays (chorus, flanger)"
    ))

    pairs.append(msg(
        "What is the decibel (dB) scale and how do you convert between linear and dB in audio?",
        "The **decibel (dB)** is a logarithmic ratio used because human hearing perception is logarithmic.\n\n**Formulas:**\n```\ndB = 20 * log10(amplitude_ratio)     # for amplitude/voltage\ndB = 10 * log10(power_ratio)         # for power\n\namplitude_ratio = 10^(dB / 20)\npower_ratio = 10^(dB / 10)\n```\n\n**Key reference points:**\n| dB | Amplitude ratio | Perception |\n|----|----------------|------------|\n| 0 dB | 1.0 | Unity (no change) |\n| -6 dB | 0.5 | Half amplitude |\n| -12 dB | 0.25 | Quarter amplitude |\n| -20 dB | 0.1 | Noticeably quieter |\n| -60 dB | 0.001 | Nearly inaudible |\n| -96 dB | ~0.000016 | 16-bit noise floor |\n| +6 dB | 2.0 | Double amplitude |\n| +20 dB | 10.0 | 10× louder |\n\n**dBFS** (dB Full Scale): 0 dBFS = maximum digital level. All values are negative.\n\n**Python conversion:**\n```python\nimport numpy as np\n\ndef linear_to_db(linear: float) -> float:\n    return 20.0 * np.log10(max(linear, 1e-10))  # avoid log(0)\n\ndef db_to_linear(db: float) -> float:\n    return 10.0 ** (db / 20.0)\n\ndef rms_db(signal: np.ndarray) -> float:\n    rms = np.sqrt(np.mean(signal ** 2))\n    return linear_to_db(rms)\n```"
    ))

    pairs.append(msg(
        "What is an envelope in audio and how does ADSR work?",
        "An **envelope** shapes how a sound's amplitude (or other parameter) changes over time.\n\n**ADSR** is the most common envelope with 4 stages:\n\n1. **Attack** (time): How quickly the sound reaches peak level after a note-on. Fast (1-10ms) for percussive, slow (100ms-2s) for pads.\n2. **Decay** (time): Time to fall from peak to sustain level.\n3. **Sustain** (level, not time): The steady level held while the key is pressed. 0.0-1.0.\n4. **Release** (time): Time to fade to zero after note-off.\n\n**Implementation:**\n```python\nclass ADSREnvelope:\n    def __init__(self, attack: float, decay: float,\n                 sustain: float, release: float, sr: float):\n        self.attack_samples = int(attack * sr)\n        self.decay_samples = int(decay * sr)\n        self.sustain_level = sustain\n        self.release_samples = int(release * sr)\n        self.state = 'idle'  # idle, attack, decay, sustain, release\n        self.pos = 0\n        self.level = 0.0\n\n    def note_on(self):\n        self.state = 'attack'\n        self.pos = 0\n\n    def note_off(self):\n        self.state = 'release'\n        self.pos = 0\n\n    def next_sample(self) -> float:\n        if self.state == 'attack':\n            self.level = self.pos / max(self.attack_samples, 1)\n            self.pos += 1\n            if self.pos >= self.attack_samples:\n                self.state = 'decay'\n                self.pos = 0\n        elif self.state == 'decay':\n            self.level = 1.0 - (1.0 - self.sustain_level) * self.pos / max(self.decay_samples, 1)\n            self.pos += 1\n            if self.pos >= self.decay_samples:\n                self.state = 'sustain'\n        elif self.state == 'sustain':\n            self.level = self.sustain_level\n        elif self.state == 'release':\n            self.level = self.sustain_level * (1.0 - self.pos / max(self.release_samples, 1))\n            self.pos += 1\n            if self.pos >= self.release_samples:\n                self.state = 'idle'\n                self.level = 0.0\n        return self.level\n```\n\n**Variations:** AR (attack-release), AHD (attack-hold-decay), AHDSR, multi-stage (Serum has unlimited points), exponential curves (more natural than linear)."
    ))

    pairs.append(msg(
        "What is the difference between peak and RMS level metering in audio?",
        "**Peak metering** shows the maximum instantaneous sample value — important for preventing digital clipping (exceeding 0 dBFS).\n\n**RMS metering** (Root Mean Square) shows the average energy over a time window — correlates better with perceived loudness.\n\n**Comparison:**\n| Aspect | Peak | RMS |\n|--------|------|-----|\n| Measures | Instantaneous maximum | Average energy |\n| Loudness correlation | Poor | Good |\n| Clipping detection | Essential | Not useful |\n| Window | Per-sample or per-block | Typically 300ms |\n| Crest factor | Always higher | Always lower |\n\n**Crest factor** = Peak - RMS (in dB). Higher crest factor means more dynamic/transient content:\n- Sine wave: 3 dB\n- Pop music: 6-10 dB\n- Classical: 15-20 dB\n- Single click: 30+ dB\n\n**LUFS** (Loudness Units Full Scale) is the modern standard for loudness measurement — uses K-weighting (approximates human hearing) and integrates over time:\n- Momentary: 400ms window\n- Short-term: 3s window\n- Integrated: entire program\n- Streaming targets: Spotify -14 LUFS, YouTube -13 LUFS, Apple Music -16 LUFS"
    ))

    pairs.append(msg(
        "Explain what a convolution reverb is and how it works.",
        "**Convolution reverb** simulates real acoustic spaces by convolving a dry signal with an **impulse response** (IR) — a recording of a room's response to a brief impulse (like a starter pistol or sine sweep).\n\n**How it works:**\n1. Record an IR in the desired space (or synthesize one)\n2. Convolve: `wet[n] = sum(dry[k] * ir[n-k] for k in range(len(ir)))`\n3. Mix dry + wet\n\n**Time-domain convolution** is O(N×M) where N=signal length, M=IR length. For a 2-second IR at 48 kHz (96,000 samples), this is extremely expensive.\n\n**FFT-based convolution** is O(N log N):\n```python\nimport numpy as np\nfrom scipy.signal import fftconvolve\n\ndef convolve_reverb(dry: np.ndarray, ir: np.ndarray, wet_mix: float = 0.3) -> np.ndarray:\n    wet = fftconvolve(dry, ir, mode='full')[:len(dry)]\n    wet /= np.max(np.abs(wet) + 1e-10)  # normalize\n    return (1 - wet_mix) * dry + wet_mix * wet\n```\n\n**Real-time challenges:**\n- Can't wait for the full signal — use **partitioned convolution** (overlap-save or overlap-add)\n- Split the IR into blocks, convolve each block with the current audio frame, sum results\n- Uniform partitioning: all blocks same FFT size\n- Non-uniform: small FFT for early reflections (low latency), larger FFT for tail (efficient)\n\n**IR sources:** Altiverb, OpenAIR database, impulse-response.net, or record your own with a sine sweep + deconvolution."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 3. Synthesis Types
# ---------------------------------------------------------------------------

def gen_synthesis() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "Explain subtractive synthesis and how it works.",
        "**Subtractive synthesis** starts with a harmonically rich waveform and sculpts the sound by removing frequencies with filters.\n\n**Signal chain:**\n```\nOscillator(s) → Mixer → Filter → Amplifier → Output\n    ↑               ↑           ↑\n   LFO            Envelope    Envelope\n```\n\n**Components:**\n1. **Oscillators:** Generate raw waveforms — sawtooth (all harmonics), square (odd harmonics), pulse (variable duty cycle), noise\n2. **Filter:** Typically low-pass (LP), removes upper harmonics. Key parameters:\n   - Cutoff frequency: where attenuation begins\n   - Resonance (Q): boost at cutoff, creates \"squelchy\" sound\n   - Slope: 12 dB/oct (2-pole), 24 dB/oct (4-pole, Moog-style)\n3. **Amplifier:** Controls volume over time via ADSR envelope\n4. **Modulation:** LFOs and envelopes modulate cutoff, pitch, amplitude\n\n**Classic synths:** Moog Minimoog (1970), Roland Juno-106, Sequential Prophet-5\n\n**Why it works:** Natural sounds have formants (resonant peaks) created by filtering. A sawtooth through a resonant low-pass mimics vowel sounds, brass, strings. Opening/closing the filter over time gives expressive timbral movement."
    ))

    pairs.append(msg(
        "Explain FM synthesis with a code example.",
        "**FM (Frequency Modulation) synthesis** creates complex timbres by modulating one oscillator's frequency with another.\n\n**Terminology:**\n- **Carrier:** The oscillator you hear\n- **Modulator:** The oscillator that modulates the carrier's frequency\n- **Modulation index:** Depth of modulation (controls harmonic richness)\n- **C:M ratio:** Frequency ratio between carrier and modulator (determines harmonic series)\n\n**The math:**\n```\noutput(t) = sin(2π * fc * t + I * sin(2π * fm * t))\n```\nwhere `fc` = carrier freq, `fm` = modulator freq, `I` = modulation index\n\n**Python implementation:**\n```python\nimport numpy as np\n\ndef fm_synthesis(\n    carrier_freq: float,\n    mod_freq: float,\n    mod_index: float,\n    duration: float,\n    sample_rate: int = 44100,\n) -> np.ndarray:\n    t = np.arange(int(duration * sample_rate)) / sample_rate\n    modulator = mod_index * np.sin(2 * np.pi * mod_freq * t)\n    carrier = np.sin(2 * np.pi * carrier_freq * t + modulator)\n    return carrier\n\n# Bell-like sound: C:M ratio = 1:1.4, high mod index\nbell = fm_synthesis(440, 616, 5.0, 2.0)\n\n# Electric piano: C:M = 1:1, moderate mod index with decay\nepiano = fm_synthesis(440, 440, 3.0, 1.5)\n```\n\n**Key properties:**\n- Integer C:M ratios → harmonic spectra (musical)\n- Non-integer ratios → inharmonic (bells, metallic sounds)\n- Higher mod index → more harmonics (brighter)\n- Time-varying index (via envelope) → timbral evolution\n\n**6-operator FM** (Yamaha DX7): 6 sine oscillators in 32 different algorithms (routing configurations). Still the standard for FM synthesis."
    ))

    pairs.append(msg(
        "What is wavetable synthesis and how does it differ from sample-based synthesis?",
        "**Wavetable synthesis** stores single-cycle waveforms in a table and morphs between them for timbral animation.\n\n**How it works:**\n1. A **wavetable** contains 32-256 single-cycle waveforms (each typically 2048 samples)\n2. A **position** parameter scans through the table\n3. The synth interpolates between adjacent waveforms for smooth morphing\n4. Phase accumulator steps through the current waveform at the desired frequency\n\n**vs Sample-based:**\n| Wavetable | Sample-based |\n|-----------|-------------|\n| Single-cycle loops | Multi-second recordings |\n| Tiny memory (~100 KB) | Large memory (100s MB) |\n| Morphable timbre | Fixed timbre per sample |\n| Synthetic/evolving sounds | Realistic instrument reproduction |\n| Pitch-independent timbre | Pitch affects timbre (playback speed) |\n\n**Implementation sketch:**\n```python\nclass WavetableOsc:\n    def __init__(self, wavetable: np.ndarray, num_waves: int):\n        self.table = wavetable.reshape(num_waves, -1)  # [waves, samples_per_wave]\n        self.phase = 0.0\n        self.position = 0.0  # 0.0 to num_waves-1\n\n    def next_sample(self, freq: float, sr: float) -> float:\n        # Interpolate between adjacent waveforms\n        idx = int(self.position)\n        frac = self.position - idx\n        wave_a = self.table[idx % len(self.table)]\n        wave_b = self.table[(idx + 1) % len(self.table)]\n\n        # Read from blended waveform\n        pos = int(self.phase * len(wave_a)) % len(wave_a)\n        sample = wave_a[pos] * (1 - frac) + wave_b[pos] * frac\n\n        self.phase += freq / sr\n        self.phase %= 1.0\n        return sample\n```\n\n**Synths:** Serum (Xfer), Vital (free), PPG Wave, Waldorf series."
    ))

    pairs.append(msg(
        "What is granular synthesis?",
        "**Granular synthesis** deconstructs audio into tiny pieces called **grains** (1-100ms) and reassembles them to create textures, stretching, and transformations.\n\n**Parameters per grain:**\n- **Position:** Where in the source audio to read from\n- **Size:** Grain duration (5-50ms typical)\n- **Density:** Grains per second (10-1000+)\n- **Pitch:** Playback rate of each grain (independent of position)\n- **Pan:** Stereo placement\n- **Window/Envelope:** Shape applied to each grain (Hann, Gaussian, trapezoid) to avoid clicks\n- **Randomization:** Jitter on all parameters for organic texture\n\n**Key capabilities:**\n1. **Time stretching without pitch change:** Freeze position, keep pitch → sustained textures\n2. **Pitch shifting without time change:** Change pitch, advance position normally\n3. **Texture generation:** High density + randomized position = ambient clouds from any source\n4. **Scrubbing:** Manually control position for DJ-like effects\n\n**Implementation concept:**\n```python\nclass GrainCloud:\n    def __init__(self, source: np.ndarray, sr: int):\n        self.source = source\n        self.sr = sr\n        self.grains = []  # active grains\n\n    def spawn_grain(self, position: float, size_ms: float,\n                     pitch: float = 1.0):\n        size = int(size_ms * self.sr / 1000)\n        start = int(position * len(self.source))\n        window = np.hanning(size)\n        grain_data = self.source[start:start+size] * window\n        self.grains.append({'data': grain_data, 'pitch': pitch, 'pos': 0})\n```\n\n**Software:** Granulator II (Ableton), Quanta (Audio Damage), Paul Stretch (extreme stretching), Borderlands (iOS)."
    ))

    pairs.append(msg(
        "Explain additive synthesis and its relationship to Fourier theory.",
        "**Additive synthesis** builds complex sounds by summing individual sine waves (partials/harmonics), directly applying Fourier's theorem: any periodic signal can be decomposed into a sum of sinusoids.\n\n**Principle:**\n```\nsound(t) = Σ Aₙ * sin(2π * n * f₀ * t + φₙ)\n```\nwhere `Aₙ` = amplitude of partial n, `f₀` = fundamental frequency, `φₙ` = phase\n\n**Building common waveforms:**\n```python\nimport numpy as np\n\ndef additive_saw(f0, num_harmonics, duration, sr=44100):\n    t = np.arange(int(duration * sr)) / sr\n    signal = np.zeros_like(t)\n    for n in range(1, num_harmonics + 1):\n        signal += (1.0 / n) * np.sin(2 * np.pi * n * f0 * t)\n    return signal / np.max(np.abs(signal))\n\ndef additive_square(f0, num_harmonics, duration, sr=44100):\n    t = np.arange(int(duration * sr)) / sr\n    signal = np.zeros_like(t)\n    for n in range(1, num_harmonics + 1, 2):  # odd harmonics only\n        signal += (1.0 / n) * np.sin(2 * np.pi * n * f0 * t)\n    return signal / np.max(np.abs(signal))\n```\n\n**Strengths:**\n- Complete control over every partial (amplitude, frequency, phase)\n- Can resynthesize analyzed sounds (spectral modeling)\n- Natural for simulating acoustic instruments (organ pipes, bells)\n- No aliasing if harmonics are kept below Nyquist\n\n**Weaknesses:**\n- CPU-expensive for many partials (100+ per voice)\n- Parameter explosion: managing 100s of envelopes\n- Hard to create noise-like or stochastic components\n\n**Software:** Razor (NI), Loom II, Alchemy (Apple), organ simulators. Additive + noise = **Spectral Modeling Synthesis** (SMS)."
    ))

    pairs.append(msg(
        "What is physical modeling synthesis?",
        "**Physical modeling synthesis** simulates the physics of real instruments mathematically rather than using samples or oscillators.\n\n**Techniques:**\n\n1. **Karplus-Strong** (plucked strings):\n   - Fill a delay line with noise (burst excitation)\n   - Feed output back through a low-pass filter (energy loss)\n   - Delay length = 1/frequency\n   ```python\n   def karplus_strong(freq, duration, sr=44100, decay=0.996):\n       N = int(sr / freq)\n       buf = np.random.uniform(-1, 1, N)  # noise burst\n       out = np.zeros(int(duration * sr))\n       for i in range(len(out)):\n           out[i] = buf[i % N]\n           buf[i % N] = decay * 0.5 * (buf[i % N] + buf[(i + 1) % N])\n       return out\n   ```\n\n2. **Digital Waveguides** (strings, tubes, plates):\n   - Model traveling waves in both directions along a string/tube\n   - Junction scattering at boundaries (bridge, nut, finger)\n   - Efficient: 2 delay lines + filters per string\n\n3. **Modal synthesis** (struck objects, bells):\n   - Decompose resonance into modes (frequency, decay, amplitude)\n   - Sum damped sinusoids excited by an impulse\n\n4. **Finite element / mass-spring** (membranes, complex bodies):\n   - 2D/3D mesh of interconnected masses\n   - Most physically accurate but very CPU-expensive\n\n**Advantages:** Expressive, responsive to continuous controller input, tiny memory, unlimited articulation. **Disadvantage:** Hard to design, computationally demanding for complex models.\n\n**Instruments:** Yamaha VL1, Pianoteq (Modartt), AAS Chromaphone, STK (Synthesis Toolkit)."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 4. Audio Effects
# ---------------------------------------------------------------------------

def gen_audio_effects() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How does a digital delay effect work? Write a basic implementation.",
        "A **digital delay** stores audio in a buffer and plays it back after a set time.\n\n**Parameters:**\n- Delay time (ms or tempo-synced)\n- Feedback: how much delayed signal feeds back into the input (0-100%)\n- Mix: dry/wet balance\n- Optional: filtering in feedback path (darker repeats)\n\n**Implementation:**\n```python\nimport numpy as np\n\nclass Delay:\n    def __init__(self, max_delay_ms: float, sample_rate: int):\n        max_samples = int(max_delay_ms * sample_rate / 1000)\n        self.buffer = np.zeros(max_samples)\n        self.write_pos = 0\n        self.sr = sample_rate\n\n    def process(self, input_signal: np.ndarray,\n                delay_ms: float, feedback: float = 0.5,\n                mix: float = 0.5) -> np.ndarray:\n        delay_samples = int(delay_ms * self.sr / 1000)\n        output = np.zeros_like(input_signal)\n\n        for i in range(len(input_signal)):\n            # Read from delay buffer\n            read_pos = (self.write_pos - delay_samples) % len(self.buffer)\n            delayed = self.buffer[read_pos]\n\n            # Write input + feedback to buffer\n            self.buffer[self.write_pos] = input_signal[i] + feedback * delayed\n            self.write_pos = (self.write_pos + 1) % len(self.buffer)\n\n            # Mix dry/wet\n            output[i] = (1 - mix) * input_signal[i] + mix * delayed\n\n        return output\n```\n\n**Variations:**\n- **Ping-pong delay:** Alternates between L and R channels\n- **Tape delay:** Modulated delay time + filtered feedback (wow and flutter)\n- **Multi-tap:** Multiple read positions for rhythmic patterns\n- **Reverse delay:** Read buffer backwards\n- **Ducking delay:** Side-chain compresses wet signal when dry is present"
    ))

    pairs.append(msg(
        "How does a chorus effect work?",
        "**Chorus** creates the illusion of multiple voices playing in unison by adding slightly detuned and delayed copies of the signal.\n\n**How it works:**\n1. Take the dry signal\n2. Pass it through a short delay line (10-30ms)\n3. Modulate the delay time with a slow LFO (0.1-5 Hz)\n4. Mix the modulated delayed signal with the dry signal\n\nThe time-varying delay creates pitch variation (Doppler effect), simulating natural detuning between players.\n\n**Implementation:**\n```python\nclass Chorus:\n    def __init__(self, sr: int, depth_ms: float = 3.0,\n                 rate_hz: float = 1.5, mix: float = 0.5):\n        self.sr = sr\n        self.depth = depth_ms * sr / 1000  # depth in samples\n        self.rate = rate_hz\n        self.mix = mix\n        self.center_delay = int(15 * sr / 1000)  # 15ms center\n        buf_size = self.center_delay + int(self.depth) + 2\n        self.buffer = np.zeros(buf_size)\n        self.write_pos = 0\n        self.phase = 0.0\n\n    def process_sample(self, x: float) -> float:\n        self.buffer[self.write_pos] = x\n        # LFO modulates delay time\n        mod = self.depth * np.sin(2 * np.pi * self.phase)\n        delay = self.center_delay + mod\n        # Fractional delay read with interpolation\n        read_pos = self.write_pos - delay\n        idx = int(read_pos) % len(self.buffer)\n        frac = read_pos - int(read_pos)\n        delayed = (self.buffer[idx] * (1 - frac) +\n                   self.buffer[(idx + 1) % len(self.buffer)] * frac)\n        self.write_pos = (self.write_pos + 1) % len(self.buffer)\n        self.phase += self.rate / self.sr\n        return (1 - self.mix) * x + self.mix * delayed\n```\n\n**Variants:**\n- **Flanger:** Same principle but shorter delay (1-5ms) + feedback → comb filter sweep\n- **Ensemble:** Multiple chorus voices (2-4 LFOs, different phases)\n- **Vibrato:** Wet-only (100% mix) = pure pitch modulation"
    ))

    pairs.append(msg(
        "How does dynamic range compression work in audio?",
        "A **compressor** reduces the dynamic range of audio by attenuating signals above a threshold.\n\n**Parameters:**\n- **Threshold** (dB): Level above which compression begins\n- **Ratio** (e.g., 4:1): For every 4 dB above threshold, output increases by 1 dB\n- **Attack** (ms): How quickly the compressor reacts to transients (0.1-100ms)\n- **Release** (ms): How quickly it stops compressing after level drops (50-500ms)\n- **Knee** (dB): Hard knee = abrupt onset; soft knee = gradual transition\n- **Makeup gain** (dB): Boost output to compensate for reduced peaks\n\n**Signal flow:**\n```\ninput → level detector → gain computer → gain smoothing → gain stage → output\n                                                                ↑\n                                                           makeup gain\n```\n\n**Gain calculation:**\n```python\ndef compute_gain_db(input_db: float, threshold: float,\n                     ratio: float) -> float:\n    if input_db <= threshold:\n        return 0.0  # no compression\n    over = input_db - threshold\n    return -(over - over / ratio)  # gain reduction\n\n# Example: input = -6 dBFS, threshold = -20 dB, ratio = 4:1\n# Over = 14 dB, compressed over = 14/4 = 3.5 dB\n# Gain reduction = -(14 - 3.5) = -10.5 dB\n# Output = -6 + (-10.5) = -16.5 dBFS\n```\n\n**Types:**\n- **Limiter:** Ratio ∞:1, fast attack — prevents clipping\n- **Expander:** Reduces signal below threshold (opposite of compressor)\n- **Gate:** Extreme expander — silences signal below threshold\n- **Multiband:** Separate compression per frequency band\n- **Sidechain:** Detector input differs from processed signal (e.g., kick ducks bass)"
    ))

    pairs.append(msg(
        "How does a parametric equalizer work?",
        "A **parametric EQ** adjusts the frequency response using configurable filter bands, each defined by three parameters.\n\n**Per-band parameters:**\n- **Frequency** (Hz): Center frequency of the band\n- **Gain** (dB): Boost or cut at that frequency\n- **Q** (quality factor): Width of the band. Higher Q = narrower. Q=0.7 is wide, Q=10 is surgical\n\n**Filter types per band:**\n- **Bell/Peak:** Boost or cut around a center frequency (most common)\n- **Low Shelf:** Boost/cut everything below a frequency\n- **High Shelf:** Boost/cut everything above a frequency\n- **Low Pass:** Cut everything above the frequency\n- **High Pass:** Cut everything below the frequency\n- **Notch:** Deep narrow cut at exact frequency (for removing hum/resonance)\n\n**Implementation** uses cascaded biquad filters (Robert Bristow-Johnson Audio EQ Cookbook):\n```python\ndef peaking_eq_coefficients(sr, freq, gain_db, q):\n    A = 10 ** (gain_db / 40)  # amplitude\n    w0 = 2 * np.pi * freq / sr\n    alpha = np.sin(w0) / (2 * q)\n\n    b0 = 1 + alpha * A\n    b1 = -2 * np.cos(w0)\n    b2 = 1 - alpha * A\n    a0 = 1 + alpha / A\n    a1 = -2 * np.cos(w0)\n    a2 = 1 - alpha / A\n    return [b0/a0, b1/a0, b2/a0], [1, a1/a0, a2/a0]\n```\n\n**Typical setup** (6-band semi-parametric):\n1. High-pass filter (rumble removal, 30-80 Hz)\n2. Low shelf (bass warmth/thinning)\n3. Low-mid bell (mud removal, 200-400 Hz)\n4. Mid bell (presence, 1-4 kHz)\n5. High shelf (air/brightness)\n6. Low-pass filter (harshness removal, 16-20 kHz)"
    ))

    pairs.append(msg(
        "Explain how reverb algorithms work (algorithmic vs convolution).",
        "**Algorithmic reverb** simulates room acoustics using delay networks and filters. **Convolution reverb** uses recorded impulse responses.\n\n**Algorithmic (Schroeder/Moorer):**\n```\nInput → Early Reflections → Late Reverb → Output\n               ↑                    ↑\n         Tap delay line      Feedback Delay Network (FDN)\n```\n\n**Components:**\n1. **Pre-delay:** Gap before first reflection (simulates room size)\n2. **Early reflections:** Sparse taps at specific delays modeling first wall bounces\n3. **Diffusion:** Allpass filters smear the signal, creating density\n4. **Late reverb tail:** Feedback delay network (FDN) with 4-16 delay lines:\n   - Different prime-number delay lengths (avoid periodicity)\n   - Mixing matrix (Hadamard or householder) redistributes energy\n   - Low-pass filters in feedback (high frequencies decay faster = natural)\n   - Modulated delay times (avoid metallic ringing)\n\n**Key parameters:**\n- Decay time (RT60): Time for reverb to drop 60 dB\n- Room size: Scales delay line lengths\n- Damping: High-frequency decay rate\n- Diffusion: How quickly reflections blur into a smooth tail\n- Pre-delay: Initial gap\n\n**Algorithmic vs Convolution:**\n| | Algorithmic | Convolution |\n|--|-------------|-------------|\n| Realism | Good to excellent | Perfect (real spaces) |\n| CPU | Low-moderate | Higher (long IRs) |\n| Tweakability | Highly adjustable | Limited (pre-baked) |\n| Size | Parameters only | Large IR files |\n| Modulation | Easy (chorus in tail) | Difficult |\n\n**Classic algorithms:** Schroeder (1962), Moorer (1979), Dattorro plate (1997), FreeVerb, Griesinger (Lexicon)."
    ))

    pairs.append(msg(
        "How does a waveshaper/distortion effect work?",
        "**Waveshaping** applies a nonlinear transfer function to the signal, creating harmonics (distortion).\n\n**Transfer function:** Maps input amplitude to output amplitude. A straight line (y=x) = clean. Curves = distortion.\n\n**Common transfer functions:**\n```python\nimport numpy as np\n\ndef soft_clip(x: np.ndarray) -> np.ndarray:\n    \"\"\"Soft saturation (tanh).\"\"\"\n    return np.tanh(x)\n\ndef hard_clip(x: np.ndarray, threshold: float = 0.8) -> np.ndarray:\n    \"\"\"Hard clipping — harsh, digital-sounding.\"\"\"\n    return np.clip(x, -threshold, threshold)\n\ndef tube_saturation(x: np.ndarray, drive: float = 2.0) -> np.ndarray:\n    \"\"\"Asymmetric tube-like saturation (even + odd harmonics).\"\"\"\n    driven = x * drive\n    pos = 1 - np.exp(-driven[driven >= 0])\n    neg = -(1 - np.exp(driven[driven < 0])) * 0.8  # asymmetry\n    out = np.empty_like(x)\n    out[x >= 0] = pos\n    out[x < 0] = neg\n    return out\n\ndef bitcrusher(x: np.ndarray, bits: int = 8) -> np.ndarray:\n    \"\"\"Reduce bit depth for lo-fi distortion.\"\"\"\n    levels = 2 ** bits\n    return np.round(x * levels) / levels\n```\n\n**Important considerations:**\n1. **Oversampling:** Distortion creates new harmonics that can alias. Process at 2-4x sample rate:\n   ```\n   upsample → waveshape → low-pass filter → downsample\n   ```\n2. **Pre-filter:** Band-limit input before distortion (controls character)\n3. **Tone control:** Post-distortion EQ (cut harsh highs, shape mids)\n4. **Drive control:** Gain stage before waveshaper (more drive = more harmonics)\n\n**Types of distortion:**\n- **Overdrive:** Soft clipping, low drive (blues)\n- **Distortion:** Harder clipping, more sustain (rock)\n- **Fuzz:** Extreme clipping, square-wave-like (stoner, psychedelic)\n- **Saturation:** Subtle warmth (tape, tube emulation)\n- **Bitcrush:** Quantization noise (lo-fi, chiptune)"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 5. Frameworks
# ---------------------------------------------------------------------------

def gen_frameworks() -> list[dict]:
    pairs = []

    # JUCE
    pairs.append(msg(
        "What is JUCE and how do you structure a basic audio plugin?",
        "**JUCE** is a C++ framework for building cross-platform audio plugins (VST3, AU, AAX) and applications.\n\n**Project structure:**\n```\nMyPlugin/\n├── CMakeLists.txt          # or .jucer file for Projucer\n├── Source/\n│   ├── PluginProcessor.h   # Audio processing\n│   ├── PluginProcessor.cpp\n│   ├── PluginEditor.h      # GUI\n│   └── PluginEditor.cpp\n```\n\n**Minimal PluginProcessor:**\n```cpp\nclass MyPluginProcessor : public juce::AudioProcessor {\npublic:\n    MyPluginProcessor()\n        : AudioProcessor(BusesProperties()\n            .withInput(\"Input\", juce::AudioChannelSet::stereo())\n            .withOutput(\"Output\", juce::AudioChannelSet::stereo())) {}\n\n    void prepareToPlay(double sampleRate, int samplesPerBlock) override {\n        // Initialize DSP with sample rate and block size\n    }\n\n    void processBlock(juce::AudioBuffer<float>& buffer,\n                      juce::MidiBuffer& midiMessages) override {\n        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {\n            auto* data = buffer.getWritePointer(ch);\n            for (int i = 0; i < buffer.getNumSamples(); ++i) {\n                data[i] *= 0.5f;  // simple gain reduction\n            }\n        }\n    }\n\n    void releaseResources() override {}\n\n    // ... editor, state, parameter boilerplate\n};\n```\n\n**Key JUCE classes:**\n- `AudioProcessor`: Core processing (override `processBlock`)\n- `AudioProcessorEditor`: GUI (Component-based, or use the generic editor)\n- `AudioProcessorValueTreeState`: Parameter management with undo/redo\n- `dsp::ProcessorChain`: Chain DSP modules (IIR, gain, waveshaper)\n- `dsp::IIR::Filter`: Biquad filter with coefficient design methods\n\n**Build with CMake:**\n```cmake\njuce_add_plugin(MyPlugin\n    PLUGIN_MANUFACTURER_CODE MYPG\n    PLUGIN_CODE MYPL\n    FORMATS VST3 AU Standalone\n    PRODUCT_NAME \"My Plugin\")\ntarget_sources(MyPlugin PRIVATE Source/PluginProcessor.cpp Source/PluginEditor.cpp)\ntarget_link_libraries(MyPlugin PRIVATE juce::juce_audio_processors juce::juce_dsp)\n```"
    ))

    # SuperCollider
    pairs.append(msg(
        "Write a SuperCollider example that creates a synthesizer with filter and envelope.",
        """**SuperCollider** is an environment for real-time audio synthesis and algorithmic composition.

```supercollider
// Boot the audio server
s.boot;

// Define a SynthDef (synthesis definition)
SynthDef(\\filteredSaw, { |out=0, freq=440, cutoff=2000, res=0.3,
                          amp=0.5, gate=1, attack=0.01, release=0.5|
    var sig, env, filt;

    // Oscillator: band-limited sawtooth
    sig = Saw.ar(freq) + Saw.ar(freq * 1.005);  // slight detune

    // Filter: resonant low-pass (Moog-style)
    filt = MoogFF.ar(sig, cutoff, res);

    // Envelope: ASR (attack-sustain-release)
    env = EnvGen.kr(Env.asr(attack, 1, release), gate, doneAction: 2);

    Out.ar(out, Pan2.ar(filt * env * amp, 0));
}).add;

// Play it
x = Synth(\\filteredSaw, [\\freq, 220, \\cutoff, 1500, \\res, 0.7]);

// Modulate the filter cutoff
x.set(\\cutoff, 800);

// Release the note
x.set(\\gate, 0);

// Pattern-based sequencing
Pbind(
    \\instrument, \\filteredSaw,
    \\degree, Pseq([0, 2, 4, 5, 7], inf),  // scale degrees
    \\dur, Pseq([0.25, 0.25, 0.5, 0.25, 0.75], inf),
    \\cutoff, Pwhite(500, 3000),  // random cutoff per note
    \\amp, 0.3,
).play;
```

**Key SuperCollider concepts:**
- **SynthDef:** Compiled DSP graph sent to the server
- **UGens:** Unit generators (Saw, SinOsc, LPF, EnvGen, etc.)
- **.ar** = audio rate, **.kr** = control rate (cheaper for envelopes/LFOs)
- **Patterns** (Pbind, Pseq, Prand): Declarative sequencing
- **doneAction: 2** = free the synth when envelope finishes
- **Server architecture:** scsynth (C++ audio engine) + sclang (client language)"""
    ))

    # Faust
    pairs.append(msg(
        "What is Faust and how does it compare to other audio DSP languages?",
        "**Faust** (Functional Audio Stream) is a functional programming language designed specifically for audio DSP. It compiles to highly optimized C++, LLVM, WebAssembly, or other targets.\n\n**Example — resonant low-pass filter with LFO:**\n```faust\nimport(\"stdfaust.lib\");\n\nfreq = hslider(\"Frequency\", 440, 20, 20000, 1);\ncutoff = hslider(\"Cutoff\", 2000, 100, 10000, 1);\nres = hslider(\"Resonance\", 0.5, 0, 1, 0.01);\nlfo_rate = hslider(\"LFO Rate\", 2, 0.1, 20, 0.1);\nlfo_depth = hslider(\"LFO Depth\", 500, 0, 5000, 1);\n\noscillator = os.sawtooth(freq);\nmodulated_cutoff = cutoff + os.osc(lfo_rate) * lfo_depth;\nprocess = oscillator : fi.resonlp(modulated_cutoff, res, 1);\n```\n\n**Key features:**\n- **Block diagram algebra:** 5 composition operators (`,` parallel, `:` sequential, `~` feedback, `<:` split, `:>` merge)\n- **Single-sample processing:** No buffers, pure math\n- **Multi-target compilation:** VST, AU, Max/MSP, PureData, Web Audio, embedded, SuperCollider\n- **Automatic differentiation:** For ML-in-the-loop DSP\n- **Guaranteed real-time safe:** No allocation, no branching on data\n\n**Comparison:**\n| Feature | Faust | SuperCollider | Csound | PureData |\n|---------|-------|---------------|--------|----------|\n| Paradigm | Functional | OOP + Patterns | Score + Orchestra | Visual dataflow |\n| Compilation | Ahead-of-time | JIT (server) | Interpreted | Interpreted |\n| Plugin export | Yes (all formats) | No | Limited | Via Camomile |\n| Learning curve | Moderate | Moderate | Steep | Low |\n| Live coding | Via FaustLive | Excellent | Possible | Real-time patching |\n\n**Use Faust when:** You need maximum performance, plugin export, or embedded deployment."
    ))

    # PureData
    pairs.append(msg(
        "How does PureData work and what are its basic objects for audio?",
        "**Pure Data (Pd)** is a visual programming language for audio, video, and graphics. You connect objects with patch cords in a dataflow paradigm.\n\n**Basic audio objects:**\n| Object | Description |\n|--------|------------|\n| `[osc~ 440]` | Sine oscillator at 440 Hz |\n| `[phasor~ 440]` | Sawtooth (0-1 ramp) |\n| `[noise~]` | White noise |\n| `[*~ 0.5]` | Multiply signal by 0.5 (gain) |\n| `[+~ ]` | Add two signals |\n| `[lop~ 1000]` | 1-pole low-pass filter at 1000 Hz |\n| `[hip~ 100]` | 1-pole high-pass filter |\n| `[bp~ 1000 10]` | Band-pass filter (freq, Q) |\n| `[vcf~ ]` | Voltage-controlled filter (resonant) |\n| `[dac~]` | Audio output (digital-to-analog) |\n| `[adc~]` | Audio input |\n| `[delwrite~ name 1000]` | Write to delay line (name, ms) |\n| `[delread~ name 500]` | Read from delay line at 500ms |\n| `[vd~ name]` | Variable delay read (for chorus/flanger) |\n| `[tabread4~ table]` | 4-point interpolated wavetable read |\n| `[line~]` | Generate ramps (for envelopes) |\n| `[env~ 1024]` | RMS envelope follower |\n\n**Conventions:**\n- `~` suffix = audio rate (signal)\n- No `~` = control rate (messages)\n- Hot inlets (leftmost) trigger computation\n- Cold inlets store values for next trigger\n\n**Simple subtractive synth patch:**\n```\n[osc~ 440] → [*~ ] → [lop~ 2000] → [*~ ] → [dac~]\n                ↑                       ↑\n           [line~]                 [line~]\n           (amp env)              (filter env)\n```\n\n**Pd variants:**\n- **Pd-vanilla:** Original by Miller Puckette, minimal\n- **Pd-extended:** (deprecated) Added externals\n- **Purr Data:** Modern fork with better GUI\n- **Pd on embedded:** Runs on Raspberry Pi, Bela, Organelle\n- **libpd:** Embed Pd in mobile apps, games, plugins"
    ))

    # Web Audio API
    pairs.append(msg(
        "How do you build a synthesizer with the Web Audio API?",
        """The **Web Audio API** provides a graph-based audio processing system in the browser.

```javascript
class WebSynth {
  constructor() {
    this.ctx = new AudioContext();
    this.voices = new Map(); // note -> nodes

    // Master chain
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 0.3;

    // Reverb (convolution)
    this.convolver = this.ctx.createConvolver();
    this.reverbGain = this.ctx.createGain();
    this.reverbGain.gain.value = 0.2;

    // Routing
    this.masterGain.connect(this.ctx.destination);
    this.masterGain.connect(this.convolver);
    this.convolver.connect(this.reverbGain);
    this.reverbGain.connect(this.ctx.destination);
  }

  noteOn(midiNote, velocity = 0.8) {
    const freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    const now = this.ctx.currentTime;

    // Oscillator
    const osc = this.ctx.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.value = freq;

    // Filter
    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 200;
    filter.Q.value = 5;
    // Filter envelope
    filter.frequency.setValueAtTime(200, now);
    filter.frequency.linearRampToValueAtTime(
      2000 * velocity, now + 0.05  // attack
    );
    filter.frequency.exponentialRampToValueAtTime(
      800, now + 0.3  // decay
    );

    // Amp envelope
    const amp = this.ctx.createGain();
    amp.gain.setValueAtTime(0, now);
    amp.gain.linearRampToValueAtTime(velocity, now + 0.01);
    amp.gain.linearRampToValueAtTime(velocity * 0.7, now + 0.1);

    osc.connect(filter).connect(amp).connect(this.masterGain);
    osc.start(now);

    this.voices.set(midiNote, { osc, amp });
  }

  noteOff(midiNote) {
    const voice = this.voices.get(midiNote);
    if (!voice) return;
    const now = this.ctx.currentTime;
    voice.amp.gain.cancelScheduledValues(now);
    voice.amp.gain.setValueAtTime(voice.amp.gain.value, now);
    voice.amp.gain.linearRampToValueAtTime(0, now + 0.3);
    voice.osc.stop(now + 0.3);
    this.voices.delete(midiNote);
  }
}

// Usage with Web MIDI API
navigator.requestMIDIAccess().then(midi => {
  for (const input of midi.inputs.values()) {
    input.onmidimessage = (e) => {
      const [status, note, vel] = e.data;
      if ((status & 0xF0) === 0x90 && vel > 0)
        synth.noteOn(note, vel / 127);
      else if ((status & 0xF0) === 0x80 || vel === 0)
        synth.noteOff(note);
    };
  }
});
```

**Key Web Audio nodes:** OscillatorNode, GainNode, BiquadFilterNode, ConvolverNode, DelayNode, DynamicsCompressorNode, AnalyserNode (FFT), AudioWorkletNode (custom DSP), WaveShaperNode."""
    ))

    # AudioWorklet
    pairs.append(msg(
        "How does the AudioWorklet API work for custom DSP in the browser?",
        """**AudioWorklet** replaces the deprecated ScriptProcessorNode for custom sample-level DSP in the browser. It runs in a separate real-time thread.

**1. Worklet processor (runs in audio thread):**
```javascript
// processor.js
class MySynthProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'frequency', defaultValue: 440, minValue: 20, maxValue: 20000 },
      { name: 'gain', defaultValue: 0.5, minValue: 0, maxValue: 1 },
    ];
  }

  constructor() {
    super();
    this.phase = 0;
    // Receive messages from main thread
    this.port.onmessage = (e) => {
      if (e.data.type === 'wavetable') {
        this.wavetable = e.data.samples;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const output = outputs[0][0]; // mono output
    const freqParam = parameters.frequency;
    const gainParam = parameters.gain;

    for (let i = 0; i < output.length; i++) {
      // a-rate (per sample) or k-rate (per block) params
      const freq = freqParam.length > 1 ? freqParam[i] : freqParam[0];
      const gain = gainParam.length > 1 ? gainParam[i] : gainParam[0];

      output[i] = Math.sin(2 * Math.PI * this.phase) * gain;
      this.phase += freq / sampleRate;
      if (this.phase >= 1) this.phase -= 1;
    }
    return true; // keep alive
  }
}

registerProcessor('my-synth', MySynthProcessor);
```

**2. Main thread setup:**
```javascript
const ctx = new AudioContext();
await ctx.audioWorklet.addModule('processor.js');

const synth = new AudioWorkletNode(ctx, 'my-synth');
synth.connect(ctx.destination);

// Control parameters
synth.parameters.get('frequency').setValueAtTime(880, ctx.currentTime);

// Send data via MessagePort
synth.port.postMessage({ type: 'wavetable', samples: myWavetable });
```

**Key rules:**
- `process()` is called per audio block (128 samples by default)
- Return `true` to keep processing, `false` to stop
- No DOM access, no `fetch()`, no allocation in `process()`
- Use `SharedArrayBuffer` for zero-copy data sharing with main thread
- `sampleRate` is a global in the worklet scope"""
    ))

    # Csound
    pairs.append(msg(
        "What is Csound and how is it used for audio programming?",
        """**Csound** is one of the oldest and most powerful audio programming languages (since 1986). It uses a score+orchestra paradigm.

**Basic structure (.csd file):**
```xml
<CsoundSynthesizer>
<CsOptions>
-odac  ; real-time audio output
</CsOptions>

<CsInstruments>
sr = 44100      ; sample rate
ksmps = 32      ; control rate = sr/ksmps
nchnls = 2      ; stereo
0dbfs = 1       ; amplitude scale

; Instrument 1: subtractive synth
instr 1
  ifreq = p4          ; frequency from score
  iamp = p5           ; amplitude from score

  ; Oscillator
  asig vco2 iamp, ifreq, 0  ; band-limited sawtooth

  ; Filter with envelope
  kenv madsr 0.01, 0.1, 0.7, 0.3
  kcutoff = 500 + (3000 * kenv)
  afilt moogladder asig, kcutoff, 0.4

  ; Output
  outs afilt * kenv, afilt * kenv
endin

; Instrument 2: FM bell
instr 2
  ifreq = p4
  iamp = p5
  kenv expseg 1, p3, 0.001  ; exponential decay

  amod poscil 5 * ifreq * kenv, ifreq * 1.4  ; modulator
  acar poscil iamp * kenv, ifreq + amod       ; carrier
  outs acar, acar
endin
</CsInstruments>

<CsScore>
; instr start  dur  freq  amp
i 1    0      1    220   0.5
i 1    1      1    330   0.5
i 1    2      2    440   0.5
i 2    4      3    880   0.3
i 2    4.5    3    1108  0.3
</CsScore>
</CsoundSynthesizer>
```

**Strengths:** 2000+ opcodes (audio, spectral, physical modeling, ML), runs everywhere (desktop, mobile, web via Emscripten, embedded), 38+ years of development. **Weakness:** Steep learning curve, less modern tooling than SuperCollider."""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 6. Audio File Formats
# ---------------------------------------------------------------------------

def gen_audio_formats() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "Compare WAV, FLAC, MP3, Ogg Vorbis, and AIFF audio file formats.",
        "| Format | Type | Compression | Quality | Typical Use |\n|--------|------|-------------|---------|-------------|\n| **WAV** | Uncompressed (PCM) | None | Lossless | Professional recording, DAW projects |\n| **AIFF** | Uncompressed (PCM) | None | Lossless | Same as WAV, Apple ecosystem |\n| **FLAC** | Compressed | Lossless (~50-60% size) | Identical to original | Archiving, audiophile distribution |\n| **MP3** | Compressed | Lossy (psychoacoustic model) | Good at 256-320 kbps | Streaming, portable |\n| **Ogg Vorbis** | Compressed | Lossy (no patent restrictions) | Better than MP3 at same bitrate | Games, open-source projects |\n| **Opus** | Compressed | Lossy (IETF standard) | Best lossy codec | VoIP, streaming, WebRTC |\n| **AAC** | Compressed | Lossy | Better than MP3 | Apple Music, YouTube, broadcasting |\n\n**Key details:**\n- **WAV/AIFF** are essentially identical containers for PCM data (WAV = Microsoft RIFF, AIFF = Apple). WAV supports 16/24/32-bit int and 32/64-bit float.\n- **FLAC** reconstructs the original PCM data perfectly. Typical compression ratios: 50-70% of original. Metadata via Vorbis Comments.\n- **MP3** uses psychoacoustic masking to discard inaudible data. At 128 kbps, noticeable artifacts on cymbals/reverb tails. 320 kbps is near-transparent.\n- **Opus** at 128 kbps beats MP3 at 256 kbps. Supports 6-510 kbps, 8-48 kHz, speech and music.\n\n**For development:** Use WAV/FLAC for lossless pipelines, Opus for networked streaming, MP3 only for compatibility."
    ))

    pairs.append(msg(
        "How do you read and write WAV files in Python without external libraries?",
        """Python's standard library includes the `wave` module for basic WAV I/O:

```python
import wave
import struct
import math

# --- Writing a WAV file ---
def write_sine_wav(filename: str, freq: float = 440.0,
                    duration: float = 2.0, sr: int = 44100,
                    bits: int = 16):
    n_samples = int(duration * sr)
    max_val = 2 ** (bits - 1) - 1

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)           # mono
        wf.setsampwidth(bits // 8)   # bytes per sample
        wf.setframerate(sr)

        for i in range(n_samples):
            t = i / sr
            sample = math.sin(2 * math.pi * freq * t)
            packed = struct.pack('<h', int(sample * max_val))  # 16-bit LE
            wf.writeframes(packed)

# --- Reading a WAV file ---
def read_wav(filename: str) -> tuple[list[float], int]:
    with wave.open(filename, 'r') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

        fmt = {1: 'b', 2: 'h', 4: 'i'}[sample_width]
        max_val = 2 ** (sample_width * 8 - 1)

        samples = struct.unpack(f'<{n_frames * n_channels}{fmt}', raw)
        normalized = [s / max_val for s in samples]

        return normalized, sr

# Usage
write_sine_wav('test.wav', freq=440, duration=1.0)
samples, sr = read_wav('test.wav')
print(f"Read {len(samples)} samples at {sr} Hz")
```

For production use, prefer `soundfile` (libsndfile wrapper) or `numpy` + `scipy.io.wavfile` for NumPy array I/O."""
    ))

    pairs.append(msg(
        "What is the WAV file header format in detail?",
        "The WAV file uses Microsoft's RIFF container format:\n\n```\nOffset  Size  Description\n------  ----  -----------\n0       4     \"RIFF\" (magic bytes)\n4       4     File size - 8 (uint32 LE)\n8       4     \"WAVE\" (format identifier)\n\n--- fmt chunk ---\n12      4     \"fmt \" (chunk ID, note trailing space)\n16      4     Chunk size (16 for PCM, 18/40 for extensible)\n20      2     Audio format (1 = PCM, 3 = IEEE float, 0xFFFE = extensible)\n22      2     Number of channels (1 = mono, 2 = stereo)\n24      4     Sample rate (e.g., 44100)\n28      4     Byte rate (sample_rate * channels * bits/8)\n32      2     Block align (channels * bits/8)\n34      2     Bits per sample (16, 24, 32)\n\n--- data chunk ---\n36      4     \"data\" (chunk ID)\n40      4     Data size in bytes\n44      ...   Sample data (interleaved for stereo: L R L R ...)\n```\n\n**Sample encoding:**\n- 8-bit: unsigned (0-255, silence = 128)\n- 16-bit: signed little-endian (-32768 to 32767)\n- 24-bit: signed LE, 3 bytes per sample\n- 32-bit float: IEEE 754 (-1.0 to 1.0)\n\n**Additional chunks** (optional):\n- `fact`: Sample count for non-PCM formats\n- `LIST`/`INFO`: Metadata (title, artist, etc.)\n- `cue `: Marker/cue points\n- `smpl`: Loop points, MIDI note, fine tuning\n\n**BWF (Broadcast WAV):** Adds `bext` chunk with timecode, originator, loudness data. Standard in broadcast and film."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 7. Real-time Audio
# ---------------------------------------------------------------------------

def gen_realtime_audio() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What determines audio latency and how do you minimize it?",
        "**Audio latency** = time between input and output. It's the sum of:\n\n**1. Input buffer latency:**\n```\ninput_latency = buffer_size / sample_rate\n```\nAt 256 samples / 48 kHz = 5.33 ms\n\n**2. Processing latency:** Usually negligible (< 1ms) for simple effects\n\n**3. Output buffer latency:** Same as input (double buffering)\n\n**4. Driver overhead:** OS audio stack adds latency\n\n**Total round-trip:**\n```\ntotal = 2 × (buffer_size / sample_rate) + driver_overhead\n```\n\n**Typical values:**\n| Buffer size | @ 44.1 kHz | @ 96 kHz | Use case |\n|-------------|-----------|----------|---------|\n| 32 samples | 1.5 ms | 0.7 ms | Ultra-low latency (risky) |\n| 64 samples | 2.9 ms | 1.3 ms | Live performance |\n| 128 samples | 5.8 ms | 2.7 ms | Recording (comfortable) |\n| 256 samples | 11.6 ms | 5.3 ms | Mixing (safe) |\n| 512 samples | 23.2 ms | 10.7 ms | Heavy plugins |\n| 1024 samples | 46.4 ms | 21.3 ms | Playback only |\n\n**Minimizing latency:**\n1. Use low-latency drivers: **ASIO** (Windows), **Core Audio** (macOS), **JACK** (Linux)\n2. Smaller buffer sizes (but increases CPU load and dropout risk)\n3. Higher sample rate (shorter buffer in ms for same sample count)\n4. Avoid blocking operations in audio callback (no malloc, no mutex, no I/O)\n5. Pin audio thread to a CPU core, set real-time priority\n6. Process in place (avoid memory copies)\n\n**Perceptible latency thresholds:**\n- < 10 ms: Unnoticeable for most performers\n- 10-20 ms: Noticeable for drummers and keyboard players\n- > 20 ms: Clearly perceptible delay\n- > 40 ms: \"Slapback\" echo effect"
    ))

    pairs.append(msg(
        "What are the rules for real-time audio programming (the audio thread)?",
        "The audio callback runs on a **real-time thread** with strict deadline guarantees. Violating these rules causes audible glitches (clicks, pops, dropouts).\n\n**The golden rules — NEVER do these in the audio thread:**\n\n1. **No memory allocation/deallocation:**\n   - No `new`, `delete`, `malloc`, `free`\n   - No STL containers that allocate (`std::vector::push_back`, `std::string`)\n   - Pre-allocate everything in `prepareToPlay()`\n\n2. **No blocking operations:**\n   - No mutex locks (use lock-free queues or atomics)\n   - No file I/O\n   - No network calls\n   - No `sleep()` or `yield()`\n\n3. **No system calls:**\n   - No `printf()` / `std::cout` (can allocate + lock)\n   - No logging\n   - No Objective-C message dispatch (macOS: `@autoreleasepool` can allocate)\n\n4. **No unbounded computation:**\n   - No algorithms with unpredictable runtime\n   - Avoid virtual function calls through deep hierarchies (cache misses)\n\n**Safe patterns:**\n```cpp\n// Lock-free FIFO for parameter changes\nstd::atomic<float> paramValue{0.5f};\n\n// In audio thread:\nfloat val = paramValue.load(std::memory_order_relaxed);\n\n// From UI thread:\nparamValue.store(newValue, std::memory_order_relaxed);\n\n// Lock-free ring buffer for events\njuce::AbstractFifo fifo(1024);\n```\n\n**Testing:** Use tools like `malloc_interpose` (macOS), ThreadSanitizer, or Howard Hinnant's `radsan` to detect real-time violations at runtime."
    ))

    pairs.append(msg(
        "Compare ASIO, Core Audio, JACK, WASAPI, and ALSA audio APIs.",
        "| API | Platform | Latency | Key Features |\n|-----|----------|---------|-------------|\n| **ASIO** | Windows | Very low (2-5ms) | Bypasses Windows audio stack, direct hardware access. Steinberg standard. |\n| **Core Audio** | macOS/iOS | Low (3-10ms) | Built-in, excellent quality. AudioUnit hosting, AUGraph/AVAudioEngine. |\n| **JACK** | Linux/macOS | Very low (1-5ms) | Client-server model, app-to-app routing, session management. Pro audio standard on Linux. |\n| **WASAPI** | Windows | Medium-low (5-15ms) | Exclusive mode bypasses mixer for lower latency. Shared mode adds latency. Built into Windows. |\n| **ALSA** | Linux | Low (2-10ms) | Kernel-level, direct hardware. Used by JACK and PulseAudio internally. Complex API. |\n| **PulseAudio** | Linux | Medium (20-50ms) | Desktop audio server. Mixing, per-app volume. Not for pro audio. |\n| **PipeWire** | Linux | Low (3-10ms) | Modern replacement for both PulseAudio and JACK. Handles audio + video. |\n\n**Choosing:**\n- Cross-platform plugin: Use JUCE or PortAudio (abstracts all backends)\n- macOS app: Core Audio via AVAudioEngine\n- Windows low-latency: ASIO (or WASAPI exclusive as fallback)\n- Linux pro audio: JACK (or PipeWire with JACK compatibility)\n- Linux desktop app: PipeWire (future-proof)\n\n**PortAudio** is the go-to cross-platform C library — wraps ASIO/CoreAudio/ALSA/WASAPI behind a unified callback API."
    ))

    pairs.append(msg(
        "How do you set up a real-time audio callback with PortAudio in C?",
        """```c
#include <portaudio.h>
#include <math.h>
#include <stdio.h>

#define SAMPLE_RATE 44100
#define BUFFER_SIZE 256

typedef struct {
    float phase;
    float frequency;
} SineData;

// Audio callback — runs in real-time thread
static int audioCallback(const void *input, void *output,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo *timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void *userData)
{
    SineData *data = (SineData *)userData;
    float *out = (float *)output;
    float phaseInc = data->frequency / SAMPLE_RATE;

    for (unsigned long i = 0; i < framesPerBuffer; i++) {
        float sample = sinf(2.0f * M_PI * data->phase) * 0.3f;
        *out++ = sample;  // left
        *out++ = sample;  // right
        data->phase += phaseInc;
        if (data->phase >= 1.0f) data->phase -= 1.0f;
    }

    return paContinue;  // or paComplete to stop
}

int main(void) {
    Pa_Initialize();

    SineData data = { .phase = 0.0f, .frequency = 440.0f };

    PaStream *stream;
    Pa_OpenDefaultStream(&stream,
        0,              // no input
        2,              // stereo output
        paFloat32,      // 32-bit float samples
        SAMPLE_RATE,
        BUFFER_SIZE,
        audioCallback,
        &data);

    Pa_StartStream(stream);

    printf("Playing 440 Hz sine. Press Enter to stop.\\n");
    getchar();

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    return 0;
}
```

Compile: `gcc -o sine sine.c -lportaudio -lm`

**Key points:**
- Callback must return quickly (< buffer_size/sample_rate seconds)
- No allocation, I/O, or locking inside the callback
- `paFloat32` samples range -1.0 to 1.0
- Check `statusFlags` for `paOutputUnderflow` (buffer underrun)"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 8. Music Theory for Programmers
# ---------------------------------------------------------------------------

def gen_music_theory() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you convert between MIDI note numbers, frequencies, and note names in code?",
        """**MIDI note → frequency:**
```
frequency = 440 × 2^((note - 69) / 12)
```

**Frequency → MIDI note:**
```
note = 69 + 12 × log2(frequency / 440)
```

**Python implementation:**
```python
import math

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))

def freq_to_midi(freq: float) -> int:
    return round(69 + 12 * math.log2(freq / 440.0))

def midi_to_name(note: int) -> str:
    octave = (note // 12) - 1
    name = NOTE_NAMES[note % 12]
    return f\"{name}{octave}\"

def name_to_midi(name: str) -> int:
    # Parse \"C#4\", \"Db4\", \"A4\"
    import re
    m = re.match(r'([A-G])([#b]?)(\\d+)', name)
    if not m:
        raise ValueError(f\"Invalid note: {name}\")
    base = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[m.group(1)]
    acc = {'#': 1, 'b': -1, '': 0}[m.group(2)]
    octave = int(m.group(3))
    return (octave + 1) * 12 + base + acc

# Examples
print(midi_to_freq(69))      # 440.0 (A4)
print(midi_to_freq(60))      # 261.63 (Middle C)
print(midi_to_name(60))      # C4
print(freq_to_midi(440.0))   # 69
print(name_to_midi('C#4'))   # 61
```

**Key reference points:**
| MIDI | Note | Frequency |\n| 21 | A0 | 27.5 Hz |\n| 60 | C4 (Middle C) | 261.63 Hz |\n| 69 | A4 (Concert A) | 440 Hz |\n| 108 | C8 | 4186 Hz |"""
    ))

    pairs.append(msg(
        "How do scales and modes work programmatically?",
        """Scales are defined by their **interval pattern** (semitone steps between notes).

```python
# Interval patterns (in semitones)
SCALES = {
    'major':            [2, 2, 1, 2, 2, 2, 1],
    'natural_minor':    [2, 1, 2, 2, 1, 2, 2],
    'harmonic_minor':   [2, 1, 2, 2, 1, 3, 1],
    'melodic_minor':    [2, 1, 2, 2, 2, 2, 1],
    'pentatonic_major': [2, 2, 3, 2, 3],
    'pentatonic_minor': [3, 2, 2, 3, 2],
    'blues':            [3, 2, 1, 1, 3, 2],
    'chromatic':        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'whole_tone':       [2, 2, 2, 2, 2, 2],
    'diminished':       [2, 1, 2, 1, 2, 1, 2, 1],  # octatonic
}

# Modes are rotations of the major scale
MODES = {
    'ionian':     0,  # major
    'dorian':     1,
    'phrygian':   2,
    'lydian':     3,
    'mixolydian': 4,
    'aeolian':    5,  # natural minor
    'locrian':    6,
}

def get_scale(root: int, scale_name: str) -> list[int]:
    \"\"\"Get MIDI notes for a scale starting from root.\"\"\"
    intervals = SCALES[scale_name]
    notes = [root]
    for interval in intervals:
        notes.append(notes[-1] + interval)
    return notes

def get_mode(root: int, mode_name: str) -> list[int]:
    \"\"\"Get MIDI notes for a mode starting from root.\"\"\"
    rotation = MODES[mode_name]
    major = SCALES['major']
    # Rotate the interval pattern
    intervals = major[rotation:] + major[:rotation]
    notes = [root]
    for interval in intervals:
        notes.append(notes[-1] + interval)
    return notes

# C major scale
print(get_scale(60, 'major'))  # [60, 62, 64, 65, 67, 69, 71, 72]
# D dorian (same notes as C major, starting on D)
print(get_mode(62, 'dorian'))  # [62, 64, 65, 67, 69, 71, 72, 74]
```

**Mode characteristics:**
| Mode | Feel | Example use |
|------|------|-------------|
| Ionian | Bright, happy | Pop, classical |
| Dorian | Minor but uplifting | Jazz, funk, Santana |
| Phrygian | Dark, exotic | Flamenco, metal |
| Lydian | Dreamy, ethereal | Film scores, Satriani |
| Mixolydian | Bluesy major | Rock, blues, folk |
| Aeolian | Sad, dark | Rock ballads, minor key pop |
| Locrian | Unstable, dissonant | Rarely used as a key |"""
    ))

    pairs.append(msg(
        "How do you build chords programmatically from scale degrees?",
        """```python
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord types defined by semitone intervals from root
CHORD_TYPES = {
    'major':      [0, 4, 7],
    'minor':      [0, 3, 7],
    'diminished': [0, 3, 6],
    'augmented':  [0, 4, 8],
    'sus2':       [0, 2, 7],
    'sus4':       [0, 5, 7],
    'dom7':       [0, 4, 7, 10],
    'maj7':       [0, 4, 7, 11],
    'min7':       [0, 3, 7, 10],
    'dim7':       [0, 3, 6, 9],
    'min7b5':     [0, 3, 6, 10],  # half-diminished
    'add9':       [0, 4, 7, 14],
    'min9':       [0, 3, 7, 10, 14],
    'dom9':       [0, 4, 7, 10, 14],
    '6':          [0, 4, 7, 9],
    'min6':       [0, 3, 7, 9],
}

def build_chord(root: int, chord_type: str) -> list[int]:
    \"\"\"Build a chord from root MIDI note.\"\"\"
    return [root + interval for interval in CHORD_TYPES[chord_type]]

def chord_name(root: int, chord_type: str) -> str:
    return f\"{NOTE_NAMES[root % 12]}{'' if chord_type == 'major' else chord_type}\"

def diatonic_chords(root: int, scale: str = 'major') -> list[tuple]:
    \"\"\"Build all diatonic triads from a scale.\"\"\"
    scale_intervals = {
        'major': [2, 2, 1, 2, 2, 2, 1],
        'natural_minor': [2, 1, 2, 2, 1, 2, 2],
    }
    # Build scale notes across 2 octaves
    intervals = scale_intervals[scale]
    notes = [root]
    for _ in range(2):
        for i in intervals:
            notes.append(notes[-1] + i)

    # Stack thirds (every other scale degree)
    chords = []
    for i in range(7):
        triad = [notes[i], notes[i+2], notes[i+4]]
        # Determine chord quality from intervals
        i1 = triad[1] - triad[0]
        i2 = triad[2] - triad[1]
        if i1 == 4 and i2 == 3:
            quality = 'major'
        elif i1 == 3 and i2 == 4:
            quality = 'minor'
        elif i1 == 3 and i2 == 3:
            quality = 'diminished'
        else:
            quality = 'augmented'
        chords.append((NOTE_NAMES[triad[0] % 12], quality, triad))
    return chords

# C major diatonic chords: C, Dm, Em, F, G, Am, Bdim
for name, quality, notes in diatonic_chords(60, 'major'):
    print(f\"{name} {quality}: {notes}\")
```"""
    ))

    pairs.append(msg(
        "What is BPM and how do you convert between BPM, beat duration, and sample counts?",
        """**BPM** (Beats Per Minute) defines the tempo of music.

**Conversions:**
```python
def bpm_to_beat_duration(bpm: float) -> float:
    \"\"\"Beat duration in seconds.\"\"\"
    return 60.0 / bpm

def beat_to_samples(bpm: float, sr: int, note_value: float = 1.0) -> int:
    \"\"\"Samples per beat (or fraction).
    note_value: 1.0 = quarter, 0.5 = eighth, 2.0 = half, etc.
    \"\"\"
    beat_sec = 60.0 / bpm * note_value
    return int(beat_sec * sr)

def delay_time_ms(bpm: float, note_division: str) -> float:
    \"\"\"Tempo-synced delay time in milliseconds.\"\"\"
    beat_ms = 60000.0 / bpm
    divisions = {
        '1/1': 4.0,      # whole note
        '1/2': 2.0,      # half note
        '1/4': 1.0,      # quarter note
        '1/8': 0.5,      # eighth note
        '1/16': 0.25,    # sixteenth note
        '1/4T': 2/3,     # quarter triplet
        '1/8T': 1/3,     # eighth triplet
        '1/16T': 1/6,    # sixteenth triplet
        '1/4.': 1.5,     # dotted quarter
        '1/8.': 0.75,    # dotted eighth
    }
    return beat_ms * divisions[note_division]

# Examples at 120 BPM
bpm = 120
print(f\"Beat duration: {bpm_to_beat_duration(bpm):.3f}s\")  # 0.500s
print(f\"Samples per beat (44.1k): {beat_to_samples(bpm, 44100)}\")  # 22050
print(f\"1/4 delay: {delay_time_ms(bpm, '1/4'):.1f}ms\")   # 500.0ms
print(f\"1/8 delay: {delay_time_ms(bpm, '1/8'):.1f}ms\")   # 250.0ms
print(f\"1/8T delay: {delay_time_ms(bpm, '1/8T'):.1f}ms\") # 166.7ms
print(f\"1/4. delay: {delay_time_ms(bpm, '1/4.'):.1f}ms\")  # 750.0ms
```

**Time signature:**
- 4/4: 4 quarter notes per bar (most common)
- 3/4: 3 quarter notes per bar (waltz)
- 6/8: 6 eighth notes per bar (compound duple)
- 7/8: 7 eighth notes per bar (progressive/Balkan)

**Bar duration** = beats_per_bar × (60/BPM) seconds.
At 120 BPM in 4/4: bar = 4 × 0.5 = 2.0 seconds."""
    ))

    pairs.append(msg(
        "What are intervals in music and how do you calculate them?",
        """An **interval** is the distance between two pitches, measured in semitones.

```python
INTERVALS = {
    0: ('P1', 'Perfect Unison'),
    1: ('m2', 'Minor 2nd'),
    2: ('M2', 'Major 2nd'),
    3: ('m3', 'Minor 3rd'),
    4: ('M3', 'Major 3rd'),
    5: ('P4', 'Perfect 4th'),
    6: ('A4/d5', 'Tritone'),
    7: ('P5', 'Perfect 5th'),
    8: ('m6', 'Minor 6th'),
    9: ('M6', 'Major 6th'),
    10: ('m7', 'Minor 7th'),
    11: ('M7', 'Major 7th'),
    12: ('P8', 'Perfect Octave'),
}

def interval_name(note1: int, note2: int) -> str:
    semitones = abs(note2 - note1) % 12
    return INTERVALS[semitones][1]

def interval_ratio(semitones: int) -> float:
    \"\"\"Frequency ratio for an interval in equal temperament.\"\"\"
    return 2.0 ** (semitones / 12.0)

# Consonance ranking (from most consonant):
# P1 (1:1) > P8 (2:1) > P5 (3:2) > P4 (4:3) > M3 (5:4) > m3 (6:5)
# > M6 (5:3) > m6 (8:5) > M2 (9:8) > m7 (16:9) > M7 > m2 > tritone

# Just intonation ratios vs equal temperament
JUST_RATIOS = {
    'P1': 1/1, 'M2': 9/8, 'M3': 5/4, 'P4': 4/3,
    'P5': 3/2, 'M6': 5/3, 'M7': 15/8, 'P8': 2/1,
}
```

**Interval quality:**
- **Perfect** (P): Unison, 4th, 5th, octave — strong consonance
- **Major** (M): 2nd, 3rd, 6th, 7th — natural scale intervals
- **Minor** (m): One semitone less than major
- **Augmented** (A): One semitone more than perfect/major
- **Diminished** (d): One semitone less than perfect/minor

**Compound intervals** (> octave): 9th = octave + 2nd, 11th = octave + 4th, 13th = octave + 6th."""
    ))

    return pairs


# ---------------------------------------------------------------------------
# Additional categories for more coverage
# ---------------------------------------------------------------------------

def gen_daw_integration() -> list[dict]:
    """DAW concepts and plugin development patterns."""
    pairs = []

    pairs.append(msg(
        "What are the main audio plugin formats (VST3, AU, AAX, CLAP) and their differences?",
        "| Format | Developer | Platform | Host compatibility |\n|--------|-----------|----------|-------------------|\n| **VST3** | Steinberg | Win/Mac/Linux | Most DAWs (Cubase, Ableton, FL Studio, Reaper, Bitwig) |\n| **AU** (AudioUnit) | Apple | macOS/iOS only | Logic Pro, GarageBand, Ableton, Final Cut |\n| **AAX** | Avid | Win/Mac | Pro Tools only |\n| **CLAP** | Free/Open | Win/Mac/Linux | Bitwig, Reaper (growing support) |\n| **LV2** | Open | Linux (primarily) | Ardour, Carla, Qtractor |\n\n**Key differences:**\n- **VST3** replaced VST2 (discontinued licensing). Supports side-chain, MIDI, resizable UI, note expression. C++ SDK.\n- **AU** uses Apple's component architecture. Required for Logic Pro. Objective-C/Swift bridge. Supports AUv3 (sandboxed, iOS compatible).\n- **AAX** uses iLok protection. Required for Pro Tools. DSP and Native variants.\n- **CLAP** is the newest (2022), open-source, designed to fix VST3 shortcomings. Features: thread-safe parameter automation, polyphonic modulation, better extension system.\n\n**For developers:** Use JUCE or iPlug2 to target multiple formats from one codebase. CLAP is promising but still gaining adoption."
    ))

    pairs.append(msg(
        "How does parameter automation work in audio plugins?",
        "**Parameter automation** allows DAWs to record and playback parameter changes over time.\n\n**Plugin side (JUCE example):**\n```cpp\nclass MyProcessor : public juce::AudioProcessor {\n    juce::AudioProcessorValueTreeState parameters;\n\n    MyProcessor()\n        : parameters(*this, nullptr, \"PARAMS\",\n            { std::make_unique<juce::AudioParameterFloat>(\n                juce::ParameterID{\"cutoff\", 1},\n                \"Filter Cutoff\",\n                juce::NormalisableRange<float>(20.f, 20000.f, 0.f, 0.3f),\n                1000.f  // default\n              ),\n              std::make_unique<juce::AudioParameterChoice>(\n                juce::ParameterID{\"type\", 1},\n                \"Filter Type\",\n                juce::StringArray{\"Low Pass\", \"High Pass\", \"Band Pass\"},\n                0  // default index\n              )\n            })\n    {}\n\n    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override {\n        // Read automated parameter (thread-safe)\n        float cutoff = parameters.getRawParameterValue(\"cutoff\")->load();\n        // Use cutoff in DSP...\n    }\n};\n```\n\n**DAW side:**\n1. Host calls `setParameter()` or updates the parameter's atomic value\n2. Parameter changes can be **sample-accurate** — the host provides exact timestamps\n3. Values are **normalized** (0.0-1.0) with the plugin doing the denormalization\n4. Automation modes: Read, Write, Touch, Latch\n\n**Thread safety:**\n- Parameters are read on the audio thread and written from the UI/host thread\n- Use `std::atomic<float>` or the framework's thread-safe parameter system\n- Never lock a mutex to read parameters in `processBlock()`"
    ))

    pairs.append(msg(
        "What is a sample-accurate automation system and why does it matter?",
        "**Sample-accurate automation** means parameter changes take effect at the exact sample position within a buffer, not just per-buffer.\n\n**Why it matters:**\n- At 512 samples/buffer (11.6ms at 44.1kHz), per-buffer updates cause audible zipper noise on fast parameter changes\n- Sample-accurate automation prevents clicks when automating volume, pan, or filter cutoff\n\n**Implementation pattern:**\n```cpp\nvoid processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) {\n    auto* gainParam = parameters.getRawParameterValue(\"gain\");\n    float targetGain = gainParam->load();\n\n    // Option 1: Simple smoothing (good enough for most cases)\n    for (int i = 0; i < buffer.getNumSamples(); ++i) {\n        currentGain += (targetGain - currentGain) * 0.001f;  // one-pole smooth\n        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {\n            buffer.getWritePointer(ch)[i] *= currentGain;\n        }\n    }\n\n    // Option 2: Sub-block processing with automation events\n    // Split buffer at automation change points, process each sub-block\n    // with the correct parameter value\n}\n```\n\n**Parameter smoothing types:**\n- **Linear ramp:** `value += (target - value) / rampSamples` — constant time, good for volume\n- **Exponential (one-pole):** `value += (target - value) * coeff` — smooth, asymptotic\n- **JUCE SmoothedValue:** Built-in, configurable type and ramp time\n\n```cpp\njuce::SmoothedValue<float> smoothGain;\nsmoothGain.reset(sampleRate, 0.02);  // 20ms ramp\nsmoothGain.setTargetValue(newGain);\n// In processBlock: smoothGain.getNextValue() per sample\n```"
    ))

    return pairs


def gen_spatial_audio() -> list[dict]:
    """Spatial audio and immersive sound."""
    pairs = []

    pairs.append(msg(
        "How does binaural audio work and how do you implement HRTF-based spatialization?",
        "**Binaural audio** creates 3D sound positioning using headphones by simulating how sounds reach each ear differently.\n\n**HRTF** (Head-Related Transfer Function) captures how the head, pinnae (outer ears), and torso modify sound from different directions.\n\n**Key cues:**\n- **ITD** (Interaural Time Difference): Sound arrives at the closer ear first (up to ~0.7ms)\n- **ILD** (Interaural Level Difference): Sound is louder at the closer ear (especially > 1kHz)\n- **Spectral cues:** Pinna reflections create direction-dependent frequency coloring (especially for elevation)\n\n**Implementation:**\n```python\nimport numpy as np\nfrom scipy.signal import fftconvolve\n\nclass BinauralPanner:\n    def __init__(self, hrtf_left: dict, hrtf_right: dict, sr: int):\n        \"\"\"\n        hrtf_left/right: dict mapping (azimuth, elevation) to impulse response arrays\n        \"\"\"\n        self.hrtf_l = hrtf_left\n        self.hrtf_r = hrtf_right\n        self.sr = sr\n\n    def spatialize(self, mono_signal: np.ndarray,\n                    azimuth: int, elevation: int = 0) -> np.ndarray:\n        \"\"\"Place a mono source at (azimuth, elevation) in 3D space.\"\"\"\n        ir_l = self.hrtf_l[(azimuth, elevation)]\n        ir_r = self.hrtf_r[(azimuth, elevation)]\n\n        left = fftconvolve(mono_signal, ir_l, mode='same')\n        right = fftconvolve(mono_signal, ir_r, mode='same')\n\n        return np.column_stack([left, right])\n```\n\n**HRTF datasets:** MIT KEMAR, CIPIC, SADIE II, ARI. Personalized HRTFs via ear scanning or ML estimation.\n\n**Ambisonics** is an alternative: encode audio into spherical harmonics (B-format), decode to any speaker layout or binaural. Orders: 1st (4 channels), 3rd (16 channels), 7th (64 channels)."
    ))

    pairs.append(msg(
        "What is ambisonics and how does it represent 3D audio?",
        "**Ambisonics** encodes sound fields using **spherical harmonics**, independent of any specific speaker layout.\n\n**B-Format (First Order Ambisonics - FOA):**\n4 channels: W (omnidirectional), X (front-back), Y (left-right), Z (up-down)\n\n**Encoding a mono source at (azimuth, elevation):**\n```python\nimport numpy as np\n\ndef encode_foa(mono: np.ndarray, azimuth_deg: float,\n                elevation_deg: float) -> np.ndarray:\n    \"\"\"Encode mono source to First Order Ambisonics B-format.\"\"\"\n    az = np.radians(azimuth_deg)\n    el = np.radians(elevation_deg)\n\n    W = mono * (1.0 / np.sqrt(2))  # omnidirectional\n    X = mono * np.cos(az) * np.cos(el)  # front-back\n    Y = mono * np.sin(az) * np.cos(el)  # left-right\n    Z = mono * np.sin(el)               # up-down\n\n    return np.column_stack([W, X, Y, Z])\n\ndef decode_stereo(bformat: np.ndarray) -> np.ndarray:\n    \"\"\"Simple stereo decode from B-format.\"\"\"\n    W, X, Y, Z = bformat.T\n    left  = W + 0.5 * X + 0.5 * Y\n    right = W + 0.5 * X - 0.5 * Y\n    return np.column_stack([left, right])\n```\n\n**Higher Order Ambisonics (HOA):**\n- Order N requires (N+1)^2 channels\n- Higher order = sharper spatial resolution\n- 1st order: 4ch, 3rd: 16ch, 5th: 36ch, 7th: 64ch\n\n**Advantages:**\n- Speaker-layout independent (decode to any array)\n- Rotatable sound field (great for VR head tracking)\n- Scalable resolution\n\n**Tools:** IEM Plugin Suite, SPARTA, ambix, AmbiX-based workflows in Reaper."
    ))

    return pairs


def gen_music_programming() -> list[dict]:
    """Music programming patterns and algorithms."""
    pairs = []

    pairs.append(msg(
        "How do you implement a step sequencer in code?",
        """A **step sequencer** triggers notes in a repeating pattern, typically 16 steps.

```python
import time
import numpy as np

class StepSequencer:
    def __init__(self, bpm: float = 120, steps: int = 16,
                 step_division: int = 4):
        \"\"\"
        step_division: steps per beat (4 = 16th notes at 16 steps = 1 bar of 4/4)
        \"\"\"
        self.bpm = bpm
        self.steps = steps
        self.step_division = step_division
        self.current_step = 0

        # Pattern: list of dicts per step
        self.pattern = [
            {'active': False, 'note': 60, 'velocity': 100, 'gate': 0.5}
            for _ in range(steps)
        ]

    @property
    def step_duration(self) -> float:
        \"\"\"Duration of one step in seconds.\"\"\"
        return 60.0 / (self.bpm * self.step_division)

    def set_step(self, step: int, note: int = 60,
                  velocity: int = 100, gate: float = 0.5):
        self.pattern[step] = {
            'active': True, 'note': note,
            'velocity': velocity, 'gate': gate,
        }

    def tick(self) -> dict | None:
        \"\"\"Advance one step. Returns note event or None.\"\"\"
        step = self.pattern[self.current_step]
        self.current_step = (self.current_step + 1) % self.steps
        if step['active']:
            return {
                'note': step['note'],
                'velocity': step['velocity'],
                'duration': self.step_duration * step['gate'],
            }
        return None

    def get_pattern_string(self) -> str:
        return ''.join('X' if s['active'] else '.' for s in self.pattern)

# Classic 4-on-the-floor kick pattern
seq = StepSequencer(bpm=120, steps=16)
for i in [0, 4, 8, 12]:  # every beat
    seq.set_step(i, note=36)  # C1 = kick

# Hi-hat on off-beats
for i in [2, 6, 10, 14]:
    seq.set_step(i, note=42, velocity=80)  # closed hi-hat

print(seq.get_pattern_string())  # X.X.X.X.X.X.X.X.
```

**Advanced features:** Probability per step, swing (delay even steps), ratcheting (subdivide a step), parameter locks (per-step filter/effect values), polyrhythm (different step counts per track)."""
    ))

    pairs.append(msg(
        "How do you generate random melodies that sound musical?",
        """Use **constrained randomness** — limit choices to scale degrees, apply contour rules, and use rhythm patterns.

```python
import random
import numpy as np

class MelodyGenerator:
    SCALES = {
        'major':          [0, 2, 4, 5, 7, 9, 11],
        'minor':          [0, 2, 3, 5, 7, 8, 10],
        'pentatonic':     [0, 2, 4, 7, 9],
        'blues':          [0, 3, 5, 6, 7, 10],
        'dorian':         [0, 2, 3, 5, 7, 9, 10],
    }

    RHYTHM_PATTERNS = [
        [1, 1, 1, 1],           # steady quarter notes
        [2, 1, 1],              # long-short-short
        [1, 1, 2],              # short-short-long
        [1.5, 0.5, 1, 1],      # dotted eighth + sixteenth
        [1, 0.5, 0.5, 1, 1],   # syncopated
    ]

    def __init__(self, root: int = 60, scale: str = 'pentatonic',
                 octave_range: int = 2):
        self.root = root
        self.scale_notes = []
        intervals = self.SCALES[scale]
        for octave in range(-1, octave_range):
            for interval in intervals:
                note = root + octave * 12 + interval
                if 36 <= note <= 96:  # reasonable range
                    self.scale_notes.append(note)

    def generate(self, num_bars: int = 4, beats_per_bar: int = 4) -> list[dict]:
        melody = []
        current_note_idx = len(self.scale_notes) // 2  # start in middle

        for bar in range(num_bars):
            # Pick a rhythm pattern for this bar
            rhythm = random.choice(self.RHYTHM_PATTERNS)
            beat_pos = 0

            for duration in rhythm:
                # Prefer stepwise motion (intervals of 1-2 scale degrees)
                step = random.choices(
                    [-2, -1, 0, 1, 2, 3],
                    weights=[0.1, 0.3, 0.1, 0.3, 0.15, 0.05]
                )[0]
                current_note_idx = max(0, min(len(self.scale_notes) - 1,
                                               current_note_idx + step))

                # Occasional rest (10% chance)
                if random.random() < 0.1:
                    melody.append({'note': None, 'duration': duration, 'velocity': 0})
                else:
                    velocity = random.randint(70, 110)
                    # Accent on beat 1
                    if beat_pos == 0:
                        velocity = min(127, velocity + 20)
                    melody.append({
                        'note': self.scale_notes[current_note_idx],
                        'duration': duration,
                        'velocity': velocity,
                    })
                beat_pos += duration

        return melody

gen = MelodyGenerator(root=60, scale='pentatonic')
melody = gen.generate(num_bars=4)
```

**Musicality tips:**
- Constrain to a scale (pentatonic is safest)
- Prefer stepwise motion (adjacent scale degrees)
- End phrases on the root or 5th
- Use rhythmic repetition/variation
- Apply contour rules (arch shapes, rising/falling phrases)"""
    ))

    pairs.append(msg(
        "How do you implement a basic audio mixer in code?",
        """```python
import numpy as np

class AudioMixer:
    \"\"\"Multi-channel audio mixer with gain, pan, mute, and solo.\"\"\"

    def __init__(self, num_channels: int, buffer_size: int, sr: int):
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.sr = sr

        # Per-channel state
        self.gain = np.ones(num_channels)       # 0.0 to 1.0+
        self.pan = np.zeros(num_channels)        # -1.0 (L) to 1.0 (R)
        self.mute = np.zeros(num_channels, dtype=bool)
        self.solo = np.zeros(num_channels, dtype=bool)

        # Master
        self.master_gain = 1.0

    def set_gain_db(self, channel: int, db: float):
        self.gain[channel] = 10 ** (db / 20.0)

    def set_pan(self, channel: int, pan: float):
        \"\"\"Pan: -1.0 = full left, 0.0 = center, 1.0 = full right.\"\"\"
        self.pan[channel] = max(-1.0, min(1.0, pan))

    def _pan_law(self, pan: float) -> tuple[float, float]:
        \"\"\"Constant-power pan law (-3 dB center).\"\"\"
        angle = (pan + 1) * np.pi / 4  # 0 to pi/2
        return np.cos(angle), np.sin(angle)

    def mix(self, inputs: list[np.ndarray]) -> np.ndarray:
        \"\"\"Mix mono input channels to stereo output.\"\"\"
        assert len(inputs) == self.num_channels
        output = np.zeros((self.buffer_size, 2))

        any_solo = np.any(self.solo)

        for ch in range(self.num_channels):
            if self.mute[ch]:
                continue
            if any_solo and not self.solo[ch]:
                continue

            signal = inputs[ch] * self.gain[ch]
            left_gain, right_gain = self._pan_law(self.pan[ch])

            output[:, 0] += signal * left_gain
            output[:, 1] += signal * right_gain

        return output * self.master_gain

# Usage
mixer = AudioMixer(num_channels=4, buffer_size=512, sr=44100)
mixer.set_gain_db(0, -3.0)   # channel 0: -3 dB
mixer.set_pan(1, -0.5)       # channel 1: panned left
mixer.mute[2] = True         # mute channel 2

inputs = [np.random.randn(512) * 0.5 for _ in range(4)]
stereo_out = mixer.mix(inputs)
```

**Constant-power pan law** ensures no perceived volume change when panning — center is -3 dB per channel so L+R sum stays consistent. Alternative: linear pan law (simpler but center is louder)."""
    ))

    return pairs


def gen_audio_analysis() -> list[dict]:
    """Audio analysis and feature extraction."""
    pairs = []

    pairs.append(msg(
        "How do you detect the pitch of an audio signal?",
        "**Pitch detection** finds the fundamental frequency (f0) of a signal. Key algorithms:\n\n**1. Autocorrelation:**\n```python\nimport numpy as np\n\ndef autocorrelation_pitch(signal: np.ndarray, sr: int,\n                          min_freq: float = 80,\n                          max_freq: float = 1000) -> float:\n    # Compute autocorrelation\n    corr = np.correlate(signal, signal, mode='full')\n    corr = corr[len(corr)//2:]  # keep positive lags only\n\n    # Search range\n    min_lag = int(sr / max_freq)\n    max_lag = int(sr / min_freq)\n\n    # Find peak in valid range\n    search = corr[min_lag:max_lag]\n    if len(search) == 0:\n        return 0.0\n    peak_lag = np.argmax(search) + min_lag\n    return sr / peak_lag\n```\n\n**2. YIN algorithm** (de Cheveigné, 2002):\n- Improved autocorrelation with cumulative mean normalization\n- More robust to octave errors\n- Used in Aubio, librosa\n\n**3. pYIN** (probabilistic YIN):\n- Multiple pitch candidates per frame + HMM smoothing\n- Best for monophonic music\n\n**4. CREPE** (neural network, 2018):\n- CNN trained on synthesized data\n- State-of-the-art accuracy, especially for noisy signals\n- `pip install crepe` → `crepe.predict(audio, sr)`\n\n**Using librosa:**\n```python\nimport librosa\n\ny, sr = librosa.load('audio.wav')\nf0, voiced, _ = librosa.pyin(y, fmin=80, fmax=1000, sr=sr)\n```\n\n**Polyphonic pitch:** Use NMF (Non-negative Matrix Factorization), harmonic-percussive separation, or deep learning (Omnizart, basic-pitch)."
    ))

    pairs.append(msg(
        "How do you compute a spectrogram and mel spectrogram?",
        """**Spectrogram:** Time-frequency representation via STFT (Short-Time Fourier Transform).

```python
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('audio.wav', sr=22050)

# --- Linear spectrogram ---
S = librosa.stft(y, n_fft=2048, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, hop_length=512,
                          x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# --- Mel spectrogram ---
S_mel = librosa.feature.melspectrogram(
    y=y, sr=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=128,      # number of mel bands
    fmin=20,
    fmax=8000,
)
S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_mel_db, sr=sr, hop_length=512,
                          x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()
```

**Key parameters:**
- `n_fft`: FFT window size (2048 = ~93ms at 22050 Hz, 43 Hz resolution)
- `hop_length`: Step between frames (512 = 75% overlap with 2048 FFT)
- `n_mels`: Number of mel-scale frequency bands (64-128 typical)

**Mel scale** approximates human perception — logarithmic above ~1 kHz, linear below. Mel spectrograms are the standard input for ML audio models (speech recognition, music classification).

**MFCCs** (Mel-Frequency Cepstral Coefficients) = DCT of the mel spectrogram log-energy. 13-20 coefficients capture timbral shape:
```python
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
```"""
    ))

    pairs.append(msg(
        "How do you implement beat detection / onset detection?",
        """**Onset detection** finds the start of musical events (beats, notes, transients).

```python
import librosa
import numpy as np

y, sr = librosa.load('audio.wav')

# --- Method 1: librosa onset detection ---
onset_frames = librosa.onset.onset_detect(
    y=y, sr=sr,
    hop_length=512,
    backtrack=True,     # snap to nearest preceding minimum
    units='frames'
)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

# --- Method 2: spectral flux (manual) ---
def spectral_flux(y, sr, hop_length=512, n_fft=2048):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # Half-wave rectified difference
    flux = np.zeros(S.shape[1])
    for i in range(1, S.shape[1]):
        diff = S[:, i] - S[:, i-1]
        flux[i] = np.sum(np.maximum(0, diff))  # only increases
    return flux

flux = spectral_flux(y, sr)
# Peak-pick above threshold
threshold = np.mean(flux) + 2 * np.std(flux)
onsets = np.where(flux > threshold)[0]

# --- Beat tracking ---
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(f\"Estimated tempo: {tempo:.1f} BPM\")
print(f\"Beat times: {beat_times[:8]}\")

# --- Method 3: Energy-based (for percussive content) ---
def energy_onset(y, sr, frame_size=1024, hop=512):
    energy = np.array([
        np.sum(y[i:i+frame_size]**2)
        for i in range(0, len(y)-frame_size, hop)
    ])
    # Derivative of energy
    diff = np.diff(energy)
    diff = np.maximum(0, diff)  # only rises
    return diff
```

**Onset detection functions:**
- **Spectral flux:** Change in spectral magnitude (general purpose)
- **High-frequency content:** Weighted spectral flux (good for percussive)
- **Complex domain:** Phase + magnitude deviation (most sensitive)
- **RMS energy:** Simple but effective for drums"""
    ))

    return pairs


def gen_midi_programming() -> list[dict]:
    """MIDI programming with Python libraries."""
    pairs = []

    pairs.append(msg(
        "How do you read and write MIDI files in Python using mido?",
        """```python
import mido

# --- Creating a MIDI file ---
mid = mido.MidiFile(ticks_per_beat=480)

# Track 0: tempo and time signature
track0 = mido.MidiTrack()
mid.tracks.append(track0)
track0.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
track0.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))

# Track 1: melody
track1 = mido.MidiTrack()
mid.tracks.append(track1)
track1.append(mido.MetaMessage('track_name', name='Melody'))
track1.append(mido.Message('program_change', program=0, channel=0))  # piano

# C major scale: C4-D4-E4-F4-G4-A4-B4-C5
notes = [60, 62, 64, 65, 67, 69, 71, 72]
for note in notes:
    track1.append(mido.Message('note_on', note=note, velocity=80, time=0))
    track1.append(mido.Message('note_off', note=note, time=480))  # quarter note

track1.append(mido.MetaMessage('end_of_track'))
mid.save('scale.mid')

# --- Reading a MIDI file ---
mid = mido.MidiFile('scale.mid')
print(f\"Type: {mid.type}, Ticks/beat: {mid.ticks_per_beat}\")
print(f\"Duration: {mid.length:.2f} seconds\")

for i, track in enumerate(mid.tracks):
    print(f\"Track {i}: {track.name}\")
    for msg in track:
        if msg.type == 'note_on' and msg.velocity > 0:
            print(f\"  Note On: {msg.note} vel={msg.velocity} time={msg.time}\")

# --- Real-time MIDI I/O ---
# List available ports
print(mido.get_input_names())
print(mido.get_output_names())

# Send MIDI messages
with mido.open_output() as port:
    port.send(mido.Message('note_on', note=60, velocity=100))
    import time; time.sleep(0.5)
    port.send(mido.Message('note_off', note=60))

# Receive MIDI messages
with mido.open_input() as port:
    for msg in port:
        print(msg)
```

`mido` is the standard Python MIDI library. For real-time with low latency, consider `python-rtmidi` (C++ bindings)."""
    ))

    pairs.append(msg(
        "How do you process MIDI in real-time to build an arpeggiator?",
        """An **arpeggiator** takes held chord notes and plays them sequentially in a pattern.

```python
import mido
import time
import threading

class Arpeggiator:
    PATTERNS = {
        'up': lambda notes: sorted(notes),
        'down': lambda notes: sorted(notes, reverse=True),
        'updown': lambda notes: sorted(notes) + sorted(notes, reverse=True)[1:-1],
        'random': lambda notes: sorted(notes),  # shuffled at playback
    }

    def __init__(self, output_port: str, bpm: float = 120,
                 division: int = 4, pattern: str = 'up', octaves: int = 1):
        self.port = mido.open_output(output_port)
        self.bpm = bpm
        self.division = division  # steps per beat
        self.pattern_type = pattern
        self.octaves = octaves
        self.held_notes = set()
        self.running = False
        self._current_note = None

    @property
    def step_time(self) -> float:
        return 60.0 / (self.bpm * self.division)

    def note_on(self, note: int, velocity: int):
        self.held_notes.add((note, velocity))

    def note_off(self, note: int):
        self.held_notes = {(n, v) for n, v in self.held_notes if n != note}

    def _get_sequence(self) -> list[tuple[int, int]]:
        if not self.held_notes:
            return []
        base = list(self.held_notes)
        pattern_fn = self.PATTERNS[self.pattern_type]
        ordered = [(n, v) for n, v in pattern_fn([n for n, v in base])]
        # Match velocities
        vel_map = {n: v for n, v in base}
        result = [(n, vel_map.get(n, 100)) for n in [n for n, _ in ordered]]

        # Extend across octaves
        extended = []
        for octave in range(self.octaves):
            for note, vel in result:
                extended.append((note + octave * 12, vel))
        return extended

    def _play_loop(self):
        step_idx = 0
        while self.running:
            seq = self._get_sequence()
            if seq:
                # Turn off previous note
                if self._current_note is not None:
                    self.port.send(mido.Message('note_off',
                                                note=self._current_note))
                # Play next note
                note, vel = seq[step_idx % len(seq)]
                self.port.send(mido.Message('note_on',
                                            note=note, velocity=vel))
                self._current_note = note
                step_idx += 1
            time.sleep(self.step_time)

    def start(self):
        self.running = True
        threading.Thread(target=self._play_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self._current_note:
            self.port.send(mido.Message('note_off', note=self._current_note))
```

**Extensions:** Swing, gate length control, tie mode, latch mode (notes stay after release), chord mode (play full chord on each step)."""
    ))

    return pairs


def gen_audio_misc() -> list[dict]:
    """Miscellaneous audio topics for coverage."""
    pairs = []

    pairs.append(msg(
        "What is sidechain compression and how is it commonly used?",
        "**Sidechain compression** uses one signal (the **key** or **sidechain**) to control compression on a different signal.\n\n**How it works:**\n1. The compressor's **detector** listens to the sidechain input (e.g., kick drum)\n2. When the sidechain exceeds the threshold, compression is applied to the **main** signal (e.g., bass or pad)\n3. The main signal ducks in volume when the sidechain is loud\n\n**Common uses:**\n- **Kick-bass ducking:** Bass drops in volume on every kick hit, preventing low-end mud and creating rhythmic pumping (EDM, pop)\n- **De-essing:** Sidechain from a high-shelf EQ'd copy of vocals → compress only sibilant frequencies\n- **Voiceover ducking:** Music ducks when narration plays (podcasts, broadcast)\n- **Pumping effect:** Exaggerated sidechain for rhythmic movement (French house, EDM)\n\n**Settings for kick-bass ducking:**\n- Threshold: -20 to -10 dB\n- Ratio: 4:1 to 8:1\n- Attack: 0.1-1 ms (fast, catch transient)\n- Release: 50-200 ms (match kick tail)\n\n**Alternative:** Volume automation or dedicated plugins (Trackspacer, LFOTool, Kickstart) that shape the ducking envelope without needing a sidechain source."
    ))

    pairs.append(msg(
        "How does a vocoder work?",
        "A **vocoder** imposes the spectral envelope (formants) of one signal onto another.\n\n**Signals:**\n- **Modulator:** The analysis signal — typically voice (provides formant shape)\n- **Carrier:** The synthesis signal — typically a synth chord or noise (provides pitch/timbre)\n\n**How it works:**\n1. Split both modulator and carrier into N frequency bands (16-64 bands) using bandpass filters\n2. For each band, extract the **envelope** (amplitude) of the modulator using an envelope follower\n3. Multiply each carrier band by the corresponding modulator envelope\n4. Sum all bands to produce output\n\n**Result:** The carrier takes on the formant shape of the modulator — a synth chord that \"talks.\"\n\n```python\nimport numpy as np\nfrom scipy.signal import butter, sosfilt\n\nclass SimpleVocoder:\n    def __init__(self, sr: int, num_bands: int = 16,\n                 freq_range: tuple = (100, 8000)):\n        self.sr = sr\n        self.bands = []\n        freqs = np.geomspace(freq_range[0], freq_range[1], num_bands + 1)\n        for i in range(num_bands):\n            sos = butter(4, [freqs[i], freqs[i+1]],\n                        btype='bandpass', fs=sr, output='sos')\n            self.bands.append(sos)\n\n    def process(self, modulator: np.ndarray,\n                carrier: np.ndarray) -> np.ndarray:\n        output = np.zeros_like(carrier)\n        for sos in self.bands:\n            mod_band = sosfilt(sos, modulator)\n            car_band = sosfilt(sos, carrier)\n            # Envelope follower (rectify + smooth)\n            envelope = np.abs(mod_band)\n            smooth = sosfilt(butter(2, 30, fs=self.sr, output='sos'), envelope)\n            output += car_band * smooth\n        return output / len(self.bands)\n```\n\n**Classic vocoders:** Korg VC-10, Roland VP-330, EMS Vocoder. Modern: Ableton Vocoder, TAL-Vocoder."
    ))

    pairs.append(msg(
        "What is the difference between mono, stereo, and surround audio?",
        "**Channel configurations:**\n\n| Config | Channels | Use case |\n|--------|----------|----------|\n| **Mono** | 1 | Voice calls, AM radio, some podcasts |\n| **Stereo** | 2 (L, R) | Music, headphones, most media |\n| **LCR** | 3 | Film dialog (center), some broadcast |\n| **Quad** | 4 | Early surround (obsolete) |\n| **5.1** | 6 (L, R, C, LFE, Ls, Rs) | Standard surround (cinema, Blu-ray) |\n| **7.1** | 8 (+ Lss, Rss) | Premium cinema, gaming |\n| **7.1.4** | 12 (+ 4 height) | Dolby Atmos (speaker-based) |\n| **Atmos** | Object-based (128 tracks) | Immersive cinema/music |\n| **Ambisonics** | 4 (FOA) to 64+ (HOA) | VR, spatial audio |\n| **Binaural** | 2 (headphone-optimized) | VR, spatial headphone audio |\n\n**Key concepts:**\n- **Mono compatibility:** Always check stereo mixes in mono — phase cancellation can make elements disappear\n- **LFE** (Low-Frequency Effects): .1 channel, only for rumble effects (not bass guitar)\n- **Phantom center:** In stereo, identical signal in L and R creates a phantom center image\n- **Mid-Side (M/S):** Mid = (L+R)/2, Side = (L-R)/2. Useful for stereo width processing\n\n```python\n# Stereo to Mid-Side conversion\ndef stereo_to_ms(left, right):\n    mid = (left + right) / 2\n    side = (left - right) / 2\n    return mid, side\n\ndef ms_to_stereo(mid, side):\n    left = mid + side\n    right = mid - side\n    return left, right\n```"
    ))

    pairs.append(msg(
        "What is a phase vocoder and how is it used for time-stretching?",
        "A **phase vocoder** manipulates the STFT (Short-Time Fourier Transform) to modify time and pitch independently.\n\n**Time stretching (change duration without changing pitch):**\n1. Compute STFT of input\n2. Resample the magnitude frames at a new rate (stretch factor)\n3. Reconstruct phase coherently (phase accumulation)\n4. Inverse STFT (overlap-add)\n\n```python\nimport numpy as np\nfrom scipy.signal import get_window\n\ndef time_stretch(y: np.ndarray, stretch_factor: float,\n                  n_fft: int = 2048, hop: int = 512) -> np.ndarray:\n    \"\"\"Stretch audio by factor (>1 = slower, <1 = faster).\"\"\"\n    win = get_window('hann', n_fft)\n    phase_advance = np.linspace(0, np.pi * hop, n_fft // 2 + 1, endpoint=False)\n\n    # Analysis hop stays the same, synthesis hop is scaled\n    hop_out = int(hop * stretch_factor)\n\n    # STFT\n    n_frames = 1 + (len(y) - n_fft) // hop\n    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)\n    for i in range(n_frames):\n        frame = y[i * hop:i * hop + n_fft] * win\n        stft[:, i] = np.fft.rfft(frame)\n\n    # Phase vocoder processing\n    out_len = int(len(y) * stretch_factor)\n    output = np.zeros(out_len)\n    phase = np.angle(stft[:, 0])\n\n    time_steps = np.arange(0, n_frames - 1, 1.0 / stretch_factor)\n\n    for i, t in enumerate(time_steps):\n        idx = int(t)\n        if idx + 1 >= n_frames:\n            break\n        frac = t - idx\n\n        # Interpolate magnitude\n        mag = (1 - frac) * np.abs(stft[:, idx]) + frac * np.abs(stft[:, idx + 1])\n\n        # Phase advancement\n        dphi = np.angle(stft[:, idx + 1]) - np.angle(stft[:, idx])\n        dphi -= phase_advance\n        dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))\n        phase += phase_advance + dphi\n\n        # Reconstruct frame\n        frame = np.fft.irfft(mag * np.exp(1j * phase)) * win\n        start = i * hop_out\n        if start + n_fft <= out_len:\n            output[start:start + n_fft] += frame\n\n    return output\n```\n\n**Pitch shifting:** Time-stretch then resample.\n- Pitch up 1 octave: stretch to 2x duration, resample to half length\n- Pitch down: stretch shorter, resample longer\n\n**Artifacts:** Phasiness, metallic quality, transient smearing. WSOLA (Waveform Similarity Overlap-Add) handles transients better."
    ))

    pairs.append(msg(
        "How does a noise gate work and when should you use one?",
        "A **noise gate** silences audio below a threshold, removing background noise during pauses.\n\n**Parameters:**\n- **Threshold** (dB): Level below which the gate closes\n- **Attack** (ms): How quickly the gate opens when signal exceeds threshold (0.1-10ms)\n- **Hold** (ms): Minimum time the gate stays open after signal drops below threshold (10-500ms)\n- **Release** (ms): How quickly the gate closes after hold time (20-500ms)\n- **Range** (dB): How much the gate attenuates when closed (0 = mute, -20dB = gentle reduction)\n- **Sidechain filter:** HPF/LPF on the detector to focus on specific frequencies\n\n**Implementation:**\n```python\nimport numpy as np\n\nclass NoiseGate:\n    def __init__(self, sr: int, threshold_db: float = -40,\n                 attack_ms: float = 1, hold_ms: float = 50,\n                 release_ms: float = 100, range_db: float = -80):\n        self.sr = sr\n        self.threshold = 10 ** (threshold_db / 20)\n        self.attack_coeff = 1.0 / max(1, int(attack_ms * sr / 1000))\n        self.hold_samples = int(hold_ms * sr / 1000)\n        self.release_coeff = 1.0 / max(1, int(release_ms * sr / 1000))\n        self.range_gain = 10 ** (range_db / 20)\n\n        self.gate_level = 0.0  # 0=closed, 1=open\n        self.hold_counter = 0\n\n    def process_sample(self, x: float) -> float:\n        level = abs(x)\n\n        if level >= self.threshold:\n            self.hold_counter = self.hold_samples\n            self.gate_level = min(1.0, self.gate_level + self.attack_coeff)\n        elif self.hold_counter > 0:\n            self.hold_counter -= 1\n        else:\n            self.gate_level = max(self.range_gain,\n                                  self.gate_level - self.release_coeff)\n\n        return x * self.gate_level\n```\n\n**When to use:**\n- Drum close mics (reject bleed from other drums)\n- Live vocals (remove stage noise between phrases)\n- Guitar amps (eliminate hum during pauses)\n- Dialog editing (remove room tone between sentences)\n\n**Tip:** Prefer **expansion** (gentle ratio like 2:1 below threshold) over hard gating for more natural results."
    ))

    pairs.append(msg(
        "What is a low-frequency oscillator (LFO) and how is it used in audio synthesis?",
        "An **LFO** is a slow oscillator (0.01-20 Hz) used for modulation rather than generating audible sound.\n\n**Common LFO waveforms:**\n```python\nimport numpy as np\n\ndef lfo_sine(phase: float) -> float:\n    return np.sin(2 * np.pi * phase)\n\ndef lfo_triangle(phase: float) -> float:\n    return 2 * abs(2 * (phase % 1) - 1) - 1\n\ndef lfo_sawtooth(phase: float) -> float:\n    return 2 * (phase % 1) - 1\n\ndef lfo_square(phase: float) -> float:\n    return 1.0 if (phase % 1) < 0.5 else -1.0\n\ndef lfo_sample_hold(phase: float, prev_phase: float, prev_val: float) -> float:\n    \"\"\"Random value held until next cycle.\"\"\"\n    if int(phase) != int(prev_phase):  # new cycle\n        return np.random.uniform(-1, 1)\n    return prev_val\n```\n\n**Modulation targets:**\n| Target | Effect | Typical rate |\n|--------|--------|-------------|\n| Pitch | Vibrato | 4-7 Hz |\n| Amplitude | Tremolo | 3-8 Hz |\n| Filter cutoff | Wah/sweep | 0.1-5 Hz |\n| Pan | Auto-pan | 0.5-4 Hz |\n| Delay time | Chorus/flanger | 0.1-5 Hz |\n| PWM | Pulse width animation | 0.5-3 Hz |\n\n**LFO parameters:**\n- **Rate/Speed:** Frequency in Hz (or tempo-synced: 1/4, 1/8, etc.)\n- **Depth/Amount:** How much modulation is applied\n- **Phase:** Starting point in the waveform (for stereo offset)\n- **Sync:** Free-running vs tempo-synced vs key-synced (restart on note-on)\n- **Unipolar/Bipolar:** 0-1 range vs -1 to +1 range\n\n**Key-sync LFO** restarts the waveform on every note-on — essential for consistent attack modulation (e.g., filter sweep that always starts the same way)."
    ))

    pairs.append(msg(
        "What are the key audio features used in music information retrieval (MIR)?",
        "**Music Information Retrieval (MIR)** uses audio features for classification, recommendation, and analysis.\n\n**Spectral features:**\n| Feature | Description | Use |\n|---------|-------------|-----|\n| **Spectral centroid** | \"Center of mass\" of the spectrum | Brightness |\n| **Spectral bandwidth** | Spread around centroid | Timbre fullness |\n| **Spectral rolloff** | Frequency below which 85% energy lies | Brightness/noise |\n| **Spectral flux** | Frame-to-frame spectral change | Onset detection |\n| **Spectral flatness** | How noise-like vs tonal (0=tonal, 1=noise) | Speech/noise classification |\n\n**Timbral features:**\n| Feature | Description | Use |\n|---------|-------------|-----|\n| **MFCCs** | Mel-frequency cepstral coefficients (13-20) | Timbre, instrument ID |\n| **Chroma** | Energy per pitch class (12 bins: C to B) | Harmony, key detection |\n| **Tonnetz** | Tonal centroid from chroma (6D) | Harmonic analysis |\n| **Zero-crossing rate** | How often signal crosses zero | Percussive vs harmonic |\n\n**Rhythm features:**\n| Feature | Description | Use |\n|---------|-------------|-----|\n| **Tempo/BPM** | Beats per minute | Tempo classification |\n| **Beat histogram** | Distribution of inter-beat intervals | Rhythm similarity |\n| **Tempogram** | Time-varying tempo estimate | Structure analysis |\n| **Onset strength** | Envelope of onset detection function | Rhythmic energy |\n\n**librosa extraction:**\n```python\nimport librosa\ny, sr = librosa.load('track.wav')\n\ncentroid = librosa.feature.spectral_centroid(y=y, sr=sr)\nmfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)\nchroma = librosa.feature.chroma_stft(y=y, sr=sr)\ntempo, beats = librosa.beat.beat_track(y=y, sr=sr)\nzcr = librosa.feature.zero_crossing_rate(y)\n```\n\n**For ML pipelines:** Typically compute features per frame (hop_length=512), then aggregate with mean/std/delta across time for fixed-size feature vectors."
    ))

    pairs.append(msg(
        "How do you implement a basic audio resampler?",
        """**Resampling** changes the sample rate of audio. Naive approaches cause aliasing; proper resampling requires filtering.

```python
import numpy as np
from scipy.signal import resample_poly, resample

def resample_simple(audio: np.ndarray, orig_sr: int,
                     target_sr: int) -> np.ndarray:
    \"\"\"High-quality resampling using scipy.\"\"\"
    if orig_sr == target_sr:
        return audio
    # resample_poly uses polyphase filtering (best quality)
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down)

def resample_linear(audio: np.ndarray, orig_sr: int,
                     target_sr: int) -> np.ndarray:
    \"\"\"Simple linear interpolation (lower quality, fast).\"\"\"
    ratio = target_sr / orig_sr
    n_out = int(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    int_idx = indices.astype(int)
    frac = indices - int_idx
    int_idx = np.minimum(int_idx, len(audio) - 2)
    return audio[int_idx] * (1 - frac) + audio[int_idx + 1] * frac

# Downsample: 48kHz → 44.1kHz
audio_44k = resample_simple(audio_48k, 48000, 44100)

# Upsample: 44.1kHz → 96kHz
audio_96k = resample_simple(audio_44k, 44100, 96000)
```

**Important:**
- **Downsampling** requires a low-pass anti-aliasing filter BEFORE decimation
- **Upsampling** requires interpolation THEN a low-pass reconstruction filter
- `resample_poly` handles both automatically
- For real-time: use libsamplerate (Secret Rabbit Code) or Speex resampler
- Quality matters most when converting between non-multiple rates (e.g., 44.1k ↔ 48k)"""
    ))

    # Additional coverage topics
    pairs.append(msg(
        "What is a crossfade and how do you implement one between two audio clips?",
        """A **crossfade** smoothly transitions between two audio segments by fading one out while fading the other in.

```python
import numpy as np

def crossfade(audio_a: np.ndarray, audio_b: np.ndarray,
              fade_samples: int,
              curve: str = 'equal_power') -> np.ndarray:
    \"\"\"Crossfade between end of audio_a and start of audio_b.\"\"\"
    if curve == 'linear':
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.linspace(0.0, 1.0, fade_samples)
    elif curve == 'equal_power':
        # Constant power: no volume dip at crossover
        t = np.linspace(0, np.pi / 2, fade_samples)
        fade_out = np.cos(t)
        fade_in = np.sin(t)
    elif curve == 'exponential':
        fade_out = np.logspace(0, -3, fade_samples)  # 1.0 to 0.001
        fade_in = np.logspace(-3, 0, fade_samples)
    else:
        raise ValueError(f\"Unknown curve: {curve}\")

    # Overlap region
    overlap_a = audio_a[-fade_samples:] * fade_out
    overlap_b = audio_b[:fade_samples] * fade_in

    # Concatenate: non-overlap A + crossfade + non-overlap B
    result = np.concatenate([
        audio_a[:-fade_samples],
        overlap_a + overlap_b,
        audio_b[fade_samples:],
    ])
    return result

# Usage: 100ms crossfade at 44.1kHz
sr = 44100
fade_len = int(0.1 * sr)  # 4410 samples
output = crossfade(clip_a, clip_b, fade_len, curve='equal_power')
```

**Equal power crossfade** is preferred for music because linear crossfade causes a ~3dB volume dip at the midpoint. With equal power, `cos^2 + sin^2 = 1` maintains constant energy."""
    ))

    pairs.append(msg(
        "Explain the difference between mono and polyphonic synthesizers and how to implement polyphony.",
        """**Mono synth:** One voice at a time. New notes steal the current voice.
**Poly synth:** Multiple simultaneous voices (typically 4-128).

**Voice allocation strategies:**
- **Oldest:** Steal the voice that has been playing longest
- **Quietest:** Steal the voice with the lowest amplitude
- **Lowest/Highest:** Steal the lowest or highest pitch
- **Round-robin:** Cycle through voices in order

```python
class PolySynth:
    def __init__(self, num_voices: int = 8, sr: int = 44100):
        self.sr = sr
        self.voices = [Voice(sr) for _ in range(num_voices)]
        self.voice_order = list(range(num_voices))  # oldest first

    def note_on(self, note: int, velocity: int):
        # Check if note is already playing
        for v in self.voices:
            if v.note == note and v.active:
                v.retrigger(velocity)
                return

        # Find free voice
        for i, v in enumerate(self.voices):
            if not v.active:
                v.start(note, velocity)
                self.voice_order.remove(i)
                self.voice_order.append(i)
                return

        # Voice stealing: steal oldest
        oldest_idx = self.voice_order[0]
        self.voices[oldest_idx].start(note, velocity)
        self.voice_order.remove(oldest_idx)
        self.voice_order.append(oldest_idx)

    def note_off(self, note: int):
        for v in self.voices:
            if v.note == note and v.active:
                v.release()  # enter release stage, don't stop immediately

    def render(self, num_samples: int) -> np.ndarray:
        output = np.zeros(num_samples)
        for v in self.voices:
            if v.active or v.releasing:
                output += v.render(num_samples)
        return output / max(1, sum(1 for v in self.voices if v.active))

class Voice:
    def __init__(self, sr):
        self.sr = sr
        self.active = False
        self.releasing = False
        self.note = 0
        self.phase = 0.0

    def start(self, note, velocity):
        self.note = note
        self.active = True
        self.releasing = False
        self.velocity = velocity / 127.0
        self.phase = 0.0

    def release(self):
        self.releasing = True

    def retrigger(self, velocity):
        self.velocity = velocity / 127.0
        self.releasing = False

    def render(self, n: int) -> np.ndarray:
        freq = 440 * 2 ** ((self.note - 69) / 12)
        t = (np.arange(n) + self.phase * self.sr) / self.sr
        self.phase += n / self.sr
        return np.sin(2 * np.pi * freq * t) * self.velocity * 0.3
```

**Unison:** Multiple detuned voices per note (±1-30 cents) for a thick, wide sound. Supersaw = 7 detuned sawtooth voices."""
    ))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    generators = {
        "MIDI Protocol": gen_midi_protocol,
        "Audio DSP Concepts": gen_audio_dsp,
        "Synthesis Types": gen_synthesis,
        "Audio Effects": gen_audio_effects,
        "Frameworks": gen_frameworks,
        "Audio File Formats": gen_audio_formats,
        "Real-time Audio": gen_realtime_audio,
        "Music Theory": gen_music_theory,
        "DAW & Plugin Dev": gen_daw_integration,
        "Spatial Audio": gen_spatial_audio,
        "Music Programming": gen_music_programming,
        "Audio Analysis": gen_audio_analysis,
        "MIDI Programming": gen_midi_programming,
        "Audio Misc": gen_audio_misc,
    }

    total = 0
    for label, gen_fn in generators.items():
        pairs = gen_fn()
        count = len(pairs)
        total += count
        logger.info("  %-25s %5d pairs", label, count)
        for pair in pairs:
            print(json.dumps(pair, ensure_ascii=False))

    logger.info("=== TOTAL: %d pairs ===", total)


if __name__ == "__main__":
    main()
