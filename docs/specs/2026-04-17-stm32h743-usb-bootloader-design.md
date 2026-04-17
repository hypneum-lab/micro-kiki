# STM32H743 USB Bootloader With CRC32 Verification

## Goal

Design a custom bootloader for `STM32H743` that:

- updates firmware over USB
- verifies image integrity with `CRC32`
- never overwrites the running application until a complete staged image is validated
- survives reset or power loss during download
- remains simple enough to implement with STM32 HAL / CubeMX or a thin LL layer

This design assumes an `STM32H743` with `2 MB` internal flash organized as two banks and a Cortex-M7 application linked at a fixed flash address.

## Design Choice

Use a **single fixed application slot plus one staging slot**, not a symmetric A/B executable layout.

Why:

- STM32 application images are normally linked for a fixed flash base
- the same non-PIC binary cannot safely execute from two different internal flash addresses
- a staging slot in Bank 2 lets the bootloader download and verify a full image before touching the active app
- after verification, the bootloader copies the staged image into the fixed application slot

This gives reliable field updates with CRC32 verification and power-loss recovery during download, without requiring position-independent firmware.

## Flash Layout

Assume 128 KB flash sectors and Bank 2 mapped at `0x0810_0000`.

| Region | Address | Size | Purpose |
|---|---|---:|---|
| Bootloader | `0x0800_0000` | `128 KB` | boot code, USB stack, flash service stubs |
| App slot header | `0x0802_0000` | `4 KB` | metadata for installed app |
| App slot image | `0x0802_1000` | `0xDF000` | fixed executable image |
| Staging header | `0x0810_0000` | `4 KB` | metadata for downloaded image |
| Staging image | `0x0810_1000` | `0xDF000` | verified download before install |
| Metadata journal | `0x081E_0000` | `128 KB` | boot state, staged/install flags |

Notes:

- `0x0802_1000` is the application link address.
- Maximum image payload is `0xDF000` bytes (`913,408` bytes).
- If the application is larger than that, use external QSPI/OSPI staging or reduce reserved regions.

## Image Format

Reserve a small header in front of both the installed image and the staged image.

```c
typedef struct {
    uint32_t magic;          // 'BLIM'
    uint32_t header_version; // = 1
    uint32_t image_size;     // bytes of application payload only
    uint32_t image_crc32;    // CRC32/IEEE of payload only
    uint32_t load_address;   // must be 0x08021000
    uint32_t entry_address;  // reset vector from app image
    uint32_t fw_version;     // monotonic if version gating is wanted
    uint32_t flags;          // VALID, STAGED, INSTALLING, CONFIRMED
    uint32_t reserved[8];
    uint32_t header_crc32;   // CRC32 of header with this field zeroed
} bl_image_header_t;
```

Rules:

- `image_crc32` covers only the application payload, not the header.
- `header_crc32` covers the header itself.
- `load_address` is fixed and must match the linker script.
- `entry_address` must point inside the application payload and have Thumb bit set.

### Transferred Artifact

The host sends the **raw linked application binary only**. It does not prepend `bl_image_header_t`.

The bootloader constructs the on-flash header from:

- the `BEGIN` manifest (`image_size`, `image_crc32`, `fw_version`, optional compatibility ID)
- the first two words of the downloaded payload (initial MSP and reset vector)
- the device-fixed `load_address = 0x08021000`

This keeps the host-side artifact identical to the normal linked `app.bin` while still giving the bootloader durable metadata in flash.

## CRC32 Definition

Use **CRC32/IEEE** so host and device can share standard implementations.

- Polynomial: `0x04C11DB7`
- Init: `0xFFFFFFFF`
- RefIn: `true`
- RefOut: `true`
- XorOut: `0xFFFFFFFF`
- Test vector: `"123456789"` -> `0xCBF43926`

On STM32H743, configure the CRC peripheral to match this exactly. Do not rely on default peripheral settings without validating against the test vector above.

Recommended rule:

- verify CRC32 once after download to the staging slot
- verify CRC32 again after copy to the application slot
- verify CRC32 before every jump to the application unless boot-time budget is extremely tight

The full-boot verification cost is usually acceptable for an image below 1 MB on H743.

## USB Transport

Use **USB CDC ACM** with a small framed binary protocol.

Why CDC instead of USB MSC:

- simpler host tooling
- explicit update commands and status codes
- no filesystem corruption edge cases
- easy to add version checks or authentication later

Why CDC instead of standard DFU:

- easier to carry custom manifest fields like expected CRC32, version, and state
- no dependence on DFU alt-setting conventions

If host interoperability with `dfu-util` matters more than protocol control, DFU is still a valid alternative, but the state machine below assumes CDC.

### Commands

| Command | Purpose |
|---|---|
| `GET_INFO` | bootloader version, max image size, current app version, staged state |
| `BEGIN` | send manifest: size, crc32, version, compatibility ID |
| `DATA` | send image chunk with offset |
| `END` | signal end of transfer |
| `STATUS` | query current transfer or install state |
| `ABORT` | discard in-progress staging image |
| `INSTALL` | request immediate install from stage to app slot |
| `REBOOT` | reboot after install or cancel |

Suggested `BEGIN` payload:

```c
typedef struct {
    uint32_t image_size;
    uint32_t image_crc32;
    uint32_t fw_version;
    uint32_t compat_id;   // board or product family identifier
} bl_begin_req_t;
```

Protocol guidance:

- chunk size: `1024` bytes on USB FS, `4096` bytes on USB HS
- require chunk length to be a multiple of `32` bytes except the last chunk
- include `offset` and `length` in every `DATA` frame
- sequence `BEGIN -> DATA* -> END -> INSTALL`
- acknowledge every chunk; the host advances only after `ACK`
- CDC is a byte stream, so the device parser must reassemble complete protocol frames instead of assuming one USB receive callback equals one command

USB already provides packet-level CRC, so a second per-chunk CRC is optional. The image-level `CRC32` is the integrity decision point.

## Metadata Journal

Use the metadata sector as an append-only journal of small records instead of rewriting one fixed structure.

```c
typedef struct {
    uint32_t magic;          // 'BLST'
    uint32_t sequence;
    uint32_t state;          // IDLE, DOWNLOAD, STAGED_VALID, INSTALLING
    uint32_t staged_size;
    uint32_t staged_crc32;
    uint32_t staged_version;
    uint32_t app_size;
    uint32_t app_crc32;
    uint32_t app_version;
    uint32_t record_crc32;
} bl_state_record_t;
```

Recommended persistent states:

- `IDLE`
- `STAGED_VALID`
- `INSTALLING`

Do not write metadata on every data chunk. If download is interrupted, just discard the partial staging image and require the host to restart.

## Boot Decision Flow

```text
Reset
  ->
Init clocks, GPIO, CRC, backup-register access
  ->
Read metadata journal
  ->
If bootloader-request flag set: enter USB mode
  ->
If STAGED_VALID: install staged image to app slot
  ->
Validate app header
  ->
Validate app CRC32
  ->
If valid: jump to app
  ->
Else: enter USB mode
```

### Enter USB mode when

- no valid application is installed
- app requested bootloader via backup register or retained RAM flag
- a dedicated update button is held at reset
- VBUS is present and a policy says service mode should be preferred

### Stay out of USB mode when

- a valid application exists
- no explicit update request is present
- no staged install is pending

This keeps normal boot fast while preserving a deterministic recovery path.

## Download Flow

### 1. `BEGIN`

- reject if `image_size > MAX_IMAGE_SIZE`
- reject if `compat_id` does not match the product family
- erase staging header + staging image sectors
- create an in-RAM transfer context with fixed `load_address = 0x08021000`
- reply `READY`

### 2. `DATA`

- ensure `offset` matches the expected next offset
- write chunk to `STAGING_IMAGE_BASE + offset`
- update running CRC32 and byte count
- reply `ACK`

### 3. `END`

- compare received byte count to `image_size`
- compare computed CRC32 to expected CRC32
- re-read the staged image and recompute CRC32 once more for a final flash-backed check
- extract initial MSP and reset vector from the staged payload
- reject if those vectors are malformed for the fixed app slot
- write a valid staging header
- append metadata state `STAGED_VALID`
- reply `STAGED_OK`

At this point the running application is still untouched.

## Install Flow

When metadata says `STAGED_VALID`, the bootloader installs the staged image into the fixed application slot.

### Install sequence

1. append metadata state `INSTALLING`
2. copy flash erase/program routines to RAM
3. erase application header + image sectors
4. copy staged payload to application payload
5. invalidate caches
6. verify application CRC32 from the final application slot
7. program application header as the final commit point
8. append metadata state `IDLE`
9. optionally erase staging slot
10. jump to application

Why header-last matters:

- if power is lost before step 7, the application header remains erased or invalid
- the bootloader will not mistake a half-copied payload for a bootable app
- recovery still works because the intact staging slot and `INSTALLING` journal entry remain available

If reset or power loss happens during install:

- bootloader sees `INSTALLING`
- bootloader restarts install from the intact staging slot
- device remains recoverable without a debugger

This is the key power-loss recovery mechanism for the copy phase.

## Critical STM32H743 Notes

### 1. Same-bank flash programming

The bootloader lives in Bank 1 and the application slot is also in Bank 1.

That means:

- you cannot safely erase/program the application slot while executing flash code from the same bank
- the flash erase/program functions used during install must execute from RAM or ITCM

Recommended implementation:

- place flash service functions in a `.ramfunc` section
- copy them into executable RAM at boot
- disable interrupts that might vector into flash while those functions run

Download to the staging slot in Bank 2 does not have this same-bank limitation.

### 2. Flash program granularity

STM32H7 internal flash is programmed in `256-bit` flash words.

Therefore:

- write data in `32-byte` aligned units whenever possible
- pad the final chunk in RAM before programming if needed
- keep the logical `image_size` unchanged for CRC calculation

### 3. Cache maintenance

After programming flash:

- clean/invalidate D-cache as needed
- invalidate I-cache before validating or jumping to the new app

Without cache maintenance, the CPU can observe stale flash contents.

### 4. RAM range validation

Do not validate the initial stack pointer against only `0x2000_0000`.

STM32H743 can legitimately place the stack in multiple SRAM regions, for example:

- `DTCM SRAM` at `0x2000_0000`
- `AXI SRAM` at `0x2400_0000`
- `SRAM1/2/3` at `0x3000_0000`
- `SRAM4` at `0x3800_0000`

Accept any stack pointer that falls inside the RAM regions actually enabled by the product.

## Application Linker Requirements

The application must be linked for the fixed bootloader handoff address.

Example:

```ld
MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08021000, LENGTH = 0x000DF000
    RAM   (xrw) : ORIGIN = 0x24000000, LENGTH = 512K
}
```

The vector table is therefore at `0x08021000`.

The bootloader header is not part of the linked application binary.

## Jump To Application

Before jumping:

- stop SysTick
- disable USB and all NVIC interrupts used by the bootloader
- clear pending interrupts
- deinit clocks/peripherals that the app will reconfigure
- set `SCB->VTOR` to the application vector table
- load MSP from the application vector table
- branch to the reset handler

Reference sequence:

```c
typedef void (*entry_fn_t)(void);

static void jump_to_app(uint32_t app_base)
{
    uint32_t new_msp   = *(volatile uint32_t *)(app_base + 0U);
    uint32_t reset_vec = *(volatile uint32_t *)(app_base + 4U);

    __disable_irq();

    SysTick->CTRL = 0;
    SysTick->LOAD = 0;
    SysTick->VAL  = 0;

    HAL_RCC_DeInit();
    HAL_DeInit();

    SCB_CleanInvalidateDCache();
    SCB_InvalidateICache();

    for (uint32_t i = 0; i < 8; ++i) {
        NVIC->ICER[i] = 0xFFFFFFFFU;
        NVIC->ICPR[i] = 0xFFFFFFFFU;
    }

    SCB->VTOR = app_base;
    __DSB();
    __ISB();

    __set_MSP(new_msp);
    ((entry_fn_t)reset_vec)();
}
```

Validate `new_msp` and `reset_vec` before calling this.

## Minimal Validation Rules Before Jump

Require all of these:

- header magic valid
- header CRC valid
- `image_size > 0` and `image_size <= MAX_IMAGE_SIZE`
- `load_address == 0x08021000`
- stack pointer in a valid RAM region
- reset handler inside `[APP_IMAGE_BASE, APP_IMAGE_BASE + image_size)`
- reset handler LSB set for Thumb state
- image CRC32 matches header

If any check fails, stay in USB bootloader mode.

## Host-Side Update Tool

A small Python or C host tool is enough:

1. read `app.bin`
2. compute `CRC32/IEEE`
3. send `BEGIN(size, crc32, version, compat_id)`
4. stream `DATA` chunks
5. send `END`
6. wait for `STAGED_OK`
7. send `INSTALL`
8. poll `STATUS` until success or reboot

The host tool should refuse to send images larger than `0xDF000` bytes.

## What CRC32 Does And Does Not Protect

CRC32 protects against:

- USB transfer corruption
- flash write corruption
- incomplete images
- accidental binary mismatch between host and target

CRC32 does **not** protect against:

- malicious firmware replacement
- rollback attacks
- unauthorized downgrade or tampering

If authenticity matters, add:

- signed manifest
- SHA-256 over payload
- ECDSA or Ed25519 signature check in the bootloader

CRC32 is the right integrity mechanism for a first bootloader revision, but it is not secure boot.

## Recommended Test Matrix

Verify these cases on real hardware:

1. Valid image download, install, and boot.
2. Corrupt one payload byte on the host side and confirm staging is rejected.
3. Reset during download and confirm the old app still boots.
4. Reset during install and confirm the bootloader restarts install from staging.
5. Corrupt the application header and confirm the board stays in USB mode.
6. Corrupt the installed app payload and confirm CRC32 blocks the jump.
7. Trigger bootloader entry from the application via backup register magic.
8. Test with I-cache and D-cache enabled, not only under debugger.

## Recommended Implementation Split

- `boot_main.c`: reset flow, mode selection, jump logic
- `boot_usb.c`: CDC transport and packet parser
- `boot_crc.c`: CRC32 peripheral wrapper validated against the standard test vector
- `boot_flash.c`: staging writes and application install path
- `boot_flash_ram.c`: RAM-resident erase/program functions
- `boot_state.c`: metadata journal read/write
- `boot_image.c`: header parsing and validation

## Bottom Line

For `STM32H743`, the cleanest CRC32-based USB updater is:

- fixed application image at `0x08021000`
- staged download in Bank 2 over USB CDC
- CRC32 validation in staging
- RAM-resident copy into the fixed application slot
- CRC32 verification again before jump

That design is robust, implementable on stock H743 hardware, and avoids the false simplicity of a dual executable slot layout that the application cannot actually run from without PIC or bank remap support.
