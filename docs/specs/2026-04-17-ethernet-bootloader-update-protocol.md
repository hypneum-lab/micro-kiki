# Ethernet Bootloader Firmware Update Protocol

## Goal

Define a small, implementable bootloader protocol for firmware update over Ethernet with:

- no TCP/IP dependency in the bootloader
- deterministic sequencing and retransmission
- CRC32 integrity at both frame and whole-image level
- safe commit into an inactive slot with rollback support

This is a transport and state-machine spec for the bootloader and the host updater.

## Design choices

This protocol runs directly on Layer 2 Ethernet using a custom EtherType instead of UDP/TCP. That keeps the bootloader small: it only needs MAC + PHY bring-up, RX/TX descriptors, and this parser.

Assumptions:

- the device and updater are on the same broadcast domain
- the bootloader owns a unique MAC address
- the flash layout has a bootloader region, metadata region, and two application slots or one application slot plus one staging slot
- all multi-byte fields in the protocol payload are little-endian

Suggested defaults:

- EtherType: `0x88B5` for development; replace with a product-assigned value for production
- maximum chunk size: `1024` bytes
- request timeout: `250 ms`
- retry count per request: `5`

Frame sizing:

- standard Ethernet MTU gives `1500` bytes of payload after the Ethernet header
- BLUP v1 consumes `32` bytes for its header, so the practical maximum `payload_len` is `1468` bytes on an untagged frame
- if 802.1Q VLAN tagging is in use and the network path still enforces a `1500`-byte IP-style payload budget, keep `payload_len <= 1464`
- `1024` bytes remains a good default because it leaves margin for future extensions and aligns well with common flash page and DMA buffering sizes

Transport rules:

- `DISCOVER_REQ` may be broadcast; all other traffic should be unicast once the host selects a target device
- the bootloader must ignore frames not addressed to its own MAC, except broadcast discovery
- once a session is active, the bootloader should accept frames only from the selected host MAC for that `session_id`

## Why protocol CRC32 is still required

Ethernet frames already carry an FCS, but most MACs verify and strip it before software sees the payload. The bootloader therefore cannot rely on Ethernet FCS alone for end-to-end update integrity.

This protocol uses CRC32 in two places:

- `frame_crc32`: detects corruption in each protocol frame the bootloader actually receives
- `image_crc32`: verifies the complete firmware image before the bootloader marks it bootable

## CRC32 definition

Use standard CRC-32 / IEEE 802.3 parameters:

| Field | Value |
|---|---|
| Polynomial | `0x04C11DB7` |
| Init | `0xFFFFFFFF` |
| RefIn | `true` |
| RefOut | `true` |
| XorOut | `0xFFFFFFFF` |
| Check for `"123456789"` | `0xCBF43926` |

Rules:

- `frame_crc32` is computed over the protocol header with the `frame_crc32` field set to `0`, followed by the payload bytes.
- `image_crc32` is computed over the firmware image bytes only, in ascending offset order, with no padding bytes from flash erase blocks included.

## Flash model

Recommended layout:

| Region | Purpose |
|---|---|
| `BOOT` | immutable or separately updated bootloader |
| `META` | active slot, pending slot, image size, image CRC, version, flags, metadata CRC |
| `SLOT_A` | application image A |
| `SLOT_B` | application image B |

Update policy:

1. Bootloader always writes the inactive slot.
2. Bootloader verifies the full image CRC before modifying `META`.
3. Bootloader marks the new slot as `pending`, not `confirmed`.
4. New application must self-confirm within a watchdog window.
5. If confirmation does not happen, bootloader rolls back to the previous confirmed slot.

## Ethernet frame layout

```text
+----------------------+----------------------+----------------------+
| Destination MAC (6)  | Source MAC (6)       | EtherType (2)        |
+----------------------+----------------------+----------------------+
| BLUP header (32 bytes)                                       ... |
+---------------------------------------------------------------...+
| Optional payload (0..N bytes)                                 ... |
+---------------------------------------------------------------...+
```

`BLUP` means BootLoader Update Protocol.

## BLUP v1 header

```c
typedef struct __attribute__((packed)) {
    uint32_t magic;       // 0x50554C42 = 'BLUP'
    uint8_t  version;     // 1
    uint8_t  msg_type;    // enum blup_msg_type
    uint16_t flags;       // enum blup_flags bitmask
    uint32_t session_id;  // 0 until START_RSP assigns one
    uint32_t seq_no;      // host request sequence, starts at 1 per session
    uint32_t ack_no;      // set by device in responses, else 0
    uint32_t offset;      // image byte offset for DATA and DATA_ACK
    uint16_t payload_len; // bytes after this header
    uint16_t header_len;  // must be 32 in v1
    uint32_t frame_crc32; // CRC32(header with this field zeroed + payload)
} blup_hdr_v1_t;
```

Header rules:

- `magic` must be `BLUP`.
- `version` must be `1`.
- `header_len` must be `32`.
- `frame_crc32` must be validated before any semantic checks that could trigger flash writes or state changes.
- `seq_no` is meaningful on host-to-device frames.
- device responses set `ack_no = request.seq_no`.
- `session_id` is assigned by the device in `START_RSP` and must be echoed by the host for the rest of the session.

For idempotent retry behavior:

- the device must cache the last response generated for the active `session_id`
- if it receives a duplicate request with the same `session_id` and `seq_no`, it must replay the cached response without performing the flash operation again
- if it receives an older non-duplicate `seq_no`, it should reject it with `BLUP_ERR_SEQUENCE`

## Flags

| Bit | Name | Meaning |
|---|---|---|
| `0` | `ACK_REQUIRED` | sender expects a response |
| `1` | `LAST_CHUNK` | final `DATA` frame of the image stream |
| `2` | `FORCE_REBOOT` | allow immediate reboot after successful `END_REQ` |
| `3` | `RESUME_QUERY` | `STATUS_REQ` asks whether transfer can resume |

Unused bits must be zero in v1.

## Message types

| Value | Name | Direction | Purpose |
|---|---|---|---|
| `0x01` | `DISCOVER_REQ` | host -> broadcast/unicast | find bootloader-capable devices |
| `0x02` | `DISCOVER_RSP` | device -> host | identify device and limits |
| `0x10` | `START_REQ` | host -> device | declare incoming image |
| `0x11` | `START_RSP` | device -> host | accept or reject session and chunk size |
| `0x12` | `DATA` | host -> device | send a firmware chunk |
| `0x13` | `DATA_ACK` | device -> host | acknowledge programmed chunk |
| `0x14` | `END_REQ` | host -> device | request full-image verify and commit |
| `0x15` | `END_RSP` | device -> host | final result before reboot |
| `0x16` | `STATUS_REQ` | host -> device | ask for current session state |
| `0x17` | `STATUS_RSP` | device -> host | report expected offset and state |
| `0x18` | `ABORT_REQ` | host -> device | cancel active session |
| `0x7F` | `ERROR_RSP` | device -> host | reject request with reason |

## Payloads

### `DISCOVER_REQ`

```c
typedef struct __attribute__((packed)) {
    uint32_t host_nonce;
} blup_discover_req_t;
```

Send to broadcast MAC when the host does not know the device address yet.

### `DISCOVER_RSP`

```c
typedef struct __attribute__((packed)) {
    uint32_t host_nonce;
    uint32_t device_id;
    uint16_t hw_rev;
    uint16_t boot_ver;
    uint16_t max_chunk_size;
    uint16_t erase_block_size;
    uint32_t slot_size;
    uint8_t  active_slot;
    uint8_t  confirmed_slot;
    uint16_t capabilities;
} blup_discover_rsp_t;
```

`capabilities` bit suggestions:

- bit `0`: dual-slot update
- bit `1`: rollback supported
- bit `2`: status/resume supported
- bit `3`: image signature supported

### `START_REQ`

```c
typedef struct __attribute__((packed)) {
    uint32_t image_size;
    uint32_t image_crc32;
    uint32_t image_version;
    uint32_t compat_id;
    uint16_t requested_chunk_size;
    uint8_t  target_slot;
    uint8_t  image_type;
    uint32_t reserved;
} blup_start_req_t;
```

Field rules:

- `image_size` must be `> 0` and `<= slot_size`
- `image_crc32` is the final whole-image CRC
- `image_version` is application-defined
- `compat_id` lets the bootloader reject an image for the wrong hardware family
- `target_slot = 0xFF` means "use the inactive slot automatically"

### `START_RSP`

```c
typedef struct __attribute__((packed)) {
    uint16_t accepted_chunk_size;
    uint16_t status;
    uint32_t session_id;
    uint32_t next_offset;
    uint32_t slot_base_addr;
    uint32_t reserved;
} blup_start_rsp_t;
```

`next_offset` is normally `0`. If resume is implemented and a matching partial image is retained, it may be non-zero.

Resume is valid only when the retained partial image matches at least:

- `image_size`
- `image_crc32`
- `image_version`
- `compat_id`
- `image_type`

If any of those differ, the device must erase the inactive slot or staging partition and return `next_offset = 0`.

### `DATA`

Payload is raw firmware bytes. The header fields carry the control information:

- `offset`: absolute byte offset within the image
- `payload_len`: chunk length
- `flags & LAST_CHUNK`: optional hint for the last data frame

Constraints:

- `payload_len` must be `<= accepted_chunk_size`
- `offset` must equal the bootloader's `next_expected_offset`
- `offset + payload_len` must be `<= image_size`

### `DATA_ACK`

```c
typedef struct __attribute__((packed)) {
    uint16_t status;
    uint16_t reserved;
    uint32_t next_offset;
    uint32_t bytes_written;
    uint32_t running_crc32;
} blup_data_ack_t;
```

`running_crc32` is optional but useful for debugging. It is the CRC32 of the image bytes accepted so far.

### `END_REQ`

```c
typedef struct __attribute__((packed)) {
    uint32_t image_size;
    uint32_t image_crc32;
    uint32_t total_chunks;
    uint32_t reserved;
} blup_end_req_t;
```

Bootloader behavior on receipt:

1. confirm `bytes_written == image_size`
2. recompute or finalize full image CRC32 from flash
3. validate image header or vector table according to platform policy
4. write `META` with `pending_slot`, `image_size`, `image_crc32`, `image_version`, `confirmed = 0`

### `END_RSP`

```c
typedef struct __attribute__((packed)) {
    uint16_t status;
    uint16_t reboot_delay_ms;
    uint32_t computed_image_crc32;
    uint32_t pending_slot;
    uint32_t reserved;
} blup_end_rsp_t;
```

### `STATUS_REQ`

No payload is required for a plain live-session query.

If `flags & RESUME_QUERY` is set, the payload must be:

```c
typedef struct __attribute__((packed)) {
    uint32_t image_size;
    uint32_t image_crc32;
    uint32_t image_version;
    uint32_t compat_id;
} blup_status_req_t;
```

This lets the device answer whether a persisted partial image matches the host's manifest closely enough to resume safely after reset or link loss.

### `STATUS_RSP`

```c
typedef struct __attribute__((packed)) {
    uint16_t state;
    uint16_t status;
    uint32_t session_id;
    uint32_t next_offset;
    uint32_t image_size;
    uint32_t image_crc32;
    uint16_t accepted_chunk_size;
    uint16_t flags;
} blup_status_rsp_t;
```

Suggested `state` values:

- `0`: idle
- `1`: erasing slot
- `2`: receiving data
- `3`: verifying image
- `4`: ready to reboot
- `5`: fault

Suggested `flags` bits:

- bit `0`: resumable image present
- bit `1`: active session present
- bit `2`: slot already erased for current image
- bit `3`: manifest matched current `RESUME_QUERY`

### `ABORT_REQ`

```c
typedef struct __attribute__((packed)) {
    uint16_t reason;
    uint16_t reserved;
} blup_abort_req_t;
```

On abort, the bootloader clears session RAM state. Whether it also erases the partially written slot is product policy; the safe default is yes.

### `ERROR_RSP`

```c
typedef struct __attribute__((packed)) {
    uint16_t status;
    uint16_t reserved;
    uint32_t expected_offset;
    uint32_t detail;
} blup_error_rsp_t;
```

`expected_offset` lets the host recover from duplicate or missed frames.

## Status codes

| Value | Name | Meaning |
|---|---|---|
| `0x0000` | `BLUP_OK` | success |
| `0x0001` | `BLUP_ERR_MAGIC` | wrong magic |
| `0x0002` | `BLUP_ERR_VERSION` | unsupported protocol version |
| `0x0003` | `BLUP_ERR_HEADER_LEN` | invalid header size |
| `0x0004` | `BLUP_ERR_FRAME_CRC` | frame CRC32 mismatch |
| `0x0005` | `BLUP_ERR_SESSION` | missing or stale session |
| `0x0006` | `BLUP_ERR_STATE` | request not valid in current bootloader state |
| `0x0007` | `BLUP_ERR_OFFSET` | unexpected image offset |
| `0x0008` | `BLUP_ERR_LENGTH` | payload length invalid |
| `0x0009` | `BLUP_ERR_IMAGE_SIZE` | image does not fit target slot |
| `0x000A` | `BLUP_ERR_COMPAT` | wrong hardware or image type |
| `0x000B` | `BLUP_ERR_FLASH` | erase or program failure |
| `0x000C` | `BLUP_ERR_IMAGE_CRC` | final image CRC mismatch |
| `0x000D` | `BLUP_ERR_IMAGE_INVALID` | image header/vector table invalid |
| `0x000E` | `BLUP_ERR_BUSY` | device already updating |
| `0x000F` | `BLUP_ERR_INTERNAL` | internal invariant failure |
| `0x0010` | `BLUP_ERR_SEQUENCE` | stale or unexpected `seq_no` |
| `0x0011` | `BLUP_ERR_RESUME_MISMATCH` | partial image cannot be resumed safely |

## Bootloader state machine

```text
IDLE
  -> DISCOVERABLE
  -> START_REQ accepted
ERASING
  -> RECEIVE on successful erase
RECEIVE
  -> RECEIVE on each valid DATA
  -> VERIFY on END_REQ
VERIFY
  -> READY_TO_REBOOT on valid full image
  -> ERROR/IDLE on failure
READY_TO_REBOOT
  -> reboot
```

Rules:

- bootloader must acknowledge a `DATA` frame only after bytes are programmed and read-back verification succeeds if read-back is enabled
- duplicate `DATA` for an already accepted `offset` must not be programmed again; respond with `DATA_ACK` carrying the current `next_offset`
- out-of-order future `DATA` must return `BLUP_ERR_OFFSET`
- `START_REQ` during an active session must return `BLUP_ERR_BUSY` unless the bootloader chooses to abort and restart
- after reset, the device may assign a new `session_id`; resume is based on manifest matching and `next_offset`, not on preserving the old session identifier

## Host update sequence

### 1. Discovery

1. Host sends `DISCOVER_REQ` to broadcast MAC.
2. Devices in bootloader mode reply with `DISCOVER_RSP`.
3. Host selects one device and switches to unicast.

### 2. Session start

1. Host sends `START_REQ`.
2. Device validates image metadata and target slot.
3. Device either:
   - erases the inactive slot or staging partition, or
   - recognizes a resumable partial image and retains it
4. Device replies with `START_RSP`, including the assigned `session_id` and accepted chunk size.

### 3. Data transfer

1. Host sends sequential `DATA` frames starting at `offset = START_RSP.next_offset`.
2. After each valid chunk, device responds with `DATA_ACK`.
3. Host advances only when `status == BLUP_OK` and `next_offset == offset + payload_len`.
4. On timeout, host resends the same `DATA` frame with the same `seq_no`, `offset`, and payload.

### 4. Finalize

1. After the last chunk is acknowledged, host sends `END_REQ`.
2. Device verifies the full image CRC32 and image format.
3. Device writes pending boot metadata.
4. Device returns `END_RSP`.
5. If `status == BLUP_OK`, device reboots after `reboot_delay_ms`.

## Sequence number and retry rules

- host starts `seq_no` at `1` after `START_REQ` is accepted
- host increments `seq_no` for every new request frame
- retransmission uses the same `seq_no`
- device copies the received request `seq_no` into `ack_no`
- host ignores responses whose `ack_no` does not match the outstanding request
- if the device sees the same `seq_no` again for the active session, it must replay the previous response rather than re-executing the request
- if the host starts a new session after timeout or reset, it must restart `seq_no` from `1`

This gives idempotent retry behavior without requiring a full sliding window implementation.

## Resume rules

Resume support is optional in v1.

If resume is implemented:

- the bootloader must persist enough metadata to prove that bytes `[0, next_offset)` belong to one specific manifest
- `next_offset` must always point to the first missing byte; it must never point inside a gap or partially programmed flash word
- the bootloader may report resume availability either in `START_RSP(next_offset > 0)` or in `STATUS_RSP(flags & bit0)`
- the host must resume by sending `DATA` beginning exactly at `next_offset`
- if the device reports `BLUP_ERR_RESUME_MISMATCH`, the host must restart from offset `0`

Recommended persisted fields for a resumable image:

- target slot
- image size
- image CRC32
- image version
- compatibility ID
- next expected offset
- optional page bitmap or equivalent proof that all prior bytes were committed

## Timeout and recovery behavior

Recommended behavior:

- host waits `250 ms` for a response on a local wired LAN
- if timed out, host retries the same frame up to `5` times
- after repeated timeout, host sends `STATUS_REQ`
- if it is trying to recover a lost session after reset, it should set `RESUME_QUERY` and include the manifest identity fields
- if device reports a different `next_offset`, host resumes from that offset
- if `STATUS_REQ` also fails, host declares the session lost

Power-loss behavior:

- if power is lost before `END_RSP`, the previous confirmed slot remains bootable
- partially written inactive-slot data must not change `active_slot`
- metadata writes must be atomic from the boot decision point of view

## Image validation policy

CRC32 proves integrity, not authenticity. Minimum bootloader checks should include:

- vector table or reset handler address in valid executable range
- image size fits target slot
- optional manifest version and hardware compatibility check

If the product has any hostile-network exposure, add one of:

- Ed25519 or ECDSA signature over image manifest
- HMAC-SHA256 over manifest and image
- encrypted image plus signature

CRC32 alone is not a security feature.

## Recommended metadata structure

```c
typedef struct __attribute__((packed)) {
    uint32_t meta_magic;
    uint32_t meta_version;
    uint32_t active_slot;
    uint32_t confirmed_slot;
    uint32_t pending_slot;
    uint32_t image_size;
    uint32_t image_crc32;
    uint32_t image_version;
    uint32_t flags;
    uint32_t meta_crc32;
} boot_meta_t;
```

`meta_crc32` is computed over the structure with `meta_crc32 = 0`.

Suggested flag bits:

- bit `0`: pending image present
- bit `1`: pending image confirmed
- bit `2`: rollback requested
- bit `3`: boot attempt in progress

## Implementation notes

- Keep bootloader RX logic strict: reject frames with wrong EtherType, wrong header length, wrong version, or bad CRC before any flash access.
- Keep chunking sequential. Random-access writes complicate flash handling and recovery without adding much value on a wired local link.
- If flash program granularity is smaller than chunk size, accumulate into a RAM buffer aligned to the program unit.
- If erase latency is high, do it during `START_REQ` before data streaming begins.
- If watchdog is enabled in bootloader, pet it after each successful erase/program step, not only on frame receipt.

## Minimal happy-path example

```text
Host                         Device
----                         ------
DISCOVER_REQ  -------------> 
               <------------- DISCOVER_RSP
START_REQ     -------------> 
               <------------- START_RSP(session_id=0x1234, chunk=1024)
DATA off=0    -------------> 
               <------------- DATA_ACK(next_offset=1024)
DATA off=1024 -------------> 
               <------------- DATA_ACK(next_offset=2048)
...
DATA off=N    -------------> 
               <------------- DATA_ACK(next_offset=image_size)
END_REQ       -------------> 
               <------------- END_RSP(status=OK, pending_slot=B)
reboot
```

## Scope boundary

This spec covers:

- host-to-bootloader update transport
- chunk integrity and full-image integrity
- session sequencing and retries
- commit into a pending boot slot

This spec does not define:

- how the running application asks the bootloader to enter update mode
- signed image format details
- compression, delta update, or multicast distribution
- bootloader self-update
