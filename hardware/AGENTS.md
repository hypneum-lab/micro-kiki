<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# hardware

## Purpose
KiCad schematics for hardware targets referenced by the embedded / STM32 domain stacks. Currently a single schematic, `spi_bus_4devices.kicad_sch` (SPI bus with 4 peripherals), which is paired with the C-side helpers in `src/spi_mmio.h` and the ring-buffer code in `src/uart_ring.h` (test: `tests/test_uart_ring.c`). The STM32H743 USB bootloader and Ethernet bootloader update protocol designs live in `docs/specs/2026-04-17-*.md` — this directory holds the captured-netlist side.

## Key Files
| File | Description |
|------|-------------|
| `spi_bus_4devices.kicad_sch` | KiCad 7+ schematic: SPI bus with 4 devices, used as the reference design for the `embedded` (idx 14), `stm32` (idx 15), and `kicad-dsl` (idx 11) domain stacks |

## Subdirectories
None yet. When adding PCBs, use `hardware/<board-name>/` with its own `*.kicad_pro` + `*.kicad_sch` + `*.kicad_pcb`.

## For AI Agents

### Working In This Directory
- Use KiCad 7+ s-expression format (`.kicad_sch`, `.kicad_pcb`, `.kicad_sym`, `.kicad_mod`). Never save in legacy KiCad 5/6 format.
- Do NOT commit generated outputs (gerbers, drill files, 3D step exports). Keep source only.
- Schematics here feed the `kicad-dsl` (stack 12, idx 11) and `kicad-pcb` (stack 25) training domains as reference designs — pick component names that exercise those regex patterns in `configs/micro_kiki/domains.yaml` (e.g. `fp_name`, `pad`, `module`, `kicad_sym`).
- Follow the STM32H743 USB bootloader spec in `docs/specs/2026-04-17-stm32h743-usb-bootloader-design.md` and the Ethernet bootloader update protocol in `docs/specs/2026-04-17-ethernet-bootloader-update-protocol.md` when adding related designs.
- Pair C-side changes: `src/spi_mmio.h` and `src/uart_ring.h` must stay consistent with the schematic's bus topology.

### Testing Requirements
- `tests/test_uart_ring.c` is a C unit test for the ring buffer. It compiles and runs standalone (no KiCad dependency); keep it buildable without hardware.
- No Python tests exercise KiCad files directly. DRC / ERC checks happen in the KiCad GUI or via `kicad-cli` in CI if added.
- If adding `kicad-cli` validation to CI, keep the step non-blocking (KiCad version drift breaks easily).

### Common Patterns
- One schematic per functional block; hierarchical sheets for complex boards.
- Net names match the bus names used in the C helpers (`SPI_CS0..3`, `SPI_SCK`, `SPI_MOSI`, `SPI_MISO`).
- 4-device SPI topology: shared SCK/MOSI/MISO, individual chip-selects — matches the common-case training data.

## Dependencies

### Internal
Paired with `src/spi_mmio.h`, `src/uart_ring.h`, `tests/test_uart_ring.c`. Referenced by the `embedded` and `stm32` domain LoRA training configs in `configs/mlx-per-domain/`.

### External
- KiCad 7+ for editing.
- `kicad-cli` optional for headless ERC / DRC / render.
- ARM GCC toolchain (for building `test_uart_ring.c` against firmware stubs).

<!-- MANUAL: -->
