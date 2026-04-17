/* spi_mmio.h — memory-mapped I/O register abstraction for SPI
 *
 * Modeled on the STM32F4 SPI peripheral layout (widely portable to most
 * 32-bit MCUs with a similar CR1/CR2/SR/DR block). The abstraction keeps
 * the register map as a `volatile` struct so the compiler emits exactly
 * one load/store per access — no reordering, no caching.
 *
 * Usage:
 *   #define SPI1_BASE 0x40013000u
 *   spi_regs_t * const SPI1 = (spi_regs_t *)SPI1_BASE;
 *   spi_enable(SPI1);
 *   uint8_t rx = spi_xfer8(SPI1, 0xA5);
 *
 * Portability:
 *   - Register offsets match STM32F4/F7/H7. For other silicon, re-check
 *     the offsets in the reference manual and adjust padding if needed.
 *   - All fields are uint32_t even when hardware exposes 16-bit — the bus
 *     performs word access; upper bits are RAZ/WI on STM32.
 *   - Bit positions follow STM32F4 RM0090 §28.
 *
 * Safety:
 *   - Single-writer assumed (MCU core). For multi-core SoCs, guard with
 *     a spinlock or use the peripheral's built-in lock bit.
 *   - Always poll `spi_tx_empty()` before writing DR, and `spi_rx_not_empty()`
 *     before reading DR. The ready-state helpers enforce this.
 */

#ifndef SPI_MMIO_H
#define SPI_MMIO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---------- Register map -------------------------------------------------- */

typedef struct {
    volatile uint32_t CR1;      /* 0x00 control register 1       */
    volatile uint32_t CR2;      /* 0x04 control register 2       */
    volatile uint32_t SR;       /* 0x08 status register          */
    volatile uint32_t DR;       /* 0x0C data register            */
    volatile uint32_t CRCPR;    /* 0x10 CRC polynomial           */
    volatile uint32_t RXCRCR;   /* 0x14 RX CRC register          */
    volatile uint32_t TXCRCR;   /* 0x18 TX CRC register          */
    volatile uint32_t I2SCFGR;  /* 0x1C I2S config (unused here) */
    volatile uint32_t I2SPR;    /* 0x20 I2S prescaler            */
} spi_regs_t;

/* ---------- CR1 bit masks ------------------------------------------------- */

#define SPI_CR1_CPHA        (1u << 0)   /* clock phase                    */
#define SPI_CR1_CPOL        (1u << 1)   /* clock polarity                 */
#define SPI_CR1_MSTR        (1u << 2)   /* master mode                    */
#define SPI_CR1_BR_POS      3u          /* baud-rate prescaler [5:3]      */
#define SPI_CR1_BR_MASK     (0x7u << SPI_CR1_BR_POS)
#define SPI_CR1_SPE         (1u << 6)   /* SPI enable                     */
#define SPI_CR1_LSBFIRST    (1u << 7)   /* LSB first                      */
#define SPI_CR1_SSI         (1u << 8)   /* internal slave select          */
#define SPI_CR1_SSM         (1u << 9)   /* software slave management      */
#define SPI_CR1_RXONLY      (1u << 10)  /* receive-only mode              */
#define SPI_CR1_DFF         (1u << 11)  /* data frame format: 0=8b, 1=16b */
#define SPI_CR1_CRCNEXT     (1u << 12)
#define SPI_CR1_CRCEN       (1u << 13)
#define SPI_CR1_BIDIOE      (1u << 14)
#define SPI_CR1_BIDIMODE    (1u << 15)

/* ---------- CR2 bit masks ------------------------------------------------- */

#define SPI_CR2_RXDMAEN     (1u << 0)
#define SPI_CR2_TXDMAEN     (1u << 1)
#define SPI_CR2_SSOE        (1u << 2)   /* SS output enable               */
#define SPI_CR2_FRF         (1u << 4)   /* frame format: 0=Motorola, 1=TI */
#define SPI_CR2_ERRIE       (1u << 5)
#define SPI_CR2_RXNEIE      (1u << 6)
#define SPI_CR2_TXEIE       (1u << 7)

/* ---------- SR bit masks -------------------------------------------------- */

#define SPI_SR_RXNE         (1u << 0)   /* RX buffer not empty            */
#define SPI_SR_TXE          (1u << 1)   /* TX buffer empty                */
#define SPI_SR_CHSIDE       (1u << 2)
#define SPI_SR_UDR          (1u << 3)   /* underrun                       */
#define SPI_SR_CRCERR       (1u << 4)
#define SPI_SR_MODF         (1u << 5)   /* mode fault                     */
#define SPI_SR_OVR          (1u << 6)   /* overrun                        */
#define SPI_SR_BSY          (1u << 7)   /* busy                           */
#define SPI_SR_FRE          (1u << 8)   /* frame-format error             */

/* ---------- Baud-rate prescaler values ------------------------------------ */

typedef enum {
    SPI_BR_DIV2   = 0u,
    SPI_BR_DIV4   = 1u,
    SPI_BR_DIV8   = 2u,
    SPI_BR_DIV16  = 3u,
    SPI_BR_DIV32  = 4u,
    SPI_BR_DIV64  = 5u,
    SPI_BR_DIV128 = 6u,
    SPI_BR_DIV256 = 7u,
} spi_baud_t;

/* ---------- Clock mode (CPOL, CPHA) --------------------------------------- */

typedef enum {
    SPI_MODE0 = 0u,  /* CPOL=0, CPHA=0 */
    SPI_MODE1 = 1u,  /* CPOL=0, CPHA=1 */
    SPI_MODE2 = 2u,  /* CPOL=1, CPHA=0 */
    SPI_MODE3 = 3u,  /* CPOL=1, CPHA=1 */
} spi_mode_t;

/* ---------- Low-level register helpers ------------------------------------ */

static inline void spi_reg_set(volatile uint32_t *reg, uint32_t mask) {
    *reg |= mask;
}

static inline void spi_reg_clear(volatile uint32_t *reg, uint32_t mask) {
    *reg &= ~mask;
}

static inline void spi_reg_modify(volatile uint32_t *reg, uint32_t mask, uint32_t value) {
    uint32_t tmp = *reg;
    tmp &= ~mask;
    tmp |= (value & mask);
    *reg = tmp;
}

/* ---------- Status polling ------------------------------------------------ */

static inline bool spi_tx_empty(const spi_regs_t *s) {
    return (s->SR & SPI_SR_TXE) != 0u;
}

static inline bool spi_rx_not_empty(const spi_regs_t *s) {
    return (s->SR & SPI_SR_RXNE) != 0u;
}

static inline bool spi_busy(const spi_regs_t *s) {
    return (s->SR & SPI_SR_BSY) != 0u;
}

/* ---------- Configuration ------------------------------------------------- */

static inline void spi_set_mode(spi_regs_t *s, spi_mode_t mode) {
    spi_reg_modify(&s->CR1, SPI_CR1_CPOL | SPI_CR1_CPHA, (uint32_t)mode);
}

static inline void spi_set_baud(spi_regs_t *s, spi_baud_t br) {
    spi_reg_modify(&s->CR1, SPI_CR1_BR_MASK, ((uint32_t)br) << SPI_CR1_BR_POS);
}

static inline void spi_set_master(spi_regs_t *s, bool master) {
    if (master) spi_reg_set(&s->CR1, SPI_CR1_MSTR | SPI_CR1_SSI | SPI_CR1_SSM);
    else        spi_reg_clear(&s->CR1, SPI_CR1_MSTR);
}

static inline void spi_enable(spi_regs_t *s) {
    spi_reg_set(&s->CR1, SPI_CR1_SPE);
}

static inline void spi_disable(spi_regs_t *s) {
    /* Safe-disable sequence: wait for TXE, then BSY clear, then clear SPE. */
    while (!spi_tx_empty(s)) { /* spin */ }
    while (spi_busy(s))      { /* spin */ }
    spi_reg_clear(&s->CR1, SPI_CR1_SPE);
}

/* ---------- Data transfer ------------------------------------------------- */

/* Blocking full-duplex 8-bit transfer. Returns the byte shifted in. */
static inline uint8_t spi_xfer8(spi_regs_t *s, uint8_t tx) {
    while (!spi_tx_empty(s)) { /* spin */ }
    *(volatile uint8_t *)&s->DR = tx;        /* 8-bit write keeps DFF=0 */
    while (!spi_rx_not_empty(s)) { /* spin */ }
    return *(volatile uint8_t *)&s->DR;
}

/* Blocking full-duplex 16-bit transfer (requires DFF=1). */
static inline uint16_t spi_xfer16(spi_regs_t *s, uint16_t tx) {
    while (!spi_tx_empty(s)) { /* spin */ }
    s->DR = tx;
    while (!spi_rx_not_empty(s)) { /* spin */ }
    return (uint16_t)s->DR;
}

/* Bulk transfer; tx and/or rx may be NULL (dummy bytes / discard). */
static inline void spi_xfer_buf(spi_regs_t *s,
                                const uint8_t *tx, uint8_t *rx, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        uint8_t out = tx ? tx[i] : 0xFFu;
        uint8_t in  = spi_xfer8(s, out);
        if (rx) rx[i] = in;
    }
}

#endif /* SPI_MMIO_H */
