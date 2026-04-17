/* uart_ring.h — lock-free single-producer / single-consumer ring buffer
 *
 * Intended usage:
 *   - Producer: UART RX ISR calls uart_ring_put_from_isr()
 *   - Consumer: main loop / task calls uart_ring_get() or uart_ring_read()
 *
 * Thread/ISR safety:
 *   - Safe with ONE producer (ISR) and ONE consumer (task). No locks needed.
 *   - Capacity MUST be a power of two (compile-time enforced) so that head/tail
 *     can wrap with a single AND instead of a modulo.
 *   - head and tail are declared volatile; on ARMv7-M/ARMv8-M a DMB is inserted
 *     between the data store and the index update to prevent reordering.
 *
 * Overflow policy: DROP_NEW (ISR returns false, byte discarded). Change the
 * ISR to advance tail instead if overwrite-old semantics are required.
 */

#ifndef UART_RING_H
#define UART_RING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef UART_RING_SIZE
#define UART_RING_SIZE 256u
#endif

#if (UART_RING_SIZE & (UART_RING_SIZE - 1u)) != 0u
#error "UART_RING_SIZE must be a power of two"
#endif

#define UART_RING_MASK (UART_RING_SIZE - 1u)

typedef struct {
    uint8_t           buf[UART_RING_SIZE];
    volatile uint16_t head;   /* producer writes, consumer reads */
    volatile uint16_t tail;   /* consumer writes, producer reads */
    volatile uint32_t dropped;/* overflow counter (ISR-incremented) */
} uart_ring_t;

static inline void uart_ring_init(uart_ring_t *r) {
    r->head    = 0;
    r->tail    = 0;
    r->dropped = 0;
}

static inline uint16_t uart_ring_count(const uart_ring_t *r) {
    return (uint16_t)((r->head - r->tail) & UART_RING_MASK);
}

static inline bool uart_ring_empty(const uart_ring_t *r) {
    return r->head == r->tail;
}

static inline bool uart_ring_full(const uart_ring_t *r) {
    return (uint16_t)((r->head + 1u) & UART_RING_MASK) == r->tail;
}

/* Producer (ISR context). Returns false on overflow. */
bool   uart_ring_put_from_isr(uart_ring_t *r, uint8_t byte);

/* Consumer (task context). Returns false if empty. */
bool   uart_ring_get(uart_ring_t *r, uint8_t *out);

/* Bulk consumer. Returns number of bytes copied (0..max). */
size_t uart_ring_read(uart_ring_t *r, uint8_t *dst, size_t max);

#endif /* UART_RING_H */
