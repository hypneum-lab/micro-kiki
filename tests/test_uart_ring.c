/* test_uart_ring.c — unit tests
 *
 * Build: cc -Wall -Wextra -O2 -I../src ../src/uart_ring.c test_uart_ring.c -o test_uart_ring
 */

#include "../src/uart_ring.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_init_empty(void) {
    uart_ring_t r;
    uart_ring_init(&r);
    assert(uart_ring_empty(&r));
    assert(!uart_ring_full(&r));
    assert(uart_ring_count(&r) == 0);
}

static void test_put_get_roundtrip(void) {
    uart_ring_t r;
    uart_ring_init(&r);
    for (int i = 0; i < 10; i++) {
        assert(uart_ring_put_from_isr(&r, (uint8_t)i));
    }
    assert(uart_ring_count(&r) == 10);
    for (int i = 0; i < 10; i++) {
        uint8_t b;
        assert(uart_ring_get(&r, &b));
        assert(b == (uint8_t)i);
    }
    assert(uart_ring_empty(&r));
}

static void test_overflow_drops(void) {
    uart_ring_t r;
    uart_ring_init(&r);
    /* Capacity is SIZE-1 (one slot sacrificed to disambiguate full vs empty). */
    for (unsigned i = 0; i < UART_RING_SIZE - 1u; i++) {
        assert(uart_ring_put_from_isr(&r, (uint8_t)i));
    }
    assert(uart_ring_full(&r));
    assert(!uart_ring_put_from_isr(&r, 0xFF));
    assert(r.dropped == 1);
}

static void test_wrap_around(void) {
    uart_ring_t r;
    uart_ring_init(&r);
    /* Fill half, drain half, fill again → forces head/tail past the boundary. */
    for (int i = 0; i < (int)UART_RING_SIZE / 2; i++) {
        uart_ring_put_from_isr(&r, (uint8_t)i);
    }
    uint8_t scratch[UART_RING_SIZE];
    size_t got = uart_ring_read(&r, scratch, UART_RING_SIZE);
    assert(got == UART_RING_SIZE / 2u);

    for (unsigned i = 0; i < UART_RING_SIZE - 1u; i++) {
        assert(uart_ring_put_from_isr(&r, (uint8_t)(i + 100u)));
    }
    got = uart_ring_read(&r, scratch, UART_RING_SIZE);
    assert(got == UART_RING_SIZE - 1u);
    for (size_t i = 0; i < got; i++) {
        assert(scratch[i] == (uint8_t)(i + 100u));
    }
}

static void test_bulk_read_partial(void) {
    uart_ring_t r;
    uart_ring_init(&r);
    for (int i = 0; i < 5; i++) {
        uart_ring_put_from_isr(&r, (uint8_t)(i + 1));
    }
    uint8_t out[16] = {0};
    size_t got = uart_ring_read(&r, out, sizeof(out));
    assert(got == 5);
    for (int i = 0; i < 5; i++) {
        assert(out[i] == (uint8_t)(i + 1));
    }
    assert(uart_ring_empty(&r));
}

int main(void) {
    test_init_empty();
    test_put_get_roundtrip();
    test_overflow_drops();
    test_wrap_around();
    test_bulk_read_partial();
    printf("uart_ring: all tests passed\n");
    return 0;
}
