#!/usr/bin/env python3
"""Generate test vectors for the CDNA field-arithmetic differential test.

Emits N pairs of 256-bit operands as little-endian u64 limbs. The first block of
vectors is a fixed edge-case set (zero, one, p-1, p, 2^256-1, values just below
2^256 that stress the lazy-reduction fold, and small values that make the
divsteps inverse take its worst-case path); the rest are random.
"""
import random, struct, sys

P = (1 << 256) - (1 << 32) - 977
MASK256 = (1 << 256) - 1

def limbs(x):
    return struct.pack('<4Q', x & 0xFFFFFFFFFFFFFFFF, (x >> 64) & 0xFFFFFFFFFFFFFFFF,
                       (x >> 128) & 0xFFFFFFFFFFFFFFFF, (x >> 192) & 0xFFFFFFFFFFFFFFFF)

def edge_values():
    v = [0, 1, 2, 3, 977, 1 << 32,
         P - 1, P, P + 1, P + 2,
         MASK256, MASK256 - 1,
         (1 << 255), (1 << 255) - 1,
         # near 2^256: these are what make the second Solinas fold overflow,
         # the case RC's reference silently gets wrong by 2^256
         MASK256 - 0x1000003D1, MASK256 - (1 << 67), MASK256 - (1 << 67) + 1,
         # dense low words exercise the divstep inner loop's 3-bit correction
         0xFFFFFFFF, 0x80000000, 0x40000000, 0x7FFFFFFF,
         2 * P if 2 * P <= MASK256 else P]
    return v

def main():
    out_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 20
    rnd = random.Random(0xC0FFEE)

    ev = edge_values()
    buf = bytearray()
    count = 0

    # full cross product of edge cases first
    for a in ev:
        for b in ev:
            buf += limbs(a) + limbs(b)
            count += 1

    # mixed edge x random
    for a in ev:
        for _ in range(64):
            buf += limbs(a) + limbs(rnd.getrandbits(256))
            count += 1
            buf += limbs(rnd.getrandbits(256)) + limbs(a)
            count += 1

    # bulk random
    while count < n:
        buf += limbs(rnd.getrandbits(256)) + limbs(rnd.getrandbits(256))
        count += 1

    with open(out_path, 'wb') as f:
        f.write(buf)
    print(f"wrote {count} vectors ({len(buf)} bytes) to {out_path}")

if __name__ == '__main__':
    main()
