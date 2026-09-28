#!/usr/bin/env python3
"""Verify GPU field-arithmetic results against exact arbitrary-precision arithmetic.

The device uses a lazy reduction convention: results are in [0, 2^256) and are
congruent to the true value mod p, but not necessarily fully reduced. So every
modular comparison is made mod p, with a separate range check that the value fits
in 256 bits.
"""
import struct, sys

P = (1 << 256) - (1 << 32) - 977
MASK256 = (1 << 256) - 1

R_MUL, R_SQR, R_SUB, R_ADD, R_NEG, R_INV, R_ADD192, R_SUB192 = range(8)
NAMES = {R_MUL: 'MulModP', R_SQR: 'SqrModP', R_SUB: 'SubModP', R_ADD: 'AddModP',
         R_NEG: 'NegModP', R_INV: 'InvModP', R_ADD192: 'Add192to192',
         R_SUB192: 'Sub192from192'}
NOPS = 8

def to_int(limbs):
    return limbs[0] | (limbs[1] << 64) | (limbs[2] << 128) | (limbs[3] << 192)

def main():
    vec_path, res_path = sys.argv[1], sys.argv[2]
    vec = open(vec_path, 'rb').read()
    res = open(res_path, 'rb').read()
    n = len(vec) // (8 * 8)
    assert len(res) == n * NOPS * 4 * 8, f"result size mismatch: {len(res)} vs {n*NOPS*4*8}"

    fails = {k: 0 for k in range(NOPS)}
    first = {}
    checked = 0
    skipped = [0]   # vectors where an operand was >= p (out of contract for add/sub/neg)

    for i in range(n):
        v = struct.unpack_from('<8Q', vec, i * 64)
        a, b = to_int(v[0:4]), to_int(v[4:8])
        r = struct.unpack_from(f'<{NOPS*4}Q', res, i * NOPS * 32)
        got = [to_int(r[k*4:(k+1)*4]) for k in range(NOPS)]

        exp = {
            R_MUL:  (a * b) % P,
            R_SQR:  (a * a) % P,
            R_SUB:  (a - b) % P,
            R_ADD:  (a + b) % P,
            R_NEG:  (P - a) % P if a % P != 0 else None,   # NegModP(0)==p by design
            R_ADD192: None,
            R_SUB192: None,
        }

        # 192-bit ops are plain wrapping integer arithmetic, not modular
        m192 = (1 << 192) - 1
        a192 = v[0] | (v[1] << 64) | (v[2] << 128)
        b192 = v[4] | (v[5] << 64) | (v[6] << 128)
        exp[R_ADD192] = (a192 + b192) & m192
        exp[R_SUB192] = (a192 - b192) & m192

        # MulModP/SqrModP accept any operands in [0, 2^256).
        for k in (R_MUL, R_SQR):
            if got[k] > MASK256 or got[k] % P != exp[k]:
                fails[k] += 1
                first.setdefault(k, (i, a, b, got[k], exp[k]))

        # SubModP/AddModP/NegModP apply a single conditional correction, so their
        # contract is operands in [0, p). See the note in RCGpuUtils_cdna.h: the
        # solver satisfies this with probability 1 - 2^-224.
        in_range = (a < P and b < P)
        if in_range:
            for k in (R_SUB, R_ADD):
                if got[k] > MASK256 or got[k] % P != exp[k]:
                    fails[k] += 1
                    first.setdefault(k, (i, a, b, got[k], exp[k]))
            if exp[R_NEG] is not None and got[R_NEG] % P != exp[R_NEG]:
                fails[R_NEG] += 1
                first.setdefault(R_NEG, (i, a, b, got[R_NEG], exp[R_NEG]))
        else:
            skipped[0] += 1

        for k in (R_ADD192, R_SUB192):
            if got[k] != exp[k]:
                fails[k] += 1
                first.setdefault(k, (i, a, b, got[k], exp[k]))

        # Inverse: check the defining property rather than recomputing it.
        # a == 0 (mod p) has no inverse; the kernel's output is unconstrained there.
        if a % P != 0:
            if (got[R_INV] * a - 1) % P != 0:
                fails[R_INV] += 1
                first.setdefault(R_INV, (i, a, b, got[R_INV], pow(a, -1, P)))

        checked += 1

    total = sum(fails.values())
    print(f"checked {checked} vectors"
          + (f" ({skipped[0]} had an operand >= p, so add/sub/neg were skipped there"
             f" - see the input contract in RCGpuUtils_cdna.h)" if skipped[0] else ""))
    width = max(len(v) for v in NAMES.values())
    for k in range(NOPS):
        status = "OK" if fails[k] == 0 else f"FAIL {fails[k]}"
        print(f"  {NAMES[k]:<{width}}  {status}")
        if fails[k]:
            i, a, b, g, e = first[k]
            print(f"      first at vector {i}")
            print(f"      a   = 0x{a:064x}")
            print(f"      b   = 0x{b:064x}")
            print(f"      got = 0x{g:064x}")
            print(f"      exp = 0x{e:064x}")
    print("PASS" if total == 0 else f"FAILED ({total} mismatches)")
    return 0 if total == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
