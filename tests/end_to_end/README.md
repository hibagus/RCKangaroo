# Deterministic end-to-end vectors

The cases in `vectors.json` are the initial small-range acceptance vectors for
the CUDA reference and HIP ports. Their public keys are fixed secp256k1 points
for private scalars 2 and 3 and are also checked by the CPU unit tests.

Example commands:

```sh
./rckangaroo -range 32 -dp 14 -start 0 \
  -pubkey 02C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5 \
  --seed 1 --duration 300

./rckangaroo -range 32 -dp 14 -start 1 \
  -pubkey 02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9 \
  --seed 1 --duration 300
```

Expected private keys are `2` and `3`, respectively. These cases are recorded
now so every backend targets identical inputs; GPU validation remains required
before either backend is declared supported.
