# contracts

What a compiled model requires of a runtime, extracted once and checked
everywhere.

```
python -m tools.dxnn_inspect model.dxnn --contract contracts/model.json
cpp/build/w5_contract.exe --contract contracts/model.json --npu-memory 512M
```

`densenet121.json` is committed even though the `.dxnn` it came from is not.
The model is 25 MB and is not this repository's to redistribute; the contract is
1 KB of numbers derived from it, and committing it is what lets
`cpp/src/w5_contract.cpp` run on a machine that has never seen a DEEPX
compiler.

It is derived, not authored. Regenerate it rather than editing it, and if the
inspector's checks fail the contract is not written -- a file here has already
passed them.

See [`docs/findings-w5.md`](../docs/findings-w5.md), "W5-3 part two".
