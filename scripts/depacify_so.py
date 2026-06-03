#!/usr/bin/env python3
"""Strip ARM64 pointer-authentication (PAC) instructions from an ELF .so, in place.

    python3 depacify_so.py <libfoo.so> [more.so ...]

Why: the Android emulator on Apple-Silicon Macs mis-virtualizes ARM PAC keys, so the auth
instructions baked into the NDK prebuilt libc++/libunwind fault with SIGILL (autia/autib...).
Real arm64 devices virtualize PAC correctly, so this is an EMULATOR-ONLY workaround — never ship a
depacified .so.

Correctness invariant (the part that matters): depacify is whole-program — every `pac*` (sign) AND
every `aut*` (auth) hint becomes NOP, and `retaa`/`retab` become plain `ret`. Removing sign without
removing the matching auth (or vice versa) would leave a signed pointer used unauthenticated and
crash differently; "never sign, never auth, plain ret" is internally consistent.

Only bytes inside SHF_EXECINSTR (executable) sections are touched, so rodata that happens to contain
these word patterns is never corrupted.
"""
import struct
import sys

NOP = 0xD503201F
RET_X30 = 0xD65F03C0

# HINT-space PAC sign/auth instructions (no register operand) -> NOP.
PAC_HINTS = {
    0xD503233F: "paciasp", 0xD50323BF: "autiasp",
    0xD503231F: "paciaz", 0xD503239F: "autiaz",
    0xD503237F: "pacibsp", 0xD50323FF: "autibsp",
    0xD503235F: "pacibz", 0xD50323DF: "autibz",
    0xD503211F: "pacia1716", 0xD503219F: "autia1716",
    0xD503215F: "pacib1716", 0xD50321DF: "autib1716",
}
# Authenticated returns -> plain ret x30.
PAC_RETS = {0xD65F0BFF: "retaa", 0xD65F0FFF: "retab"}
# Combined auth+branch (braa/blraa/...) cannot become a bare NOP; detect and refuse so we never emit
# a silently-broken binary. (Our libsmollm.so has none of these.)
BR_AUTH_MASK = 0xFE1FF800  # BRAA/BRAB/BLRAA/BLRAB family fixed bits
BR_AUTH_VAL = 0xD61F0800


def exec_section_ranges(data: bytes):
    """Yield (file_offset, size) for every SHF_EXECINSTR section of an ELF64 LE image."""
    assert data[:4] == b"\x7fELF", "not an ELF file"
    assert data[4] == 2, "only ELF64 supported"
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize = struct.unpack_from("<H", data, 58)[0]
    e_shnum = struct.unpack_from("<H", data, 60)[0]
    for i in range(e_shnum):
        base = e_shoff + i * e_shentsize
        sh_flags = struct.unpack_from("<Q", data, base + 8)[0]
        sh_offset = struct.unpack_from("<Q", data, base + 24)[0]
        sh_size = struct.unpack_from("<Q", data, base + 32)[0]
        if sh_flags & 0x4:  # SHF_EXECINSTR
            yield sh_offset, sh_size


def depacify(path: str) -> int:
    with open(path, "rb") as f:
        data = bytearray(f.read())

    counts = {}
    refused = 0
    for off, size in exec_section_ranges(data):
        for p in range(off, off + size - (size % 4), 4):
            w = struct.unpack_from("<I", data, p)[0]
            if w in PAC_HINTS:
                struct.pack_into("<I", data, p, NOP)
                counts[PAC_HINTS[w]] = counts.get(PAC_HINTS[w], 0) + 1
            elif w in PAC_RETS:
                struct.pack_into("<I", data, p, RET_X30)
                counts[PAC_RETS[w]] = counts.get(PAC_RETS[w], 0) + 1
            elif (w & BR_AUTH_MASK) == BR_AUTH_VAL:
                refused += 1

    if refused:
        raise SystemExit(f"{path}: REFUSING — found {refused} braa/blraa-style auth-branches that "
                         f"cannot be NOP'd safely; this binary needs a smarter rewrite.")

    total = sum(counts.values())
    with open(path, "wb") as f:
        f.write(data)
    detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"{path}: patched {total} PAC instruction(s) [{detail or 'none'}]")
    return total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    grand = 0
    for arg in sys.argv[1:]:
        grand += depacify(arg)
    print(f"done: {grand} PAC instruction(s) removed across {len(sys.argv) - 1} file(s)")
