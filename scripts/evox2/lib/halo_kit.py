#!/usr/bin/env python3
"""Entry point of the EVO-X2 field kit tooling (docs/evox2.md).

    halo_kit.py run-tests  ...   run the gtest binaries (scripts/evox2/run-tests.sh)
    halo_kit.py compare A B      per-test differences between two run-tests summaries
    halo_kit.py verify-env ...   pre-flight checks (scripts/evox2/verify-env.sh)
    halo_kit.py collect ...      the diagnostic bundle (scripts/evox2/collect.sh)
    halo_kit.py package ...      build-host helpers for scripts/package.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info < (3, 10):
    sys.exit("halo_kit.py needs Python >= 3.10 (found %d.%d)" % sys.version_info[:2])

sys.path.insert(0, str(Path(__file__).resolve().parent))

from halokit import common  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    cmd, rest = sys.argv[1], sys.argv[2:]
    try:
        if cmd == "run-tests":
            from halokit import tests
            return tests.main(rest)
        if cmd == "compare":
            from halokit import tests
            return tests.compare_main(rest)
        if cmd == "verify-env":
            from halokit import verify
            return verify.main(rest)
        if cmd == "collect":
            from halokit import collect
            return collect.main(rest)
        if cmd == "package":
            from halokit import package
            return package.main(rest)
    except common.KitError as e:
        print(f"halo_kit {cmd}: {e}", file=sys.stderr)
        return 2
    print(f"halo_kit: unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
