#!/usr/bin/env bash
# One command that captures everything needed to judge a HALO run on the EVO-X2 (docs/evox2.md):
# system/GPU/driver facts, HALO facts, all tests with gtest XML, and with --model a trace-level
# generation, benchmarks, the tune lookup and llama.cpp baselines. Writes
# halo-diag-<host>-<utc>/ and halo-diag-<host>-<utc>.tar.gz with SUMMARY.md and manifest.json.
#   collect.sh [--model GGUF [--mtp GGUF]] [--llama-dir DIR] [--power-mode MODE] [--host-label L]
#              [--anonymize] [--quick] [--out DIR] [--help for everything]
# Exit 0 = everything as expected; 1 = something to look at (SUMMARY.md section 1); 2 = usage/kit error.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
halo_kit collect "$@"
