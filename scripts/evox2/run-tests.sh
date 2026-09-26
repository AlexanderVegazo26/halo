#!/usr/bin/env bash
# Runs every gtest binary of the unpacked kit with XML output; see docs/evox2.md.
#   run-tests.sh [--out DIR] [--full-bandwidth] [--vk-icd radv|amdvlk|lavapipe] [--category hip|vulkan|cpu]
#                [--only REGEX] [--ref DIR] [--expect-gpu auto|yes|no] [--timeout S] [--label NAME]
# Exit 0 = no failures and every expectation met; 1 = something to look at (summary.md); 2 = kit error;
# 3 = the requested Vulkan driver is not installed.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
halo_kit run-tests "$@"
