#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -e
cd "$(dirname "$0")"

PACKAGE_DIR="ops_multimodal_fusion"

resolve_npu_arch() {
    case "$1" in
        ascend910b|ascend910_93) echo "dav-2201" ;;
        ascend950)               echo "dav-3510" ;;
        *)                        echo "" ;;
    esac
}

resolve_arch_dir() {
    case "$1" in
        ascend910b|ascend910_93) echo "arch22" ;;
        ascend950)               echo "arch35" ;;
        *)                        echo "" ;;
    esac
}

usage() {
    cat <<EOF
Usage:
  $0                                Full build: produce wheel only (no install, no tests).
  $0 --soc=<soc>                    Specify SoC type. Supported: ascend910b, ascend910_93, ascend950.
                                      Default: ascend950. Can also be set via SOC env var.
  $0 --ops=<a>[,<b>,...]            Incrementally build libops_multimodal_fusion_<op>.so for one or
                                      more ops (no install, no tests). Names are separated
                                      by ','. Example: --ops=add,rms_norm
  $0 --soc=<soc> --ops=<a>[,...]    Combine both flags.
  $0 -h | --help                    Show this help.

Available ops:
$(find applications -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sed 's|applications/[^/]*/|  |' | sort)
EOF
}

# Site-packages ops_multimodal_fusion/, or empty if not installed. cwd is stripped from
# sys.path so running from the project root doesn't resolve to the source tree.
resolve_install_dir() {
    python3 - <<'PY' 2>/dev/null || true
import sys, os
sys.path = [p for p in sys.path if p not in ('', os.getcwd())]
try:
    import ops_multimodal_fusion
    print(os.path.dirname(ops_multimodal_fusion.__file__))
except Exception:
    pass
PY
}

configure_build() {
    [[ -f build/CMakeCache.txt ]] && return
    echo "Configuring build/ (first run)..."
    local torch_dir torch_npu_path
    torch_dir=$(python3 -c 'import os,torch;print(os.path.join(torch.utils.cmake_prefix_path,"Torch"))')
    torch_npu_path=$(python3 -c 'import os,torch_npu;print(os.path.dirname(torch_npu.__file__))')
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DTorch_DIR="$torch_dir" \
        -DTORCH_NPU_PATH="$torch_npu_path" \
        -DNPU_ARCH="$NPU_ARCH" \
        -DARCH_DIR="$ARCH_DIR"
}

# Parse --ops=a,b,c. Sets global OP_NAMES.
parse_ops_arg() {
    local ops_arg="${1#--ops=}"
    OP_NAMES=()
    local _parts _p
    IFS=',' read -r -a _parts <<< "$ops_arg"
    for _p in "${_parts[@]}"; do
        [[ -n "$_p" ]] && OP_NAMES+=("$_p")
    done
    if [[ ${#OP_NAMES[@]} -eq 0 ]]; then
        echo "ERROR: --ops= requires at least one operator name" >&2
        usage
        exit 1
    fi
    for _p in "${OP_NAMES[@]}"; do
        local _found=0
        for _cat in applications/*/; do
            if [[ -d "${_cat}${_p}/${ARCH_DIR}" ]]; then
                _found=1
                break
            fi
        done
        if [[ $_found -eq 0 ]]; then
            echo "ERROR: op '${_p}' not found under applications/*/${_p}/${ARCH_DIR}" >&2
            exit 1
        fi
    done
}

# Build every op in OP_NAMES. Sets global BUILT_SOS (absolute paths).
build_ops() {
    configure_build

    local targets=() op
    for op in "${OP_NAMES[@]}"; do targets+=("ops_multimodal_fusion_${op}"); done
    echo "Building: ${targets[*]}"
    cmake --build build --target "${targets[@]}" --parallel "$(nproc)"

    BUILT_SOS=()
    for op in "${OP_NAMES[@]}"; do
        local so="${PACKAGE_DIR}/libops_multimodal_fusion_${op}.so"
        if [[ ! -f "$so" ]]; then
            echo "ERROR: build succeeded but $so not found" >&2
            exit 1
        fi
        BUILT_SOS+=("$(readlink -f "$so")")
    done
}

# Only print the site-packages cp hint — and only when there's an installed
# copy distinct from the source tree with stale/missing .so files.
print_ops_summary() {
    local src_dir install_dir
    src_dir="$(readlink -f "${PACKAGE_DIR}")"
    install_dir="$(resolve_install_dir)"
    [[ -z "$install_dir" || "$install_dir" == "$src_dir" ]] && return

    local stale=() op src dst
    for op in "${OP_NAMES[@]}"; do
        src="${src_dir}/libops_multimodal_fusion_${op}.so"
        dst="${install_dir}/libops_multimodal_fusion_${op}.so"
        if [[ ! -f "$dst" ]] || ! cmp -s "$src" "$dst"; then
            stale+=("$src")
        fi
    done
    [[ ${#stale[@]} -eq 0 ]] && return

    echo
    echo "Installed ops_multimodal_fusion in site-packages is stale (${#stale[@]}/${#OP_NAMES[@]} .so)."
    echo "Run the following to sync:"
    echo "  cp ${stale[*]} \"${install_dir}/\""
}

build_wheel() {
    echo "Installing build dependencies..."
    pip install -r requirements.txt

    echo "Building the wheel..."
    python3 setup.py clean
    NPU_ARCH="$NPU_ARCH" ARCH_DIR="$ARCH_DIR" SOC="$SOC" python3 -m build --wheel --no-isolation

    local wheel
    wheel="$(ls -t dist/*.whl 2>/dev/null | head -n 1 || true)"
    if [[ -z "$wheel" ]]; then
        echo "ERROR: build completed but no wheel found under dist/" >&2
        exit 1
    fi
    wheel="$(readlink -f "$wheel")"

    echo
    echo "============================================================"
    echo "Full build done."
    echo "============================================================"
    echo "Wheel:"
    echo "  ${wheel}"
    echo
    echo "Install it with:"
    echo "  pip install \"${wheel}\" --force-reinstall --no-deps"
    echo
    echo "Run tests after install (optional):"
    echo "  pytest tests/ -v"
    echo "============================================================"
}

OPS_ARG=""
SOC=${SOC:-ascend950}
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --soc=*)
            SOC="${1#--soc=}"
            shift
            ;;
        --ops=*)
            OPS_ARG="$1"
            shift
            ;;
        *)
            echo "ERROR: unexpected argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

NPU_ARCH=$(resolve_npu_arch "$SOC")
ARCH_DIR=$(resolve_arch_dir "$SOC")
if [[ -z "$NPU_ARCH" ]]; then
    echo "ERROR: unsupported SoC '$SOC'. Supported: ascend910b, ascend910_93, ascend950" >&2
    exit 1
fi

if [[ -n "$OPS_ARG" ]]; then
    parse_ops_arg "$OPS_ARG"
    build_ops
    print_ops_summary
else
    build_wheel
fi
