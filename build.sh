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

resolve_npuarch() {
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
  $0                                Full build: produce wheel (all ops).
  $0 --soc=<soc>                    Specify SoC type. Supported: ascend910b, ascend910_93, ascend950.
                                      Default: ascend950. Can also be set via SOC env var.
  $0 --ops=<a>[,<b>,...]            Build specified ops and produce wheel. Names are separated
                                      by ','. Example: --ops=add,rms_norm
  $0 --soc=<soc> --ops=<a>[,...]    Combine both flags.
  $0 --filelist=<path>              Read a file with modified paths (one per line),
                                      auto-detect affected ops and SoCs, then build.
                                      - Only arch35/ touched  → ascend950
                                      - Only arch22/ touched  → ascend910b (same arch as ascend910_93)
                                      - Common files touched  → all SoCs the op supports
  $0 -h | --help                    Show this help.

Available ops:
$(find applications -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sed 's|applications/[^/]*/|  |' | sort)
EOF
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

# Parse --filelist=<path>. Reads a file where each line is a modified path.
# Sets globals: ARCH22_OPS, ARCH35_OPS, TOUCHED_ARCH22, TOUCHED_ARCH35.
parse_filelist() {
    local file="$1"
    [[ -f "$file" ]] || { echo "ERROR: filelist '$file' not found" >&2; exit 1; }

    TOUCHED_ARCH22=0
    TOUCHED_ARCH35=0
    ARCH22_OPS=()
    ARCH35_OPS=()

    local _line _cat _op _global=0
    while IFS= read -r _line || [[ -n "$_line" ]]; do
        [[ -z "$_line" ]] && continue
        _line="${_line#./}"

        if [[ "$_line" =~ ^applications/([^/]+)/([^/]+)/arch22/ ]]; then
            TOUCHED_ARCH22=1
            ARCH22_OPS+=("${BASH_REMATCH[2]}")
        elif [[ "$_line" =~ ^applications/([^/]+)/([^/]+)/arch35/ ]]; then
            TOUCHED_ARCH35=1
            ARCH35_OPS+=("${BASH_REMATCH[2]}")
        elif [[ "$_line" =~ ^applications/([^/]+)/([^/]+)/ ]]; then
            # Op-level common file (outside arch*/) — build for all its supported SoCs.
            TOUCHED_ARCH22=1
            TOUCHED_ARCH35=1
            _cat="${BASH_REMATCH[1]}"
            _op="${BASH_REMATCH[2]}"
            [[ -d "applications/${_cat}/${_op}/arch22" ]] && ARCH22_OPS+=("$_op")
            [[ -d "applications/${_cat}/${_op}/arch35" ]] && ARCH35_OPS+=("$_op")
        else
            # Global common file (build.sh, CMakeLists.txt, etc.) — one op per arch to verify the flow.
            _global=1
            break
        fi
    done < "$file"

    if [[ $_global -eq 1 ]]; then
        TOUCHED_ARCH22=1
        TOUCHED_ARCH35=1
        ARCH22_OPS=()
        ARCH35_OPS=()
        local _d
        for _d in applications/*/*/; do
            _op="$(basename "$_d")"
            [[ -d "${_d}arch22" && ${#ARCH22_OPS[@]} -eq 0 ]] && ARCH22_OPS+=("$_op")
            [[ -d "${_d}arch35" && ${#ARCH35_OPS[@]} -eq 0 ]] && ARCH35_OPS+=("$_op")
            [[ ${#ARCH22_OPS[@]} -gt 0 && ${#ARCH35_OPS[@]} -gt 0 ]] && break
        done
    fi

    # Deduplicate.
    local _sorted _op2
    if [[ ${#ARCH22_OPS[@]} -gt 0 ]]; then
        _sorted=()
        while IFS= read -r _op2; do
            [[ -n "$_op2" ]] && _sorted+=("$_op2")
        done < <(printf '%s\n' "${ARCH22_OPS[@]}" | sort -u)
        ARCH22_OPS=("${_sorted[@]}")
    fi
    if [[ ${#ARCH35_OPS[@]} -gt 0 ]]; then
        _sorted=()
        while IFS= read -r _op2; do
            [[ -n "$_op2" ]] && _sorted+=("$_op2")
        done < <(printf '%s\n' "${ARCH35_OPS[@]}" | sort -u)
        ARCH35_OPS=("${_sorted[@]}")
    fi

    if [[ ${#ARCH22_OPS[@]} -eq 0 && ${#ARCH35_OPS[@]} -eq 0 ]]; then
        if [[ $TOUCHED_ARCH22 -eq 1 || $TOUCHED_ARCH35 -eq 1 ]]; then
            echo "ERROR: common files reference operators whose arch directories do not exist." >&2
            echo "  Ensure arch-specific files (under arch22/ or arch35/) are included in the filelist." >&2
        else
            echo "ERROR: no recognizable operator or arch paths found in '$file'" >&2
        fi
        exit 1
    fi
}

build_wheel() {
    echo "Installing build dependencies..."
    pip install -r requirements.txt

    echo "Building the wheel..."
    python3 setup.py clean

    local op_list=""
    if [[ ${#OP_NAMES[@]} -gt 0 ]]; then
        op_list="$(IFS=,; echo "${OP_NAMES[*]}")"
        echo "Building ops: ${OP_NAMES[*]}"
    fi

    OP_LIST="$op_list" \
        NPU_ARCH="$NPU_ARCH" ARCH_DIR="$ARCH_DIR" SOC="$SOC" \
        python3 -m build --wheel --no-isolation

    local wheel
    wheel="$(ls -t dist/*.whl 2>/dev/null | head -n 1 || true)"
    if [[ -z "$wheel" ]]; then
        echo "ERROR: build completed but no wheel found under dist/" >&2
        exit 1
    fi
    wheel="$(readlink -f "$wheel")"

    echo
    echo "============================================================"
    echo "Build done."
    echo "============================================================"
    echo "Wheel:"
    echo "  ${wheel}"
    echo
    echo "Install it with:"
    echo "  cd /tmp && pip install \"${wheel}\" --force-reinstall --no-deps"
    echo
    echo "Run tests after install (optional):"
    echo "  pytest tests/ -v"
    echo "============================================================"
}

OPS_ARG=""
FILELIST=""
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
        --filelist=*)
            FILELIST="${1#--filelist=}"
            shift
            ;;
        *)
            echo "ERROR: unexpected argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

NPU_ARCH=$(resolve_npuarch "$SOC")
ARCH_DIR=$(resolve_arch_dir "$SOC")
if [[ -z "$NPU_ARCH" ]]; then
    echo "ERROR: unsupported SoC '$SOC'. Supported: ascend910b, ascend910_93, ascend950" >&2
    exit 1
fi

if [[ -n "$FILELIST" && -n "$OPS_ARG" ]]; then
    echo "ERROR: --filelist and --ops are mutually exclusive" >&2
    exit 1
fi

if [[ -n "$FILELIST" ]]; then
    parse_filelist "$FILELIST"

    _soc_list=()
    [[ $TOUCHED_ARCH22 -eq 1 && ${#ARCH22_OPS[@]} -gt 0 ]] && _soc_list+=(ascend910b)
    [[ $TOUCHED_ARCH35 -eq 1 && ${#ARCH35_OPS[@]} -gt 0 ]] && _soc_list+=(ascend950)

    for _build_soc in "${_soc_list[@]}"; do
        SOC="$_build_soc"
        NPU_ARCH=$(resolve_npuarch "$SOC")
        ARCH_DIR=$(resolve_arch_dir "$SOC")
        if [[ "$ARCH_DIR" == "arch22" ]]; then
            OP_NAMES=("${ARCH22_OPS[@]}")
        else
            OP_NAMES=("${ARCH35_OPS[@]}")
        fi
        echo "=== [$SOC] arch=$ARCH_DIR, ${#OP_NAMES[@]} op(s) ==="
        build_wheel
    done
elif [[ -n "$OPS_ARG" ]]; then
    parse_ops_arg "$OPS_ARG"
    build_wheel
else
    build_wheel
fi
