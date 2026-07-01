#!/bin/bash
# ============================================================================
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="$SCRIPT_DIR/third_party"
REPO_DIR="$THIRD_PARTY_DIR/cannbot-skills"
REPO_URL="https://gitcode.com/cann/cannbot-skills.git"
PLUGIN_ROOT="$REPO_DIR/plugins-official/ops-direct-invoke-flash"
VERSION="1.2.2"

echo "==> Setting up cannbot-skills..."

# Ensure third_party directory exists
mkdir -p "$THIRD_PARTY_DIR"

# Clone or update the repo
if [ -d "$REPO_DIR/.git" ]; then
    echo "==> Updating existing cannbot-skills repo..."
    git -C "$REPO_DIR" fetch origin
    # Detect the default branch and track it for pull
    BRANCH=""
    if git -C "$REPO_DIR" show-ref --verify --quiet refs/remotes/origin/master; then
        BRANCH="master"
    elif git -C "$REPO_DIR" show-ref --verify --quiet refs/remotes/origin/main; then
        BRANCH="main"
    fi
    if [ -n "$BRANCH" ]; then
        git -C "$REPO_DIR" checkout "$BRANCH" 2>/dev/null || true
        git -C "$REPO_DIR" pull --ff-only origin "$BRANCH" 2>/dev/null || {
            echo "WARNING: Failed to fast-forward pull '$BRANCH'. The repo may have local changes."
        }
    else
        echo "WARNING: Cannot determine default branch (master/main). Skipping update."
    fi
else
    echo "==> Cloning cannbot-skills into third_party/..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo "==> cannbot-skills repo is up to date."
echo ""

# ── install for a given tool ──
install_for_tool() {
    local TOOL="$1"
    local CONFIG_ROOT

    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$SCRIPT_DIR/.opencode"
    elif [ "$TOOL" = "claude" ]; then
        CONFIG_ROOT="$SCRIPT_DIR/.claude"
    else
        echo "Unknown tool: $TOOL"; return 1
    fi

    echo "==> Installing for $TOOL ($CONFIG_ROOT)..."

    # 1. Skills — load all skills under PLUGIN_ROOT/skills/
    mkdir -p "$CONFIG_ROOT/skills"
    local skill_count=0
    for skill_entry in "$PLUGIN_ROOT/skills"/*; do
        [ -e "$skill_entry" ] || continue
        local name=$(basename "$skill_entry")
        local dst="$CONFIG_ROOT/skills/$name"

        # Resolve source: directory = use directly; file = read pointer to actual path
        local src
        if [ -d "$skill_entry" ]; then
            src="$skill_entry"
        elif [ -f "$skill_entry" ]; then
            local rel_path=$(cat "$skill_entry")
            src="$(cd "$PLUGIN_ROOT/skills" && cd "$rel_path" 2>/dev/null && pwd)" || {
                echo "  WARNING: cannot resolve pointer for '$name', skipping"
                continue
            }
        else
            continue
        fi

        if [ -d "$dst" ] && [ ! -L "$dst" ]; then
            rm -rf "$dst"
        fi
        rm -f "$dst"
        ln -sfn "$src" "$dst"
        echo "  Skill linked: $name"
        skill_count=$((skill_count + 1))
    done

    # 2. Agents — remove old copies, then symlink
    mkdir -p "$CONFIG_ROOT/agents"
    local agent_count=0
    for agent_entry in "$PLUGIN_ROOT/agents"/*; do
        [ -e "$agent_entry" ] || continue
        local name=$(basename "$agent_entry")
        local dst="$CONFIG_ROOT/agents/$name"
        if [ -f "$dst" ] && [ ! -L "$dst" ]; then
            rm -f "$dst"
        fi
        rm -f "$dst"
        ln -sfn "$agent_entry" "$dst"
        agent_count=$((agent_count + 1))
    done
    echo "  Agents linked: $agent_count"

    # 3. Manifest
    cat > "$CONFIG_ROOT/cannbot-manifest.json" << EOF
{
  "brand": "CANNBot",
  "version": "$VERSION",
  "team": "ops-direct-invoke-flash",
  "level": "project",
  "tool": "$TOOL",
  "installed_skills": ["ops-direct-invoke-flash", "ascendc-st-design", "ascendc-whitebox-design"],
  "installed_agents": ["ops-direct-invoke-flash-reviewer"],
  "brand_dir": "$CONFIG_ROOT",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "  Manifest written"
    echo ""
}

install_for_tool "opencode"
install_for_tool "claude"

# ── Extra: set skillDirectories in Claude Code settings.local.json ──
SETTINGS_FILE="$SCRIPT_DIR/.claude/settings.local.json"
PYTHON=""
for py in python python3; do
    if command -v "$py" &>/dev/null && "$py" -c "print('ok')" 2>/dev/null; then
        PYTHON="$py"
        break
    fi
done

if [ -n "$PYTHON" ]; then
    if [ -f "$SETTINGS_FILE" ]; then
        $PYTHON -c '
import json, sys
with open(sys.argv[1], "r") as f:
    config = json.load(f)
existing = config.get("skillDirectories", [])
sd = "third_party/cannbot-skills/plugins-official/ops-direct-invoke-flash"
if sd not in existing:
    existing.insert(0, sd)
    config["skillDirectories"] = existing
    with open(sys.argv[1], "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print("  skillDirectories updated in settings.local.json")
' _ "$SETTINGS_FILE" || true
    else
        cat > "$SETTINGS_FILE" << 'SETTINGSEOF'
{
  "skillDirectories": [
    "third_party/cannbot-skills/plugins-official/ops-direct-invoke-flash"
  ]
}
SETTINGSEOF
        echo "  Created settings.local.json"
    fi
fi

echo "Done! Configuration summary:"
echo "  Repo:    $REPO_DIR"
echo "  Skills:  .claude/skills/ + .opencode/skills/ (symlinked from cannbot-skills)"
echo ""
echo "When you start opencode or Claude Code from this project root,"
echo "cannbot-skills will be auto-loaded."
