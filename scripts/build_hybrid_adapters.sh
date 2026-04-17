#!/usr/bin/env bash
# build_hybrid_adapters.sh — Build hybrid adapter directory from V2/V3 eval results.
#
# Creates output/micro-kiki/stacks-hybrid/ with symlinks to either V2 or V3
# adapters per domain, based on which version performed better.
#
# Runs on Studio (ssh studio) where both adapter directories exist.
#
# Usage:
#   ./scripts/build_hybrid_adapters.sh
#   ./scripts/build_hybrid_adapters.sh --v2-dir /custom/v2 --v3-dir /custom/v3
#   ./scripts/build_hybrid_adapters.sh --dry-run

set -euo pipefail

# --- Defaults (Studio paths) ---
V2_DIR="/Users/clems/KIKI-Mac_tunner/output/micro-kiki/stacks-v2"
V3_DIR="/Users/clems/KIKI-Mac_tunner/output/micro-kiki/stacks-v3-r16"
OUT_DIR="output/micro-kiki/stacks-hybrid"
MANIFEST="$OUT_DIR/hybrid_manifest.json"
DRY_RUN=false

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --v2-dir)   V2_DIR="$2"; shift 2 ;;
        --v3-dir)   V3_DIR="$2"; shift 2 ;;
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--v2-dir DIR] [--v3-dir DIR] [--out-dir DIR] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MANIFEST="$OUT_DIR/hybrid_manifest.json"

# --- Domain → version mapping from V2 vs V3 eval ---
# Format: domain:version:reason
# V3 wins (lower val_loss = better):
#   electronics (-0.55), llm-orch (-0.18), devops (-0.11),
#   security (-0.12), web-frontend (-0.06)
# V2 wins (V3 regressed):
#   spice-sim (+1.51), math (+0.20), kicad-pcb (+0.19),
#   web-backend (+0.17), music-audio (+0.08)
# Ties → use V3 (has null-space): 22 domains
# V3-only (no V2): components, llm-ops, ml-training

declare -A DOMAIN_VERSION
declare -A DOMAIN_REASON

# V3 wins
DOMAIN_VERSION[electronics]=v3;   DOMAIN_REASON[electronics]="v3 wins -0.55"
DOMAIN_VERSION[llm-orch]=v3;      DOMAIN_REASON[llm-orch]="v3 wins -0.18"
DOMAIN_VERSION[devops]=v3;        DOMAIN_REASON[devops]="v3 wins -0.11"
DOMAIN_VERSION[security]=v3;      DOMAIN_REASON[security]="v3 wins -0.12"
DOMAIN_VERSION[web-frontend]=v3;  DOMAIN_REASON[web-frontend]="v3 wins -0.06"

# V2 wins (V3 regressed)
DOMAIN_VERSION[spice-sim]=v2;     DOMAIN_REASON[spice-sim]="v3 regressed +1.51"
DOMAIN_VERSION[math]=v2;          DOMAIN_REASON[math]="v3 regressed +0.20"
DOMAIN_VERSION[kicad-pcb]=v2;     DOMAIN_REASON[kicad-pcb]="v3 regressed +0.19"
DOMAIN_VERSION[web-backend]=v2;   DOMAIN_REASON[web-backend]="v3 regressed +0.17"
DOMAIN_VERSION[music-audio]=v2;   DOMAIN_REASON[music-audio]="v3 regressed +0.08"

# Ties — use V3 (has null-space)
for d in python shell embedded reasoning chat-fr docker kicad-dsl spice \
         rust typescript c-cpp cmake platformio git fastapi react-ui \
         api-design testing networking database hardware-desc system-design \
         documentation; do
    DOMAIN_VERSION[$d]=v3
    DOMAIN_REASON[$d]="tie (null-space)"
done

# V3-only domains (no V2 adapter exists)
DOMAIN_VERSION[components]=v3;    DOMAIN_REASON[components]="v3-only"
DOMAIN_VERSION[llm-ops]=v3;       DOMAIN_REASON[llm-ops]="v3-only"
DOMAIN_VERSION[ml-training]=v3;   DOMAIN_REASON[ml-training]="v3-only"

# --- Validate source dirs ---
if [[ "$DRY_RUN" == false ]]; then
    if [[ ! -d "$V2_DIR" ]]; then
        echo "ERROR: V2 directory not found: $V2_DIR"
        echo "       Run this script on Studio or pass --v2-dir"
        exit 1
    fi
    if [[ ! -d "$V3_DIR" ]]; then
        echo "ERROR: V3 directory not found: $V3_DIR"
        echo "       Run this script on Studio or pass --v3-dir"
        exit 1
    fi
fi

# --- Create output dir ---
mkdir -p "$OUT_DIR"

# --- Summary counters ---
count_v2=0
count_v3=0
count_skip=0

# Helper to avoid set -e killing on ((0++))
incr() { eval "$1=$(( ${!1} + 1 ))"; }

# --- Build manifest header ---
echo '{' > "$MANIFEST"
echo '  "created": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",' >> "$MANIFEST"
echo '  "v2_source": "'"$V2_DIR"'",' >> "$MANIFEST"
echo '  "v3_source": "'"$V3_DIR"'",' >> "$MANIFEST"
echo '  "domains": {' >> "$MANIFEST"

first=true

# --- Summary table header ---
printf "\n%-20s %-8s %-30s %s\n" "DOMAIN" "VERSION" "REASON" "STATUS"
printf "%-20s %-8s %-30s %s\n" "--------------------" "--------" "------------------------------" "------"

# --- Process each domain ---
for domain in $(echo "${!DOMAIN_VERSION[@]}" | tr ' ' '\n' | sort); do
    version="${DOMAIN_VERSION[$domain]}"
    reason="${DOMAIN_REASON[$domain]}"

    # Determine source directory
    if [[ "$version" == "v2" ]]; then
        src_dir="$V2_DIR/$domain"
    else
        src_dir="$V3_DIR/$domain"
    fi

    # Check source exists
    if [[ "$DRY_RUN" == false && ! -d "$src_dir" ]]; then
        printf "%-20s %-8s %-30s %s\n" "$domain" "$version" "$reason" "SKIP (not found)"
        incr count_skip
        continue
    fi

    # Try to read val_loss from training_args.json or adapter_config.json
    val_loss="null"
    if [[ "$DRY_RUN" == false ]]; then
        # Check for training log with final val_loss
        for candidate in "$src_dir/training_args.json" "$src_dir/adapter_config.json"; do
            if [[ -f "$candidate" ]]; then
                # Try to extract val_loss
                extracted=$(python3 -c "
import json, sys
try:
    d = json.load(open('$candidate'))
    vl = d.get('val_loss') or d.get('final_val_loss') or d.get('best_val_loss')
    if vl: print(vl)
except: pass
" 2>/dev/null || true)
                if [[ -n "$extracted" ]]; then
                    val_loss="$extracted"
                    break
                fi
            fi
        done
    fi

    # Create symlink
    status="OK"
    if [[ "$DRY_RUN" == false ]]; then
        ln -sfn "$src_dir" "$OUT_DIR/$domain"
    else
        status="DRY-RUN"
    fi

    # Append to manifest
    if [[ "$first" == true ]]; then
        first=false
    else
        echo ',' >> "$MANIFEST"
    fi
    printf '    "%s": {"version": "%s", "val_loss": %s, "reason": "%s"}' \
        "$domain" "$version" "$val_loss" "$reason" >> "$MANIFEST"

    # Update counters
    if [[ "$version" == "v2" ]]; then
        incr count_v2
    else
        incr count_v3
    fi

    printf "%-20s %-8s %-30s %s\n" "$domain" "$version" "$reason" "$status"
done

# --- Close manifest ---
echo '' >> "$MANIFEST"
echo '  }' >> "$MANIFEST"
echo '}' >> "$MANIFEST"

# --- Summary ---
total=$((count_v2 + count_v3))
echo ""
echo "=== HYBRID BUILD SUMMARY ==="
echo "V2 adapters: $count_v2"
echo "V3 adapters: $count_v3"
echo "Skipped:     $count_skip"
echo "Total:       $total"
echo "Manifest:    $MANIFEST"
echo "Output dir:  $OUT_DIR"
if [[ "$DRY_RUN" == true ]]; then
    echo "(DRY RUN — no symlinks created)"
fi
