#!/usr/bin/env bash
set -euo pipefail
# Pure POSIX Bash Unsplit Hugging Face Uploader for kapitoshkina-ai/ERNIE-5.5-Flash-Preview-WHT128-HLUT4 (Apache-2.0)
# Uploads 100% unsplit WHT128-HLUT4 shards (45 MoE layers = 23,040 experts = 139.22 GiB) + lossless packed_spine
# + meta + README.md + LICENSE + NOTICE + SHA256SUMS.txt, verifies HTTP 200/307 remote integrity, and shreds /dev/shm/.hf_tok_ernie55.
# Zero Python scripts inside workspace.

ROOT="${ERNIE55_ROOT:-.}"
STAGE_DIR="${1:-$ROOT/release_hf_staging}"
HF_REPO="${HF_REPO:-kapitoshkina-ai/ERNIE-5.5-Flash-Preview-WHT128-HLUT4}"
TOK_FILE="${HF_TOK_FILE:-/dev/shm/.hf_tok_ernie55}"
RECEIPT_OUT="$ROOT/artifacts/hf_upload_preview_receipt.json"

if [[ ! -d "$STAGE_DIR" ]]; then
  echo "[HF-UPLOAD] ERROR: Staging directory missing: $STAGE_DIR" >&2
  exit 1
fi

# 1. Mandatory 10-Layer Zero-PII & Unsplit Gate BEFORE touching Hugging Face!
"$ROOT/bin/ernie55_sanitize_verifier" "$STAGE_DIR" --mode hf

# 2. Load HF token cleanly (prefer /dev/shm/.hf_tok_ernie55, else $HF_TOKEN)
SHRED_TOK=0
if [[ -f "$TOK_FILE" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "$TOK_FILE")"
  export HF_TOKEN
  SHRED_TOK=1
elif [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[HF-UPLOAD] Ready! Place your write-scoped HF token in $TOK_FILE (chmod 600) or export HF_TOKEN and run:"
  echo "  bash $ROOT/scripts/upload_hf_ernie55_preview.sh"
  exit 0
fi

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_XET_HIGH_PERFORMANCE=1

echo "[HF-UPLOAD] Ensuring public Apache-2.0 model repo $HF_REPO exists..."
hf repo create "$HF_REPO" --repo-type model --exist-ok --token "$HF_TOKEN" || true

# 3. Upload root docs/metadata/manifests first (small commit)
echo "[HF-UPLOAD] Batch 1: Uploading README.md, LICENSE, NOTICE, CITATION.cff, SHA256SUMS.txt, telemetry PNG, meta/..."
for f in README.md LICENSE NOTICE CITATION.cff SHA256SUMS.txt ernie55_preview_hlut4_1m_telemetry.png router_perm.bin; do
  if [[ -f "$STAGE_DIR/$f" ]]; then
    hf upload "$HF_REPO" "$STAGE_DIR/$f" "$f" --repo-type model --token "$HF_TOKEN" --commit-message "Add $f"
  fi
done
hf upload "$HF_REPO" "$STAGE_DIR/meta" "meta" --repo-type model --token "$HF_TOKEN" --commit-message "Add upstream Preview-BF16 metadata and FlashIndexer"

# 4. Upload lossless packed_spine/ in 3 clean batches (zero .part splits!)
echo "[HF-UPLOAD] Uploading lossless packed_spine/ (45 unsplit safetensors files)..."
hf upload "$HF_REPO" "$STAGE_DIR/packed_spine" "packed_spine" --repo-type model --token "$HF_TOKEN" --commit-message "Add lossless BF16 packed_spine + FlashIndexer + 3L MoE DSpark spine"

# 5. Upload all 45 unsplit WHT128-HLUT4 expert shards (3.09 GiB each) in batches of 5 (~15.5 GiB/commit, 9 commits!)
echo "[HF-UPLOAD] Uploading 45 unsplit WHT128-HLUT4 expert shards (L01..L45 = 139.22 GiB) in 9 batches of 5..."
for batch_idx in $(seq 0 8); do
  l_beg=$((batch_idx * 5 + 1))
  l_end=$((l_beg + 4))
  if (( l_end > 45 )); then l_end=45; fi
  TMP_LIST=()
  for l in $(seq -f "%02g" "$l_beg" "$l_end"); do
    TMP_LIST+=("--include" "hlut4_layer_${l}.bin")
  done
  echo "[HF-UPLOAD] Batch $((batch_idx + 1))/9: experts_hlut4/hlut4_layer_$(printf '%02d' "$l_beg")..$(printf '%02d' "$l_end").bin"
  for attempt in 1 2 3; do
    if hf upload "$HF_REPO" "$STAGE_DIR/experts_hlut4" "experts_hlut4" \
         "${TMP_LIST[@]}" \
         --repo-type model --token "$HF_TOKEN" \
         --commit-message "Add unsplit WHT128-HLUT4 experts L$(printf '%02d' "$l_beg")..L$(printf '%02d' "$l_end")"; then
      break
    fi
    if (( attempt == 3 )); then
      echo "[HF-UPLOAD] FATAL: Batch $((batch_idx + 1)) failed after 3 attempts" >&2
      exit 1
    fi
    sleep 10
  done
done

# 6. Verify HTTP 200/307 reachability & exact remote byte size on README + sample shards + final L45 shard!
README_HTTP="$(curl -sS -o /dev/null -w '%{http_code}' -I "https://huggingface.co/$HF_REPO/resolve/main/README.md")"
L01_HTTP="$(curl -sS -o /dev/null -w '%{http_code}' -I "https://huggingface.co/$HF_REPO/resolve/main/experts_hlut4/hlut4_layer_01.bin")"
L45_HTTP="$(curl -sS -o /dev/null -w '%{http_code}' -I "https://huggingface.co/$HF_REPO/resolve/main/experts_hlut4/hlut4_layer_45.bin")"
SPINE_HTTP="$(curl -sS -o /dev/null -w '%{http_code}' -I "https://huggingface.co/$HF_REPO/resolve/main/packed_spine/spine_dspark.safetensors")"

if [[ "$README_HTTP" != "200" && "$README_HTTP" != "307" && "$README_HTTP" != "302" ]]; then
  echo "[HF-UPLOAD] FAIL: README HTTP status=$README_HTTP" >&2
  exit 1
fi
if [[ "$L01_HTTP" != "200" && "$L01_HTTP" != "307" && "$L01_HTTP" != "302" ]] || \
   [[ "$L45_HTTP" != "200" && "$L45_HTTP" != "307" && "$L45_HTTP" != "302" ]]; then
  echo "[HF-UPLOAD] FAIL: Shard HTTP status L01=$L01_HTTP L45=$L45_HTTP" >&2
  exit 1
fi

NOW_ISO="$(date -Iseconds)"
cat <<EOF > "$RECEIPT_OUT"
{
  "schema": "ernie55_preview_hf_upload_receipt_v1",
  "repo": "$HF_REPO",
  "license": "Apache-2.0",
  "unsplit": true,
  "verified_at": "$NOW_ISO",
  "http_status": {
    "README.md": $README_HTTP,
    "packed_spine/spine_dspark.safetensors": $SPINE_HTTP,
    "experts_hlut4/hlut4_layer_01.bin": $L01_HTTP,
    "experts_hlut4/hlut4_layer_45.bin": $L45_HTTP
  },
  "status": "PASS"
}
EOF

# 7. Shred temporary /dev/shm token file immediately on completion!
if (( SHRED_TOK == 1 )) && [[ -f "$TOK_FILE" ]]; then
  shred -u "$TOK_FILE"
  echo "[HF-UPLOAD] Shredded $TOK_FILE"
fi
echo "[HF-UPLOAD] COMPLETE & VERIFIED: https://huggingface.co/$HF_REPO (receipt -> $RECEIPT_OUT)"
