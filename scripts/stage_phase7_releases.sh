#!/usr/bin/env bash
set -euo pipefail
# Phase 7 Zero-PII Release Staging, Freeze & GitHub Publisher for
# baidu/ERNIE-5.5-Flash-Preview WHT128-HLUT4 (v1.0.0-preview-hlut4-192g, Apache-2.0).
# Pure POSIX Bash + C++17 verifier (zero Python anywhere).

ROOT="${ERNIE55_ROOT:-.}"
GH_STAGE="$ROOT/release_github_staging"
HF_STAGE="$ROOT/release_hf_staging"
FREEZE_DIR="$ROOT/FREEZE_ERNIE55_PREVIEW_HLUT4_192G_20261124"
GH_REPO="${GH_REPO:-bespokeontology/ernie55-flash-preview-wht128-hlut4}"
GH_TAG="${GH_TAG:-v1.0.0-preview-hlut4-192g}"
DO_PUBLISH="${PUBLISH_GITHUB:-0}"

cd "$ROOT"

echo "=== [PHASE 7.1] Staging Sanitized GitHub Release -> $GH_STAGE ==="
rm -rf "$GH_STAGE"
mkdir -p "$GH_STAGE"/{src,mi50,p150/kernels,tools,config,scripts,docs,artifacts}

# Copy pure-C++/HIP/TT-Metal source, build files, topology configs, and docs
cp -a README.md LICENSE NOTICE CITATION.cff Makefile "$GH_STAGE/"
cp -a docs/ERNIE55_PREVIEW_HLUT4_ISO_DEMING_REPORT.md "$GH_STAGE/docs/"
cp -a src/*.hpp src/*.cpp "$GH_STAGE/src/"
cp -a mi50/*.hpp mi50/*.hip mi50/*.cpp "$GH_STAGE/mi50/"
cp -a p150/*.hpp p150/*.cpp p150/build.sh "$GH_STAGE/p150/"
cp -a p150/kernels/* "$GH_STAGE/p150/kernels/"
cp -a tools/*.cpp "$GH_STAGE/tools/"
cp -a config/topology_192g.json config/topology_224g.json config/topology_240g.json "$GH_STAGE/config/"
cp -a scripts/start_p150_worker.sh scripts/start_ernie55_fe.sh scripts/upload_hf_ernie55_preview.sh "$GH_STAGE/scripts/"

# Copy & path-sanitize GATE-0..GATE-6 receipts + telemetry PNG (IHDR/IDAT/IEND only!)
for j in \
  baseline_targets.json \
  cal_disjoint_receipt.json \
  hlut4_preview_pack_manifest.json \
  gate1_ernie55_preview_hlut4_1m.json \
  gate2_ernie55_preview_hlut4_1m.json \
  gate3_ernie55_preview_hlut4_1m.json \
  gate4_ernie55_preview_hlut4_1m.json \
  gate5_live_8052_3081_receipt.json \
  gate6_long_horizon_1m_campaign.json; do
  if [[ -f "artifacts/$j" ]]; then
    sed "s|$ROOT/||g; s|$ROOT|.|g" "artifacts/$j" > "$GH_STAGE/artifacts/$j"
  fi
done
cp -a artifacts/ernie55_preview_hlut4_1m_telemetry.png "$GH_STAGE/artifacts/"

# Generate deterministic SHA256SUMS.txt inside GitHub staging tree
(
  cd "$GH_STAGE"
  find . -type f ! -name 'SHA256SUMS.txt' | LC_ALL=C sort | sed 's|^\./||' | xargs sha256sum > SHA256SUMS.txt
)

# Run mandatory 10-Layer Zero-PII Gate on GitHub staging tree!
"$ROOT/bin/ernie55_sanitize_verifier" "$GH_STAGE" --mode github

# Build clean single-commit orphan git repository (bespokeontology <noreply@github.com>)
(
  cd "$GH_STAGE"
  git init -b main >/dev/null
  git config user.name "bespokeontology"
  git config user.email "noreply@github.com"
  git add -A
  git commit -m "Release $GH_TAG: ERNIE-5.5-Flash-Preview (300B + 3L MoE DSpark) 100% WHT128-HLUT4 All-VRAM 1M Engine" >/dev/null
  git tag -a "$GH_TAG" -m "$GH_TAG (10/10 Zero-PII Verified, Apache-2.0)"
)
echo "[PHASE 7.1] Orphan GitHub staging repo ready & 10/10 verified at $GH_STAGE"

echo "=== [PHASE 7.2] Staging Unsplit Hugging Face Repo -> $HF_STAGE ==="
rm -rf "$HF_STAGE"
mkdir -p "$HF_STAGE"/{meta,packed_spine,experts_hlut4}

cp -a HF_README.md "$HF_STAGE/README.md"
cp -a LICENSE NOTICE CITATION.cff "$HF_STAGE/"
cp -a artifacts/ernie55_preview_hlut4_1m_telemetry.png "$HF_STAGE/"
cp -a checkpoints/preview-bf16/router_perm.bin "$HF_STAGE/"
for mf in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors.index.json flashindexer.safetensors; do
  if [[ -f "meta/preview-bf16/$mf" ]]; then
    cp -a "meta/preview-bf16/$mf" "$HF_STAGE/meta/$mf"
  fi
done
# Symlink unsplit packed_spine/ and experts_hlut4/ (zero disk duplication, zero .part splits!)
for sf in "$ROOT"/checkpoints/preview-bf16/packed_spine/*.safetensors; do
  ln -sfn "$sf" "$HF_STAGE/packed_spine/$(basename "$sf")"
done
for ef in ./weights_hlut4/experts_hlut4/hlut4_layer_*.bin; do
  ln -sfn "$ef" "$HF_STAGE/experts_hlut4/$(basename "$ef")"
done

# Generate complete SHA256SUMS.txt across docs, meta, packed_spine, and all 45 unsplit HLUT4 shards!
(
  cd "$HF_STAGE"
  {
    sha256sum README.md LICENSE NOTICE CITATION.cff ernie55_preview_hlut4_1m_telemetry.png router_perm.bin
    find meta -type f | LC_ALL=C sort | xargs sha256sum
    find packed_spine -maxdepth 1 \( -type f -o -type l \) -name '*.safetensors' | LC_ALL=C sort | xargs sha256sum
    find experts_hlut4 -maxdepth 1 \( -type f -o -type l \) -name 'hlut4_layer_*.bin' | LC_ALL=C sort | xargs sha256sum
  } > SHA256SUMS.txt
)

# Run mandatory 10-Layer Zero-PII Gate on HF staging tree!
"$ROOT/bin/ernie55_sanitize_verifier" "$HF_STAGE" --mode hf
echo "[PHASE 7.2] Unsplit Hugging Face staging tree ready & 10/10 verified at $HF_STAGE"

echo "=== [PHASE 7.3] Locking Read-Only Freeze -> $FREEZE_DIR ==="
if [[ -d "$FREEZE_DIR" ]]; then
  echo "[PHASE 7.3] Freeze dir already exists ($FREEZE_DIR) — leaving untouched!"
else
  mkdir -p "$FREEZE_DIR"/{src,mi50,p150/kernels,tools,config,scripts,docs,artifacts,bin}
  cp -a "$GH_STAGE"/{README.md,LICENSE,NOTICE,CITATION.cff,Makefile} "$FREEZE_DIR/"
  cp -a "$GH_STAGE"/docs/* "$FREEZE_DIR/docs/"
  cp -a "$GH_STAGE"/src/* "$FREEZE_DIR/src/"
  cp -a "$GH_STAGE"/mi50/* "$FREEZE_DIR/mi50/"
  cp -a "$GH_STAGE"/p150/* "$FREEZE_DIR/p150/"
  cp -a "$GH_STAGE"/tools/* "$FREEZE_DIR/tools/"
  cp -a "$GH_STAGE"/config/* "$FREEZE_DIR/config/"
  cp -a "$GH_STAGE"/scripts/* "$FREEZE_DIR/scripts/"
  cp -a "$GH_STAGE"/artifacts/* "$FREEZE_DIR/artifacts/"
  cp -a "$ROOT"/bin/{ernie55_fe,ernie55_build_weights,ernie55_build_weights_gpu,ernie55_build_cal_corpus,ernie55_swap_checkpoint,ernie55_verify_unit,ernie55_verify_gates,ernie55_gate6_campaign,ernie55_chart_gen,ernie55_sanitize_verifier,p150_ernie55_worker} "$FREEZE_DIR/bin/"
  (
    cd "$FREEZE_DIR"
    find . -type f ! -name 'SHA256SUMS_FREEZE.txt' | LC_ALL=C sort | sed 's|^\./||' | xargs sha256sum > SHA256SUMS_FREEZE.txt
  )
  chmod -R a-w "$FREEZE_DIR"
  echo "[PHASE 7.3] Locked read-only freeze at $FREEZE_DIR (chmod -R a-w)"
fi

if [[ "$DO_PUBLISH" == "1" ]]; then
  echo "=== [PHASE 7.4] Publishing Sanitized GitHub Repo & Release $GH_REPO ($GH_TAG) ==="
  gh auth status
  if ! gh repo view "$GH_REPO" >/dev/null 2>&1; then
    gh repo create "$GH_REPO" --public --description "Pure-C++17/20 gfx906 HIP & Tenstorrent Blackhole P150A WHT128-HLUT4 (4.125 bpw) 1M Engine for baidu/ERNIE-5.5-Flash-Preview (300B + 3L MoE DSpark, Apache-2.0)"
  fi
  (
    cd "$GH_STAGE"
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://github.com/$GH_REPO.git"
    git push -u origin main --force
    git push origin "$GH_TAG" --force
  )
  if gh release view "$GH_TAG" --repo "$GH_REPO" >/dev/null 2>&1; then
    gh release delete "$GH_TAG" --repo "$GH_REPO" --yes
  fi
  gh release create "$GH_TAG" \
    --repo "$GH_REPO" \
    --title "ERNIE-5.5-Flash-Preview WHT128-HLUT4 ($GH_TAG)" \
    --notes-file "$GH_STAGE/README.md" \
    "$GH_STAGE/artifacts/ernie55_preview_hlut4_1m_telemetry.png" \
    "$GH_STAGE/docs/ERNIE55_PREVIEW_HLUT4_ISO_DEMING_REPORT.md" \
    "$GH_STAGE/SHA256SUMS.txt" \
    "$GH_STAGE/artifacts/gate1_ernie55_preview_hlut4_1m.json" \
    "$GH_STAGE/artifacts/gate2_ernie55_preview_hlut4_1m.json" \
    "$GH_STAGE/artifacts/gate3_ernie55_preview_hlut4_1m.json" \
    "$GH_STAGE/artifacts/gate4_ernie55_preview_hlut4_1m.json" \
    "$GH_STAGE/artifacts/gate5_live_8052_3081_receipt.json" \
    "$GH_STAGE/artifacts/gate6_long_horizon_1m_campaign.json"

  echo "=== [PHASE 7.4] Verifying HTTP/2 200 on GitHub Release Assets ==="
  for asset in \
    ernie55_preview_hlut4_1m_telemetry.png \
    ERNIE55_PREVIEW_HLUT4_ISO_DEMING_REPORT.md \
    SHA256SUMS.txt \
    gate1_ernie55_preview_hlut4_1m.json \
    gate2_ernie55_preview_hlut4_1m.json \
    gate3_ernie55_preview_hlut4_1m.json \
    gate4_ernie55_preview_hlut4_1m.json \
    gate5_live_8052_3081_receipt.json \
    gate6_long_horizon_1m_campaign.json; do
    code="$(curl -sSL -o /dev/null -w '%{http_code}' -I "https://github.com/$GH_REPO/releases/download/$GH_TAG/$asset")"
    echo "  HTTP $code -> $asset"
    test "$code" = "200"
  done
  echo "[PHASE 7.4] GitHub Release $GH_REPO ($GH_TAG) LIVE & HTTP 200 VERIFIED!"
fi
