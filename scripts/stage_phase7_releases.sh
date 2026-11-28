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
# Wave-13 Cut W13.6: default to the Wave-13 1M prefill milestone tag. Override GH_TAG to
# stage an earlier milestone (for example v1.0.0-preview-hlut4-192g) without editing the script.
GH_TAG="${GH_TAG:-v1.5.0-wave13-1k-prefill-1m}"
DO_PUBLISH="${PUBLISH_GITHUB:-0}"

cd "$ROOT"

echo "=== [PHASE 7.1] Staging Sanitized GitHub Release -> $GH_STAGE ==="
rm -rf "$GH_STAGE"
mkdir -p "$GH_STAGE"/{src,mi50,p150/kernels,tools,config,scripts,docs,artifacts}

# Copy pure-C++/HIP/TT-Metal source, build files, topology configs, and docs
cp -a README.md LICENSE NOTICE CITATION.cff Makefile "$GH_STAGE/"
cp -a docs/ERNIE55_PREVIEW_HLUT4_ISO_DEMING_REPORT.md "$GH_STAGE/docs/"
# Wave-13 Cut W13.6: stage the complete Wave-13 design and signoff ledger set.
for w13doc in docs/WAVE13_*.md; do
  if [[ -f "$w13doc" ]]; then
    cp -a "$w13doc" "$GH_STAGE/docs/"
  fi
done
cp -a src/*.hpp src/*.cpp "$GH_STAGE/src/"
cp -a mi50/*.hpp mi50/*.hip mi50/*.cpp "$GH_STAGE/mi50/"
cp -a p150/*.hpp p150/*.cpp p150/build.sh "$GH_STAGE/p150/"
cp -a p150/kernels/* "$GH_STAGE/p150/kernels/"
cp -a tools/*.cpp "$GH_STAGE/tools/"
# Wave-13 Cut W13.6: the Wave-13 chart renderer and sanitizer are part of the release surface.
cp -a tools/ernie55_chart_gen.cpp tools/ernie55_sanitize_verifier.cpp "$GH_STAGE/tools/" 
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
  gate6_long_horizon_1m_campaign.json \
  gate_wave13_1k_prefill_1m_receipt.json; do
  if [[ -f "artifacts/$j" ]]; then
    sed "s|$ROOT/||g; s|$ROOT|.|g" "artifacts/$j" > "$GH_STAGE/artifacts/$j"
  fi
done
# Wave-13 Cut W13.9: stage the session replay artifacts.
# Text markdown is path-sanitized with sed; binary .zip archives are copied verbatim with
# cp -a, because running sed over a binary archive corrupts it. Both the *-1m-* and the
# *-wave13-* replay/sanitized names are matched so a new session id is picked up cleanly.
for rp_md in artifacts/dsh-session-*-1m-replay.md artifacts/dsh-session-*-wave13-replay.md; do
  if [[ -f "$rp_md" ]]; then
    sed "s|$ROOT/||g; s|$ROOT|.|g" "$rp_md" > "$GH_STAGE/artifacts/$(basename "$rp_md")"
  fi
done
for rp_zip in artifacts/dsh-session-*-1m-sanitized.zip artifacts/dsh-session-*-wave13-sanitized.zip; do
  if [[ -f "$rp_zip" ]]; then
    cp -a "$rp_zip" "$GH_STAGE/artifacts/$(basename "$rp_zip")"
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

# Wave-13 Cut W13.9: collect the staged replay artifacts (markdown + sanitized zip) so they
# can be attached to the release and HTTP-verified. Binary zips are never passed through sed.
# Wave-13 Cut W13.13: build the release asset list from files that actually exist, so
# absent optional receipts (gate3/gate5) are skipped cleanly and never passed to
# `gh release create`.
RELEASE_ASSETS=()
RELEASE_ASSET_NAMES=()
for ra in "$GH_STAGE/artifacts/ernie55_preview_hlut4_1m_telemetry.png" \
          "$GH_STAGE/docs/ERNIE55_PREVIEW_HLUT4_ISO_DEMING_REPORT.md" \
          "$GH_STAGE/SHA256SUMS.txt" \
          "$GH_STAGE/artifacts/gate1_ernie55_preview_hlut4_1m.json" \
          "$GH_STAGE/artifacts/gate2_ernie55_preview_hlut4_1m.json" \
          "$GH_STAGE/artifacts/gate3_ernie55_preview_hlut4_1m.json" \
          "$GH_STAGE/artifacts/gate4_ernie55_preview_hlut4_1m.json" \
          "$GH_STAGE/artifacts/gate5_live_8052_3081_receipt.json" \
          "$GH_STAGE/artifacts/gate6_long_horizon_1m_campaign.json" \
          "$GH_STAGE/artifacts/gate_wave13_1k_prefill_1m_receipt.json" \
          "$GH_STAGE/docs/WAVE13_1K_PREFILL_1M_SIGNOFF.md"; do
  if [[ -f "$ra" ]]; then
    RELEASE_ASSETS+=("$ra")
    RELEASE_ASSET_NAMES+=("$(basename "$ra")")
  fi
done
REPLAY_ASSETS=()
REPLAY_ASSET_NAMES=()
for ra in "$GH_STAGE"/artifacts/dsh-session-*-1m-replay.md \
          "$GH_STAGE"/artifacts/dsh-session-*-wave13-replay.md \
          "$GH_STAGE"/artifacts/dsh-session-*-1m-sanitized.zip \
          "$GH_STAGE"/artifacts/dsh-session-*-wave13-sanitized.zip; do
  if [[ -f "$ra" ]]; then
    REPLAY_ASSETS+=("$ra")
    REPLAY_ASSET_NAMES+=("$(basename "$ra")")
  fi
done

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
    "${RELEASE_ASSETS[@]}" \
    "${REPLAY_ASSETS[@]}"

  echo "=== [PHASE 7.4] Verifying HTTP/2 200 on GitHub Release Assets ==="
  for asset in "${RELEASE_ASSET_NAMES[@]}" "${REPLAY_ASSET_NAMES[@]}"; do
    code="$(curl -sSL -o /dev/null -w '%{http_code}' -I "https://github.com/$GH_REPO/releases/download/$GH_TAG/$asset")"
    echo "  HTTP $code -> $asset"
    test "$code" = "200"
  done
  echo "[PHASE 7.4] GitHub Release $GH_REPO ($GH_TAG) LIVE & HTTP 200 VERIFIED!"
fi
