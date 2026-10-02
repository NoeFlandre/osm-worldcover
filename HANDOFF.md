# WorldCover pause handoff

**Snapshot:** 2026-10-02 07:54:45 UTC. This is a recovery handoff, not a final dataset release. At the process check for this handoff there was no active WorldCover builder, supervisor, finalizer, test runner, mutation run, or HF upload. No build, test, mutation run, or data upload was started during this pause. No local data or branches were deleted.

## Code and quality status

- Current `main`: `5e492349f5904cfd3f981bffbfa81c64a7ad2d7a` (PR #12 merge). The earlier approved PR #8 provenance commit is `3ddd472e7deac10d116fd763cbf612e0d1a9c8db`.
- PR #9–#12 are merged in order: heads `7e10cd8`, `609d2f2`, `92d32dd`, `91a75fa`; merge commits `852a505e`, `b5731d57`, `99f3ffff`, `5e492349` respectively.
- PR #13 is still open/draft at `89bd7ad5e795755dce8ee9e3228b912d23653111`, based on main. No data was built with PR #13. The released checkpoints have mixed legacy/current provenance (including `3ddd472e` and `91a75fae`); keep per-receipt pins and do not assign them one code revision. Source inventory pins are listed below.
- Earlier local validation on the PR #13 tree passed `pytest --cov --cov-report=json -q`, Ruff check/format, `ty check src/`, import-boundary checks, and `mkdocs build --strict`. The strict CRAP report measured 1,111 callables, maximum 5.58, zero scores ≥6, and no allowlisted exceptions. These checks were not rerun during this pause. PR #13 has not completed a full-region output-equivalence rebuild.
- Hosted Actions run [36974757574](https://github.com/NoeFlandre/osm-worldcover/actions/runs/36974757574) on the PR #13 head completed at 2026-10-02 07:04:40 UTC. Quality gates and Docker passed; the mutation-score step failed. The report counted 4,999 generated mutants: 4,029 killed, 960 survived, 2 timeout, 7 segfault, and 1 unreported; the other categories were zero. Confirmed-kill rate is 80.596%, above the 80% floor, but the evidence is incomplete/mismatched: the export says suspicious=1 while raw metadata says suspicious=0 and unreported=1. Artifact 11213489456 is available (31,465 bytes; SHA-256 `a23707b1ac152e0a324534349c7bb94c961e7c5b15beee8c2024ce7558c25f2e`). Do not report this run as passing mutation.

## Local code preservation

The 22 changed files in `/tmp/owc-worldcover-pr13-exact` have Git blob hashes identical to PR #13 head `89bd7ad5`. Twenty other source snapshots that differ from current main were copied byte-for-byte into `recovery/local-code-snapshots/` on this branch; six previously untracked workspace recovery/upload helpers were copied into `recovery/local-helper-snapshots/workspace-root/`. Their source heads, original paths, Git blob IDs, and SHA-256 values are in `recovery/LOCAL-CODE-SNAPSHOT-MANIFEST.json`. These snapshots also exist in the hash-verified archives on `codex/incomplete-recovery-backup-20261001`: #05 record-dedup (`ad145462cc20625626a348b9319a2845b100b2368b4ff47804e142401985e2eb`), #06 release-3ddd472 (`27a1817088a254729efd70358b7fc1a86415c6376891fbb62ccbd7a780cbe4a4`), #08 Alaska bounded extraction (`84ce62bce806b09ea55170f34cb9395a92b1d8dca51c897da76e5067963ca17a`). Their archive payloads were checked against the respective 20 local code files. No unique source-code changes remain only on disk. Untracked virtualenv and preflight log/region-list artifacts were not copied. A temporary verification worktree had staged tracked-file deletions already present; it was left untouched, and its Git objects remain available.

## HF recovery checkpoint

All files below are local staged recovery data, not release outputs. Each source inventory has 386 regions. The accepted deduplication policy keeps distinct polygon identities with matching normalized text and label; only repeated records with the same stable polygon identity, normalized text, and label are removed. Residual identical text across splits remains a diagnostic.

| Source | HF dataset | Local stage | Parquet pending | Last hash-verified remote prefix revision |
| --- | --- | --- | ---: | --- |
| Description | `NoeFlandre/osm-polygon-description-tag-worldcover` | `/workspace/worldcover-backup-20261001/hf-staging/description` | 2,181 / 1,788,666,224 bytes | `1b146b5dcb92bb7304fe9cd69124d61b11564d67` |
| Website | `NoeFlandre/osm-polygon-website-tag-worldcover` | `/workspace/worldcover-backup-20261001/hf-staging/website` | 2,129 / 7,939,031,097 bytes | `dacfdea30a045cff244be79a7e7ce462a0fa0d95` |
| Wikidata and Wikipedia | `NoeFlandre/osm-wikidata-worldcover` | `/workspace/worldcover-backup-20261001/hf-staging/wikidata` | 373 / 1,446,140,166 bytes | `e59c5f01bbe59864b78b09eb2b5ffbbefbfe3ef1` |

Every remote prefix is `incomplete-recovery/2026-10-01-backup/`. At the last verified readback, remote Parquet count was zero in all three repos; 4,683 Parquets / 11,173,837,487 bytes remain staged locally. The local Parquet files match the staged manifests by size and SHA-256. The metadata/control-file readbacks were verified; ordinary small-file Git blob checks reported 877, 1,002, and 131 verified files respectively. The remote incomplete status commits above followed initial metadata commits `eebdf1bae437bfb0b23b1ebcee9c2151ee7b90ae`, `9da6f16d659b529c86b5feb344ff88c539214526`, and `0dfee76c07fdc0fd3a14b28bb9e51a843f5feb1e`. These are last verified commits from 2026-10-01, not claims of a fresh Oct 2 commit readback.

Local manifest hashes (SHA-256):

- Description: `BACKUP-MANIFEST.json` `d534589abdc311897140f98c0664faff43c0bab75d6d64a179b0547215f8edb9`; `REMOTE-STAGING-STATUS.json` `c536c8e2e2e5013a3a072dcf859909bd8cb4d472248e1354240cc85fce6a1f80`.
- Website: `BACKUP-MANIFEST.json` `67e919ae73f330c2c97a5f88cecc4152ee5851881dc85dc61179de31cd076d2a`; `REMOTE-STAGING-STATUS.json` `1d5f2cd4df1a5acb34a139f9869d672baa93f13b8a8472cbf89d9a9194258d38`.
- Wikidata: `BACKUP-MANIFEST.json` `5cb41409ebf5ab553f30830203a1b20cf0239ef87b72f3e296b6a6fa80b8d8de`; `REMOTE-STAGING-STATUS.json` `ed6b0993fad3ff6885d1129d12f73852c85e75b67968e978d5704b43b232b189`.

The original checkpoint root is `/workspace/worldcover-release-3ddd472/full/{description,website,wikidata}`; staging and manifest root is `/workspace/worldcover-backup-20261001/hf-staging/{description,website,wikidata}`. No receipt was deleted. Old run-plan counts are stale; reconcile from the durable per-region receipts before resuming.

## 403 and environment inheritance

The last surviving `HTTP 403 Forbidden` trace is in `/workspace/worldcover-backup-20261001/hf-upload.log`, whose last-write time is **2026-10-01T17:50:58.539026Z**. The log has no per-response timestamp; this is the exact latest log-write timestamp, not an independently recorded server-event time. The exception category is `httpcore.ProxyError` / `httpx.ProxyError`, with reason text `403 Forbidden`; response headers/body were not captured, so no more specific proxy-denial header can be reported. No signed URL parameters or credentials are included here. This log write predates the user-reported saved-environment update at 2026-10-01 18:33 UTC. Read-only HF metadata access succeeded on 2026-10-02, but that does not exercise the LFS upload host. The current task context exposes no running environment ID/revision or allowlist, and no post-update request was attempted; inheritance of the saved update is therefore unconfirmed. The saved environment already includes `hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com`; there is no evidence to add it again.

## Resume instructions

Keep this task stopped. When the user resumes, start a fresh task from the saved environment updated on Oct 1 and first verify its environment revision and allowed route without changing policy. Then use only the existing manifest-driven upload helper `/workspace/upload_recovery_backup.py` against the three existing local stage folders and existing incomplete prefixes. It checks the HF identity, skips paths already present, sends pending files in 150-file batches, records progress per source, and verifies remote size/hash plus selected readbacks. Do not run builders or finalizers. Do not mark these prefixes as final datasets; do not delete or replace existing files. If the upload host is denied, stop the sender and report the proxy response without bypassing it.
