# Incomplete WorldCover recovery backup

This backup preserves all local WorldCover Git worktrees (including uncommitted changes), standalone review/mutation source copies, repository history bundles, recovery helpers and their exact inventories. It does not merge code or certify a final dataset release.

Code archives are split into binary chunks to fit supported connector payloads. Reconstruct them without overwriting files:

```sh
python incomplete-recovery/2026-10-01/helpers/restore_worldcover_code_backup.py incomplete-recovery/2026-10-01/code /tmp/worldcover-restored-code
```

The restore script checks SHA-256 and byte counts for every archive before writing it. CODE-BACKUP-MANIFEST.json records each original workspace, commit, working-tree status and archive. Bundles retain local Git history; tar archives preserve current source, including local changes.

Hugging Face data backups are staged for the three approved dataset repositories under incomplete-recovery/2026-10-01-backup. Their uploads currently require repository write authorization; metadata/auth reads succeed but upload/commit requests return HTTP 403. No completed backup or final release is claimed for those data paths.

All original code, data, source caches, ESA tiles and checkpoints remain intact. No branch or dataset files were deleted. Raster tiles and credential/cache state are not included in these completed-output backups.
