from pathlib import Path
import json

base=Path('/workspace/worldcover-backup-20261001')
for source in ('description','website','wikidata'):
    stage=base/'hf-staging'/source
    manifest_path=stage/'BACKUP-MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    data_files=[x for x in manifest['files'] if x['path'].endswith('.parquet')]
    metadata_files=[x for x in manifest['files'] if not x['path'].endswith('.parquet')]
    manifest['upload_status']='Incomplete metadata/checkpoint upload. Parquet LFS transfer is blocked by the environment proxy HTTP 403; no Parquet upload is confirmed.'
    manifest['uploaded_groups']={'metadata_and_receipts':'pending','parquet_data':'pending proxy access'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    status={'status':'INCOMPLETE RECOVERY BACKUP','source':source,'complete_release':False,
            'remote_prefix':'incomplete-recovery/2026-10-01-backup',
            'small_file_count':len(metadata_files)+2,'parquet_file_count':len(data_files),
            'parquet_bytes':sum(x['bytes'] for x in data_files),
            'parquet_upload_status':'pending: proxy returned HTTP 403 during Hugging Face LFS transfer',
            'code_backup_commit':'2b44d759a1700dec10f12c82dfe640968a84cabd',
            'sha256_by_remote_path':{x['remote_path']:x['sha256'] for x in manifest['files']},
            'small_file_paths':[x['remote_path'] for x in metadata_files]+[
                'incomplete-recovery/2026-10-01-backup/BACKUP-MANIFEST.json',
                'incomplete-recovery/2026-10-01-backup/RECOVERY-README.md']}
    (stage/'REMOTE-STAGING-STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(source,'metadata',len(metadata_files)+2,'parquet',len(data_files),'parquet_bytes',status['parquet_bytes'])
