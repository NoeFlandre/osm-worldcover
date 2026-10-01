from pathlib import Path
import json
base=Path('/workspace/worldcover-backup-20261001');host='hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com'
for source in ('description','website','wikidata'):
 stage=base/'hf-staging'/source
 mp=stage/'BACKUP-MANIFEST.json';m=json.loads(mp.read_text())
 m['upload_status']='Incomplete metadata/checkpoint backup committed and verified. Parquet files remain pending because the environment proxy returns HTTP 403 for the signed LFS upload host.'
 m['blocked_lfs_upload_host']=host
 mp.write_text(json.dumps(m,indent=2)+'\n')
 sp=stage/'REMOTE-STAGING-STATUS.json';s=json.loads(sp.read_text())
 s['parquet_upload_status']='pending: proxy returned HTTP 403 to the authorized signed LFS upload host'
 s['required_proxy_upload_host']=host
 s['repository_write_permission']='verified for all three repositories'
 sp.write_text(json.dumps(s,indent=2)+'\n')
 print(source,'status manifests updated')
