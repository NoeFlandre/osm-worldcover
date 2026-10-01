import os
os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
from pathlib import Path
import hashlib,json
from huggingface_hub import HfApi,CommitOperationAdd,get_token,hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError
BASE=Path('/workspace/worldcover-backup-20261001');PREFIX='incomplete-recovery/2026-10-01-backup'
REPOS={'description':'NoeFlandre/osm-polygon-description-tag-worldcover','website':'NoeFlandre/osm-polygon-website-tag-worldcover','wikidata':'NoeFlandre/osm-wikidata-worldcover'}
api=HfApi(token=get_token());assert api.whoami()['name']=='NoeFlandre';results={}
for source,repo in REPOS.items():
 stage=BASE/'hf-staging'/source
 local=[p for p in stage.rglob('*') if p.is_file() and p.suffix!='.parquet']
 controls={stage/'BACKUP-MANIFEST.json',stage/'REMOTE-STAGING-STATUS.json',stage/'RECOVERY-README.md'}
 controls={p for p in controls if p.exists()}
 regular=[p for p in local if p not in controls]
 try: remote={x.path:x for x in api.list_repo_tree(repo,repo_type='dataset',path_in_repo=PREFIX,recursive=True) if getattr(x,'size',None) is not None}
 except RemoteEntryNotFoundError: remote={}
 pending=[p for p in regular if PREFIX+'/'+str(p.relative_to(stage)).replace('/.cache/','/download-cache-metadata/') not in remote]
 commits=[];print(source,'regular_files_pending',len(pending),'parquet_pending',sum(1 for p in stage.rglob('*.parquet')),flush=True)
 for start in range(0,len(pending),100):
  batch=pending[start:start+100];head=api.repo_info(repo,repo_type='dataset').sha
  ops=[CommitOperationAdd(path_in_repo=PREFIX+'/'+str(p.relative_to(stage)).replace('/.cache/','/download-cache-metadata/'),path_or_fileobj=str(p)) for p in batch]
  commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message=f'Back up incomplete {source} recovery receipts and evidence ({start+1}-{start+len(batch)}/{len(pending)})',num_threads=2)
  commits.append(commit.oid);print(source,'uploaded',start+len(batch),'/',len(pending),'commit',commit.oid,flush=True)
 # Commit the explicit incomplete status only after all ordinary git-backed evidence landed.
 head=api.repo_info(repo,repo_type='dataset').sha
 commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,operations=[CommitOperationAdd(path_in_repo=PREFIX+'/'+p.name,path_or_fileobj=str(p)) for p in sorted(controls)],commit_message=f'Mark {source} recovery backup incomplete while Parquet LFS is blocked',num_threads=1)
 commits.append(commit.oid);pin=api.repo_info(repo,repo_type='dataset').sha
 manifest=json.loads((stage/'BACKUP-MANIFEST.json').read_text());expected={x['path'].replace('/.cache/','/download-cache-metadata/'):x for x in manifest['files'] if not x['path'].endswith('.parquet')}
 remote={x.path:x for x in api.list_repo_tree(repo,repo_type='dataset',revision=pin,path_in_repo=PREFIX,recursive=True) if getattr(x,'size',None) is not None}
 verified=[]
 for rel,x in expected.items():
  path=PREFIX+'/'+rel;obj=remote.get(path)
  if obj is None or obj.size!=x['bytes']:raise RuntimeError(f'remote metadata size/path mismatch {source}/{rel}')
  expected_git=hashlib.sha1(b'blob '+str(x['bytes']).encode()+b'\0')
  h=hashlib.sha256();sha1=expected_git
  with (stage/x['path']).open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):
    h.update(block);sha1.update(block)
  if h.hexdigest()!=x['sha256'] or obj.blob_id!=sha1.hexdigest():raise RuntimeError(f'remote metadata hash mismatch {source}/{rel}')
  verified.append(rel)
 control_records=[]
 for p in controls:
  obj=remote.get(PREFIX+'/'+p.name);data=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  if obj is None or obj.size!=len(data) or obj.blob_id!=blob:raise RuntimeError('remote control readback mismatch '+source+'/'+p.name)
  control_records.append({'file':p.name,'size':len(data),'git_blob_sha':obj.blob_id})
 output={'repo':repo,'revision':pin,'commits':commits,'prefix':PREFIX,'ordinary_files_verified_by_remote_git_blob_sha':len(verified),'verified_parquet_files':0,'parquet_files_pending':sum(1 for p in stage.rglob('*.parquet')),'control_files_verified':control_records,'complete_release':False}
 results[source]=output;(BASE/'HF-METADATA-CHECKPOINT-VERIFICATION.json').write_text(json.dumps(results,indent=2)+'\n')
 print(source,'VERIFIED ordinary',len(verified),'controls',len(control_records),'PARQUET_PENDING',output['parquet_files_pending'],'head',pin,flush=True)
 # A direct HTTPS readback may be denied by the current proxy; remote Git blob hashes above remain authoritative for ordinary files.
 try:
  p=stage/'RECOVERY-README.md';download=Path(hf_hub_download(repo_id=repo,repo_type='dataset',filename=PREFIX+'/'+p.name,revision=pin,token=get_token(),cache_dir=str(BASE/'readback-cache')))
  if download.read_bytes()!=p.read_bytes():raise RuntimeError('recovery README readback bytes mismatch')
  output['download_readback']='verified';(BASE/'HF-METADATA-CHECKPOINT-VERIFICATION.json').write_text(json.dumps(results,indent=2)+'\n')
 except Exception as exc:
  output['download_readback']='blocked '+type(exc).__name__+'; all files verified against remote Git object hashes'
  (BASE/'HF-METADATA-CHECKPOINT-VERIFICATION.json').write_text(json.dumps(results,indent=2)+'\n')
print('METADATA AND RECEIPT CHECKPOINTS VERIFIED; PARQUETS INCOMPLETE',flush=True)
