import os
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
from pathlib import Path
import hashlib,json,time
from huggingface_hub import HfApi,CommitOperationAdd,get_token,hf_hub_download

BASE=Path('/workspace/worldcover-backup-20261001')
PREFIX='incomplete-recovery/2026-10-01-backup'
REPOS={'description':'NoeFlandre/osm-polygon-description-tag-worldcover','website':'NoeFlandre/osm-polygon-website-tag-worldcover','wikidata':'NoeFlandre/osm-wikidata-worldcover'}
api=HfApi(token=get_token())
assert api.whoami()['name']=='NoeFlandre'
results={}
for source,repo in REPOS.items():
    stage=BASE/'hf-staging'/source
    local=sorted(p for p in stage.rglob('*') if p.is_file())
    remote={x.path:x for x in api.list_repo_tree(repo,repo_type='dataset',path_in_repo='incomplete-recovery',recursive=True) if getattr(x,'size',None) is not None} if api.file_exists(repo,PREFIX+'/BACKUP-MANIFEST.json',repo_type='dataset') else {}
    pending=[p for p in local if PREFIX+'/'+str(p.relative_to(stage)).replace('/.cache/','/download-cache-metadata/') not in remote]
    commits=[]
    print(source,'pending',len(pending),'bytes',sum(p.stat().st_size for p in pending),flush=True)
    for start in range(0,len(pending),150):
        batch=pending[start:start+150]
        head=api.repo_info(repo,repo_type='dataset').sha
        result=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,operations=[CommitOperationAdd(path_in_repo=PREFIX+'/'+str(p.relative_to(stage)).replace('/.cache/','/download-cache-metadata/'),path_or_fileobj=str(p)) for p in batch],commit_message=f'Back up incomplete {source} recovery artifacts ({start+1}-{start+len(batch)}/{len(pending)})',num_threads=2)
        commits.append(result.oid)
        print(source,'uploaded',start+len(batch),'/',len(pending),'commit',result.oid,flush=True)
        (BASE/f'{source}-upload-progress.json').write_text(json.dumps({'repo':repo,'prefix':PREFIX,'commits':commits,'uploaded':start+len(batch),'pending_total':len(pending)},indent=2)+'\n')
    pin=api.repo_info(repo,repo_type='dataset').sha
    remote={x.path:x for x in api.list_repo_tree(repo,repo_type='dataset',revision=pin,path_in_repo=PREFIX,recursive=True) if getattr(x,'size',None) is not None}
    records=json.loads((stage/'BACKUP-MANIFEST.json').read_text())['files']
    known={x['path']:x for x in records}
    verified=[]
    for p in local:
        rel=str(p.relative_to(stage));item=remote.get(PREFIX+'/'+rel.replace('/.cache/','/download-cache-metadata/'))
        if item is None or item.size!=p.stat().st_size:raise RuntimeError('Remote size/path mismatch: '+source+'/'+rel)
        if item.lfs:
            expected=known.get(rel,{}).get('sha256') or hashlib.sha256(p.read_bytes()).hexdigest()
            actual=item.lfs.sha256
        else:
            content=p.read_bytes();expected=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest();actual=item.blob_id
        if actual!=expected:raise RuntimeError('Remote hash mismatch: '+source+'/'+rel)
        verified.append(rel)
    # Read back all backup controls, all canonical assembled splits, and one completed shard.
    readbacks=[p for p in local if p.name in ('BACKUP-MANIFEST.json','RECOVERY-README.md') or ('incomplete-mixed-' in str(p) and p.suffix=='.parquet')]
    shards=[p for p in local if p.suffix=='.parquet' and 'shards' in p.parts]
    if shards:readbacks.append(shards[0])
    samples=[]
    for p in readbacks:
        rel=str(p.relative_to(stage));download=Path(hf_hub_download(repo_id=repo,repo_type='dataset',filename=PREFIX+'/'+rel.replace('/.cache/','/download-cache-metadata/'),revision=pin,token=get_token(),cache_dir=str(BASE/'readback-cache')))
        h=hashlib.sha256()
        with download.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        expected=known.get(rel,{}).get('sha256') or hashlib.sha256(p.read_bytes()).hexdigest()
        if h.hexdigest()!=expected:raise RuntimeError('Readback mismatch: '+source+'/'+rel)
        samples.append(rel)
    results[source]={'repo':repo,'revision':pin,'prefix':PREFIX,'verified_files':len(verified),'bytes':sum(p.stat().st_size for p in local),'readback_verified':samples,'release_eligible':False}
    (BASE/'HF-BACKUP-VERIFICATION.json').write_text(json.dumps(results,indent=2)+'\n')
    print(source,'VERIFIED',len(verified),'files',len(samples),'readbacks',pin,flush=True)
print('ALL HF RECOVERY BACKUPS VERIFIED',flush=True)
