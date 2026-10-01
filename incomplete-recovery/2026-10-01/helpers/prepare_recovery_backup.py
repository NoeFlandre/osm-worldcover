from pathlib import Path
import collections, hashlib, json, os, re, subprocess, tarfile

BASE=Path('/workspace/worldcover-backup-20261001')
ROOT=Path('/workspace/worldcover-release-3ddd472')
BASE.mkdir(exist_ok=True)
CODE=BASE/'code'; CODE.mkdir(exist_ok=True)
secret=re.compile(rb'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})')
def safe(p):
    if p.stat().st_size < 20_000_000 and p.suffix not in ('.parquet','.bundle','.gz'):
        if secret.search(p.read_bytes()):
            raise RuntimeError('credential-shaped literal in '+str(p))
def git(p,*args):
    return subprocess.check_output(['git','-C',str(p),*args])
repos=sorted(set(p for base in (Path('/workspace'),Path('/tmp')) for p in base.iterdir() if ('worldcover' in p.name or p.name.startswith('owc-')) and p.is_dir() and (p/'.git').exists()))
inventory=[]
common=set()
for i,p in enumerate(repos):
    names=git(p,'ls-files','--cached','--others','--exclude-standard','-z').decode().split('\0')
    files=[p/n for n in names if n and (p/n).is_file() and '.venv' not in Path(n).parts]
    for f in files: safe(f)
    name=f'{i:02d}-{p.name}'
    archive=CODE/(name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for f in files: tar.add(f,arcname=str(f.relative_to(p)),recursive=False)
    inventory.append({'workspace':str(p),'head':git(p,'rev-parse','HEAD').decode().strip(),'status':git(p,'status','--short').decode(),'snapshot':archive.name,'files':len(files)})
    gd=git(p,'rev-parse','--git-common-dir').decode().strip()
    resolved=(p/gd).resolve()
    if resolved not in common:
        common.add(resolved)
        subprocess.run(['git','-C',str(p),'bundle','create',str(CODE/(name+'.bundle')),'--all','HEAD'],check=True,capture_output=True)
helpers=[]
for p in ROOT.rglob('*'):
    if p.is_file() and p.suffix=='.py' and '__pycache__' not in p.parts:
        safe(p);helpers.append((p,'release-workspace/'+str(p.relative_to(ROOT))))
for p in Path('/tmp').glob('owc-*.py'):
    safe(p);helpers.append((p,'temporary-helpers/'+p.name))
for p in Path('/workspace').glob('worldcover-*.md'):
    safe(p);helpers.append((p,'review-evidence/'+p.name))
with tarfile.open(CODE/'recovery-helpers.tar.gz','w:gz') as tar:
    for p,name in helpers:tar.add(p,arcname=name,recursive=False)
(CODE/'CODE-INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
# GitHub blob chunks remain comfortably below tool response limits.
parts=BASE/'github-parts';parts.mkdir(exist_ok=True)
cm=[]
for p in sorted(CODE.iterdir()):
    b=p.read_bytes(); chunks=[]
    for i in range(0,len(b),32768):
        q=parts/f'{p.name}.part-{i//32768:04d}';q.write_bytes(b[i:i+32768]);chunks.append(q.name)
    cm.append({'file':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'parts':chunks})
(parts/'CODE-BACKUP-MANIFEST.json').write_text(json.dumps({'status':'incomplete recovery backup; no merges or publication readiness implied','archives':cm,'repositories':inventory},indent=2)+'\n')
# Preserve all complete files, including earlier partial assemblies and source parquet caches.
# Raster tiles, auth/cache state, temporary bytecode/locks and unfinished .part files are reproducible or transient, not completed outputs.
stages={s:BASE/'hf-staging'/s for s in ('description','website','wikidata')}
records={s:[] for s in stages}
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if any(x.startswith('hf-home') or x=='__pycache__' for x in rel.parts):continue
    if p.suffix in ('.tif','.tiff','.pyc','.lock','.part','.metadata') or p.name=='CACHEDIR.TAG':continue
    safe(p)
    source=next((s for s in stages if any(x==s or x.startswith(s+'-') or x.startswith(s+'.') for x in rel.parts)),None)
    targets=[source] if source else list(stages)
    digest=hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size < 20_000_000 else None
    if digest is None:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        digest=h.hexdigest()
    for s in targets:
        q=stages[s]/'workspace'/rel;q.parent.mkdir(parents=True,exist_ok=True)
        if not q.exists():os.link(p,q)
        records[s].append({'path':str(q.relative_to(stages[s])),'original':str(p),'bytes':p.stat().st_size,'sha256':digest})
for s,stage in stages.items():
    data={'status':'INCOMPLETE RECOVERY BACKUP — NOT A FINAL DATASET RELEASE','source':s,'release_eligible':False,'created_utc':'2026-10-01','files':records[s],
          'excluded':'ESA raster tiles; HF auth/cache directories; bytecode; lock files; temporary .part and download metadata','code_backup_branch':'codex/incomplete-recovery-backup-20261001'}
    (stage/'BACKUP-MANIFEST.json').write_text(json.dumps(data,indent=2)+'\n')
    (stage/'RECOVERY-README.md').write_text('# Incomplete recovery backup\n\nThese are saved checkpoints, prior partial outputs and audit evidence, not a complete published release. No input was deleted. Preserve receipt code/source provenance when resuming. PR13 Andalucía is experimental and separate from release inputs.\n')
    print(s,len(records[s]),sum(x['bytes'] for x in records[s]))
print('code',len(cm),'chunks',sum(len(x['parts']) for x in cm))
