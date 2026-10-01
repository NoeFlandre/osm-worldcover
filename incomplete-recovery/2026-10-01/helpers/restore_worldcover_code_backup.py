"""Reconstruct preserved code archives from a checked-out backup branch."""
from pathlib import Path
import argparse, hashlib, json

parser=argparse.ArgumentParser()
parser.add_argument('backup_directory',type=Path)
parser.add_argument('output_directory',type=Path)
args=parser.parse_args()
manifest=json.loads((args.backup_directory/'CODE-BACKUP-MANIFEST.json').read_text())
args.output_directory.mkdir(parents=True,exist_ok=True)
for item in manifest['archives']:
    content=b''.join((args.backup_directory/name).read_bytes() for name in item['parts'])
    if len(content)!=item['bytes'] or hashlib.sha256(content).hexdigest()!=item['sha256']:
        raise ValueError('Archive verification failed: '+item['file'])
    target=args.output_directory/item['file']
    with target.open('xb') as out:out.write(content)
    print('Verified and reconstructed',target)
