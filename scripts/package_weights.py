"""Assemble trained scorers and a compatible encoder into a new inference directory."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scorers', type=Path, required=True)
    ap.add_argument('--encoder', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    files = [args.encoder] + [args.scorers/n for n in ('global_teacher.joblib', 'regional_teacher.joblib', 'arbiter.joblib')]
    for p in files:
        if not p.is_file():
            ap.error('Missing weight file: ' + str(p))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for source, name in zip(files, ['encoder.pth.tar', 'global_teacher.joblib', 'regional_teacher.joblib', 'arbiter.joblib']):
        dest = args.output/name
        shutil.copy2(source, dest)
        manifest[name] = dict(sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), bytes=dest.stat().st_size)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
