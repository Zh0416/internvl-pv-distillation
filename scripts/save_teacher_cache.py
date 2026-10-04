from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--part-size-gb", type=float, default=2.0)
    args = parser.parse_args(); source = Path(args.input); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    limit = int(args.part_size_gb * 1024**3); part = 1; current = 0; archive = None
    try:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            size = path.stat().st_size
            if archive is None or (current and current + size > limit):
                if archive: archive.close()
                archive = tarfile.open(output / f"teacher_cache_part_{part:03d}.tar.gz", "w:gz"); part += 1; current = 0
            archive.add(path, arcname=path.relative_to(source)); current += size
    finally:
        if archive: archive.close()
    print(f"Created {part - 1} archive part(s) in {output}")


if __name__ == "__main__": main()
