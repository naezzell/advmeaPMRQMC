"""Validate and assemble the benchmark evidence notebook.

The combined Markdown is deterministic and useful even without Pandoc.  The
optional LaTeX conversion deliberately creates only ignored build artifacts.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def assemble():
    manifest = json.loads((ROOT / "report_manifest.json").read_text())
    missing = [name for name in manifest["sections"] if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("missing report sections: " + ", ".join(missing))
    parts = [f"# {manifest['title']}\n"]
    for name in manifest["sections"]:
        text = (ROOT / name).read_text().strip()
        parts.append(text)
    return "\n\n\\newpage\n\n".join(parts) + "\n", manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex", action="store_true", help="also invoke Pandoc")
    args = parser.parse_args()
    text, manifest = assemble()
    build = ROOT / "report" / "build"
    build.mkdir(parents=True, exist_ok=True)
    combined = build / "combined.md"
    combined.write_text(text)
    print(f"wrote {combined}")
    if args.latex:
        pandoc = shutil.which("pandoc")
        if not pandoc:
            raise SystemExit("pandoc is not installed; combined Markdown was still written")
        output = build / "benchmark_report.tex"
        subprocess.run([
            pandoc, str(combined), "--standalone", "--citeproc",
            "--bibliography", str(ROOT / manifest["bibliography"]),
            "--output", str(output),
        ], check=True)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
