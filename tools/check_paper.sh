#!/usr/bin/env bash
# Compile the paper (default: ../paper next to this script, or $1) with pdfLaTeX + BibTeX in a temp
# copy, copy main.pdf back, then check what the AAAI-27 DAI workshop requires:
#   main text (up to \label{maintext-end} after the conclusion) ends by page 7,
#   no unresolved ?? / [?] in the PDF, no undefined references or citations, no overfull boxes.
# aaai2027.sty refuses XeTeX, so tectonic cannot build this paper; pdflatex comes from TinyTeX
# (~/Library/TinyTeX, see paper/README.md).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER="${1:-$HERE/../paper}"
PY="${KCA_WORK:-$HOME/kca_work}/venv/bin/python"
for d in "$HOME"/Library/TinyTeX/bin/*; do [ -d "$d" ] && PATH="$d:$PATH"; done
command -v pdflatex >/dev/null || { echo "✗ pdflatex not found (install TinyTeX, see paper/README.md)"; exit 1; }
MAX_PAGE=7
B="$(mktemp -d)"
trap 'rm -rf "$B"' EXIT
rsync -a --exclude '*.log' --exclude '*.aux' --exclude '*.blg' --exclude '*.bbl' "$PAPER/" "$B/"
(
  cd "$B"
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >build.log 2>&1 &&
  bibtex main >>build.log 2>&1 &&
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >>build.log 2>&1 &&
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >main.final.log 2>&1
) || { tail -30 "$B/build.log" "$B/main.final.log" 2>/dev/null; echo "✗ compile failed"; exit 1; }
cp "$B/main.pdf" "$PAPER/main.pdf"
"$PY" - "$B" "$MAX_PAGE" <<'EOF'
import logging, re, sys
from pypdf import PdfReader
logging.disable(logging.WARNING)
b, max_page = sys.argv[1], int(sys.argv[2])
aux = open(f"{b}/main.aux").read()
m = re.search(r"\\newlabel\{maintext-end\}\{\{[^}]*\}\{(\d+)\}", aux)
end = int(m.group(1)) if m else 99
pages = PdfReader(f"{b}/main.pdf").pages
unresolved = sum(len(re.findall(r"\[\?\]|\?\?", p.extract_text() or "")) for p in pages)
log = open(f"{b}/main.final.log", errors="replace").read()
undef = len(re.findall(r"(Reference|Citation) `[^']+' on page \d+ undefined", log))
overfull = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)", log)
big = [float(x) for x in overfull if float(x) > 1.0]
checks = [
    (end <= max_page, f"{len(pages)} pages; main text ends on p{end} (limit p{max_page}; floats after it are not checked)"),
    (unresolved == 0, f"unresolved ?? / [?] in the PDF: {unresolved}"),
    (undef == 0, f"undefined references/citations: {undef}"),
    (not big, f"overfull boxes > 1pt: {len(big)}" + (f" (max {max(big):.1f}pt)" if big else "")),
]
for ok, msg in checks:
    print(("✓ " if ok else "✗ ") + msg)
sys.exit(0 if all(ok for ok, _ in checks) else 1)
EOF
