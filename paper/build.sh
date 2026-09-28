#!/bin/bash
# Build the paper: PDF with tectonic, then the web edition in docs/paper/.
# usage: paper/build.sh [--figures]   (--figures also regenerates every figure)
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
if [ "${1:-}" = "--figures" ]; then
  (cd .. && uv run -q --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow \
     --with matplotlib --with scipy python paper/figures/make_figures.py)
fi
for d in diagram_overview diagram_garch; do
  (cd figures && tectonic -X compile "standalone_$d.tex" >/dev/null)
done
(cd figures && uv run -q --no-project --python 3.12 --with pymupdf python -c "
import pymupdf
for n in ('diagram_overview', 'diagram_garch'):
    p = pymupdf.open(f'standalone_{n}.pdf')[0]
    open(f'{n}.svg', 'w').write(p.get_svg_image(text_as_path=True))
")
tectonic -X compile --keep-intermediates --keep-logs main.tex
python3 build_web.py
