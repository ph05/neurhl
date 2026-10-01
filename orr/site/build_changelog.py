"""Render orr/CHANGELOG.md as docs/orr/changelog.html (GitHub Pages), in the
projection page's style.

    python3 -m orr.site.build_changelog
"""
from __future__ import annotations

import html
import re

from orr import config as C

SRC = C.PKG / "CHANGELOG.md"
OUT = C.ROOT / "docs" / "orr" / "changelog.html"


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def render(md: str) -> str:
    out, in_list = [], False
    for line in md.splitlines():
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ORR Changelog</title>
<style>
  :root { --fg:#111; --muted:#555; --rule:#ccc; --link:#0645ad; --bg:#fff; --code:#f2f2f2; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --fg:#e6e6e6; --muted:#a3a3a3; --rule:#444; --link:#8ab4f8; --bg:#121212; --code:#222; color-scheme: dark; } }
  :root[data-theme="dark"] { --fg:#e6e6e6; --muted:#a3a3a3; --rule:#444; --link:#8ab4f8; --bg:#121212; --code:#222; color-scheme: dark; }
  html { background: var(--bg); }
  body { margin: 0 auto; max-width: 860px; padding: 12px 16px 40px; color: var(--fg); background: var(--bg);
         font: 13px/1.55 Verdana, Geneva, "DejaVu Sans", Arial, sans-serif; }
  a { color: var(--link); } h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 26px; padding-bottom: 3px; border-bottom: 2px solid var(--fg); }
  p, li { max-width: 88ch; }
  table { border-collapse: collapse; } td, th { padding: 2px 10px 2px 0; text-align: left; border-bottom: 1px solid var(--rule); } code { background: var(--code); padding: 0 3px; font-size: 12px; }
  nav { font-size: 12px; margin: 6px 0; padding: 5px 0; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }
  nav a { margin-right: 14px; }
</style>
</head>
<body>
<nav><a href="index.html">Projections</a><a href="compare.html">ORR vs NeurHL</a><a href="changelog.html">Changelog</a></nav>
__BODY__
</body>
</html>
"""


def releases_table() -> str:
    """ORR 1.7: version history, one row per release note in orr/releases/."""
    import re as _re
    rows = []
    for f in sorted((C.PKG / "releases").glob("v*.md"), key=lambda p: [int(x) for x in p.stem[1:].split(".")], reverse=True):
        first = f.read_text().splitlines()[0].lstrip("# ").strip()
        title = first.split(":", 1)[1].strip() if ":" in first else first
        url = f"https://github.com/ph05/neurhl/blob/claude/gallant-maxwell-rbw7u6/orr/releases/{f.name}"
        rows.append(f"<tr><td><a href=\"{url}\">{html.escape(f.stem[1:])}</a></td><td>{html.escape(title)}</td></tr>")
    return ("<h2>Version history</h2><p>Release notes with the evidence for every change. The 2026-27 preseason file is ORR 1.0 "
            "and stays frozen; later versions change the daily in-season forecasts.</p><table><thead><tr><th>Version</th>"
            "<th>Release</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__BODY__", releases_table() + render(SRC.read_text())))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
