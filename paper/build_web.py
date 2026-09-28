"""Build the web edition of the paper into docs/paper/ (GitHub Pages).

Run after `tectonic -X compile main.tex` (it reads main.aux for figure, table
and section numbers, so the web edition numbers everything exactly as the PDF
does). Needs pandoc on PATH; uses no Python packages beyond the standard
library.

  docs/paper/index.html        the paper, with MathJax (SVG) for equations
  docs/paper/figures/          plots (PNG) and diagrams (SVG)
  docs/paper/neurhl-paper.pdf  the typeset PDF

Usage: python paper/build_web.py
"""
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "paper"
PDF_NAME = "neurhl-paper.pdf"
MATHJAX = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js"


def aux_labels():
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", (HERE / "main.aux").read_text()):
        labels[m.group(1)] = m.group(2)
    return labels


def inline_inputs(text):
    def rep(m):
        name = m.group(1)
        if name.startswith("figures/diagram_"):
            return "\\includegraphics{%s.svg}" % name
        return inline_inputs((HERE / f"{name}.tex").read_text())
    return re.sub(r"\\input\{([^}]+)\}", rep, text)


def preprocess(tex, labels):
    # figures: web images
    tex = re.sub(r"\\includegraphics(\[[^\]]*\])?\{(fig_[a-z0-9_]+)\}",
                 lambda m: "\\includegraphics{figures/%s.png}" % m.group(2), tex)
    # captions: prefix "Figure N." / "Table N." from the label in the same float
    def float_rep(m):
        env, body = m.group(1), m.group(2)
        lab = re.search(r"\\label\{([^}]+)\}", body)
        num = labels.get(lab.group(1), "?") if lab else "?"
        word = "Figure" if env == "figure" else "Table"
        body = body.replace("\\caption{", "\\caption{\\textbf{%s %s.} " % (word, num), 1)
        return "\\begin{%s}%s\\end{%s}" % (env, body, env)
    tex = re.sub(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", float_rep, tex, flags=re.S)
    # cross-references: numbers from the LaTeX build, linked to their anchors
    tex = re.sub(r"\\ref\{([^}]+)\}",
                 lambda m: "\\hyperref[%s]{%s}" % (m.group(1), labels.get(m.group(1), "?")), tex)
    # section numbers, then lettered appendices
    main, _, app = tex.partition("\\appendix")
    counters = {"s": 0, "ss": 0}

    def sec_rep(m, appendix=False):
        kind, title = m.group(1), m.group(2)
        if kind == "section":
            counters["s"] += 1
            counters["ss"] = 0
            n = chr(64 + counters["s"]) if appendix else str(counters["s"])
            return "\\section*{%s %s}" % (("Appendix " + n + ".") if appendix else n, title)
        counters["ss"] += 1
        n = (chr(64 + counters["s"]) if appendix else str(counters["s"])) + f".{counters['ss']}"
        return "\\subsection*{%s %s}" % (n, title)
    main = re.sub(r"\\(section|subsection)\{([^}]*)\}", sec_rep, main)
    counters["s"] = 0
    app = re.sub(r"\\(section|subsection)\{([^}]*)\}", lambda m: sec_rep(m, True), app)
    return main, app


def pandoc(tex):
    r = subprocess.run(
        ["pandoc", "-f", "latex", "-t", "html5", "--citeproc", "--shift-heading-level-by=1",
         "--bibliography", str(HERE / "refs.bib"), "--math-method=mathjax",
         "--wrap=none", "--metadata", "link-citations=true"],
        input=tex, capture_output=True, text=True, cwd=HERE)
    if r.returncode:
        raise SystemExit(r.stderr)
    if r.stderr.strip():
        print(r.stderr.strip()[:2000])
    return r.stdout


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>NeurHL Paper</title>
<meta name="description" content="NeurHL: a preregistered study of neural forecasting in the National Hockey League.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Alegreya:ital,wght@0,400;0,500;0,700;1,400&family=Barlow+Condensed:wght@500;600;700&family=Spline+Sans+Mono:wght@400;500&display=swap">
<style>
:root{
  color-scheme: light;
  --paper:#F7F8F6; --card:#FFFFFF; --ink:#16202B; --ink2:#3E4A57; --mut:#6B7682;
  --line:#D9DEE3; --rule:#C9D0D7; --blue:#1F4FB8; --tint:#EEF2F7;
  --serif:"Alegreya", Georgia, "Times New Roman", serif;
  --display:"Barlow Condensed", "Arial Narrow", sans-serif;
  --mono:"Spline Sans Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --paper:#0E141B; --card:#141C26; --ink:#E4EAF0; --ink2:#B3BFCB; --mut:#8793A0;
    --line:#26323F; --rule:#33414F; --blue:#7EA6FF; --tint:#18222E;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --paper:#0E141B; --card:#141C26; --ink:#E4EAF0; --ink2:#B3BFCB; --mut:#8793A0;
  --line:#26323F; --rule:#33414F; --blue:#7EA6FF; --tint:#18222E;
}
*{box-sizing:border-box}
html{scroll-padding-top:70px}
body{margin:0;background:var(--paper);color:var(--ink);font:19px/1.6 var(--serif);
  font-variant-numeric:lining-nums;-webkit-font-smoothing:antialiased}
a{color:var(--blue)}
.bar{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--line);
  padding-top:env(safe-area-inset-top,0px)}
.bar .in{max-width:1080px;margin:0 auto;padding-inline:20px;display:flex;align-items:center;
  gap:18px;min-height:54px;flex-wrap:wrap}
.mark{font:700 24px/1 var(--display);color:var(--ink);text-decoration:none}
.mark b{color:var(--blue)}
.bar nav{margin-left:auto;display:flex;gap:18px;flex-wrap:wrap}
.bar nav a{font:600 15px/1 var(--display);letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink2);text-decoration:none;padding-block:8px}
.bar nav a:hover{color:var(--ink)}
main{max-width:760px;margin:0 auto;padding-inline:20px;padding-block:40px 80px}
header.title h1{font:700 clamp(30px,5vw,44px)/1.08 var(--display);letter-spacing:.005em;
  margin:0 0 16px;text-wrap:balance}
header.title .by{font:500 15px/1.5 var(--mono);color:var(--mut)}
header.title .links{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
header.title .links a{font:600 14px/1 var(--display);letter-spacing:.07em;text-transform:uppercase;
  text-decoration:none;border:1px solid var(--rule);border-radius:4px;padding:8px 12px;color:var(--ink)}
header.title .links a:hover{border-color:var(--blue);color:var(--blue)}
.abstract{margin:32px 0 8px;padding:20px 22px;background:var(--card);border:1px solid var(--line);
  border-radius:6px}
.abstract h2{font:600 14px/1 var(--display);letter-spacing:.1em;text-transform:uppercase;
  color:var(--mut);margin:0 0 10px}
.abstract p{margin:0;font-size:18px}
nav.toc{margin:28px 0 0;font:500 15px/1.5 var(--serif)}
nav.toc h2{font:600 14px/1 var(--display);letter-spacing:.1em;text-transform:uppercase;color:var(--mut);margin:0 0 8px}
nav.toc ol{margin:0;padding:0;list-style:none;columns:2;column-gap:28px}
nav.toc li{break-inside:avoid;padding-block:1px}
nav.toc a{color:var(--ink2);text-decoration:none}
nav.toc a:hover{color:var(--blue)}
article h2{font:700 30px/1.15 var(--display);margin:56px 0 12px;letter-spacing:.005em;text-wrap:balance}
article h3{font:600 23px/1.2 var(--display);margin:34px 0 8px;letter-spacing:.01em}
article h4,article h5,article h6{font:700 19px/1.4 var(--serif);margin:22px 0 4px;display:inline}
article h4 + p,article h5 + p,article h6 + p{display:inline}
article h4::after,article h5::after,article h6::after{content:" "}
article p{margin:0 0 14px}
article dl dt{font-weight:700;margin-top:10px}
article dl dd{margin:0 0 6px 0}
figure{margin:28px 0;padding:0}
figure img{display:block;width:100%;height:auto;background:#fff;border-radius:4px;padding:6px}
figcaption{font-size:16px;line-height:1.5;color:var(--ink2);margin-top:8px}
.tbl{overflow-x:auto;margin:22px 0}
table{border-collapse:collapse;width:100%;font-size:16px;line-height:1.4}
caption{caption-side:top;text-align:left;font-size:16px;color:var(--ink2);padding-bottom:8px}
thead th{border-bottom:1.5px solid var(--ink2);text-align:left;font-weight:700;padding:6px 8px}
tbody td{padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child td{border-bottom:1.5px solid var(--ink2)}
td,th{font-variant-numeric:tabular-nums}
mjx-container{font-size:100% !important}
mjx-container[display="true"]{overflow-x:auto;overflow-y:hidden;margin:14px 0 !important}
#refs{font-size:16px;line-height:1.5}
#refs .csl-entry{padding-left:1.6em;text-indent:-1.6em;margin-bottom:6px}
footer{max-width:760px;margin:0 auto;padding:24px 20px 60px;border-top:1px solid var(--line);
  font:500 14px/1.5 var(--mono);color:var(--mut)}
</style>
<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]]},svg:{fontCache:"global"}};</script>
<script defer src="__MATHJAX__"></script>
</head>
<body>
<div class="bar"><div class="in">
  <a class="mark" href="../">Neur<b>HL</b></a>
  <nav><a href="../">Projections</a><a href="__PDF__">PDF</a><a href="https://github.com/ph05/neurhl">Code</a></nav>
</div></div>
<main>
<header class="title">
  <h1>NeurHL: A Preregistered Study of Neural Forecasting in the National Hockey League</h1>
  <div class="by">ph05 &middot; September 2026</div>
  <div class="links"><a href="__PDF__">Download PDF</a><a href="https://github.com/ph05/neurhl">Code and data</a><a href="../">Live projections</a></div>
</header>
<section class="abstract"><h2>Abstract</h2>__ABSTRACT__</section>
<nav class="toc"><h2>Contents</h2><ol>__TOC__</ol></nav>
<article>
__BODY__
</article>
</main>
<footer>NeurHL &middot; github.com/ph05/neurhl &middot; web edition built from the paper's LaTeX source</footer>
</body>
</html>
"""


def main():
    labels = aux_labels()
    src = (HERE / "main.tex").read_text()
    macros = "\n".join(l for l in src.splitlines() if l.startswith("\\newcommand"))
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1)
    body = src[src.index("\\input{sections/introduction}"):src.index("\\end{document}")]
    body = body.replace("\\bibliographystyle{plainnat}", "").replace("\\bibliography{refs}", "")
    body = body.replace("\\clearpage", "")
    body = inline_inputs(body)
    main_tex, app_tex = preprocess(body, labels)
    marker = "APPENDIXMARKER"
    html_body = pandoc(macros + "\n" + main_tex + "\n\n" + marker + "\n\n" + app_tex)
    html_abs = pandoc(macros + "\n" + abstract.replace("\\noindent", ""))
    # references before the appendices
    refs = re.search(r'<div id="refs".*?</div>\s*</div>\s*$', html_body, re.S)
    ref_html = ""
    if refs:
        ref_html = refs.group(0)
        html_body = html_body[:refs.start()]
    ref_block = ('<h2 id="references">References</h2>\n' + ref_html) if ref_html else ""
    html_body = html_body.replace("<p>" + marker + "</p>", ref_block)
    # tables scroll horizontally on narrow screens
    html_body = re.sub(r"(<table.*?</table>)", r'<div class="tbl">\1</div>', html_body, flags=re.S)
    # contents from the numbered section headings
    toc = []
    for m in re.finditer(r'<h2[^>]*\bid="([^"]+)"[^>]*>(.*?)</h2>', html_body):
        text = re.sub(r"<[^>]+>", "", m.group(2))
        toc.append(f'<li><a href="#{m.group(1)}">{text}</a></li>')
    page = (PAGE.replace("__MATHJAX__", MATHJAX).replace("__PDF__", PDF_NAME)
            .replace("__ABSTRACT__", html_abs).replace("__TOC__", "".join(toc))
            .replace("__BODY__", html_body))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for f in (HERE / "figures").glob("fig_*.png"):
        shutil.copy2(f, OUT / "figures" / f.name)
    for f in (HERE / "figures").glob("diagram_*.svg"):
        shutil.copy2(f, OUT / "figures" / f.name)
    shutil.copy2(HERE / "main.pdf", OUT / PDF_NAME)
    (OUT / "index.html").write_text(page)
    print(f"-> {OUT / 'index.html'} ({len(page) / 1024:.0f} KB), {len(toc)} sections")


if __name__ == "__main__":
    main()
