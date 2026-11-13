"""Markdown -> PDF via Edge headless (Chinese fonts from the OS)."""
import subprocess
import sys
from pathlib import Path

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CSS = """
body { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; margin: 36px 42px;
       font-size: 13px; line-height: 1.65; color: #1a1a1a; }
h1 { font-size: 21px; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 16px; margin-top: 22px; border-bottom: 1px solid #bbb; padding-bottom: 4px; }
h3 { font-size: 14px; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 11.5px; }
th, td { border: 1px solid #999; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #eee; }
code { font-family: Consolas, monospace; background: #f2f2f2; padding: 0 3px; font-size: 11.5px; }
pre { background: #f6f6f6; padding: 10px; overflow-x: auto; font-size: 11px; }
blockquote { border-left: 3px solid #888; margin-left: 0; padding-left: 12px; color: #444; }
"""


def convert(md_path: Path, pdf_path: Path) -> None:
    import markdown

    html_body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, timeout=300, capture_output=True)
    html_path.unlink(missing_ok=True)
    print(f"PDF {pdf_path} {pdf_path.stat().st_size // 1024} KB", flush=True)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        src = Path(arg).resolve()
        convert(src, src.with_suffix(".pdf"))
