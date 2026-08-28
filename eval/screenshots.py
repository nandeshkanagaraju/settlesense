"""Regenerate the committed screenshots. `make screenshots`.

WHY THIS EXISTS. Six PNGs under reports/ui/ are linked from the README, and
until now there was no script, no target and no written procedure for making
them. A reviewer asked to check that a screenshot still matches what the code
renders could not: the only way to compare was to trust the image. A committed
image nobody can reproduce is a claim without a method, which is the same
objection this project makes to a figure with no artifact behind it.

WHAT IS AND IS NOT REPRODUCIBLE. The METHOD is: same page, same viewport, same
injected selection, every time. The BYTES are not, and are not claimed to be -
Chrome's version, the installed fonts and the display scale all move them, in
the same way `bench.md` records durations that differ per machine. So these are
evidence of what the current HTML renders, checked by regenerating and looking,
not byte-compared in a test.

THE FOUR STATIC SHOTS ARE DERIVED FROM COMMITTED HTML. reports/ui/queue.html
and queue-outage.html are byte-identical on every run of `make ui-static` and
`make ui-outage`, so what these capture is fixed by the pipeline rather than by
whatever happened to be on screen.

THE TWO STREAMLIT SHOTS ARE NOT AUTOMATED, and this says so rather than
pretending. `streamlit-queue.png` and `streamlit-outage.png` come from the
Streamlit app, which renders client-side; headless Chrome loads the page, the
server accepts the connection, and the capture comes out as the grey skeleton
placeholders. Tried and rejected: a 4s settle, a 25s settle, and disabling
XSRF and CORS so the websocket could not be the cause. Each produced a 29KB
image of nothing.

A 29KB image of a loading state would have REPLACED two real screenshots and
reported success, which is worse than having no target at all. So those two
stay manual, and the procedure is written down in the README instead of being
implied by a script that half works.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SHOTS", "PageHasNoTable", "Shot", "main"]

REPO = Path(__file__).resolve().parent.parent
# INTEGER NANOSECONDS, NOT SECONDS AS FLOAT. D6 bans float literals on any
# code path that decides something, with no module exemption - and "have we
# waited long enough" is a decision. `time.monotonic_ns` keeps the whole poll
# loop in ints, which is the charter's answer rather than a carve-out for it.
CAPTURE_TIMEOUT_NS = 120 * 1_000_000_000
STABLE_FOR_NS = 1_000_000_000
POLL_SLEEP_MS = 250

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)

# ---------------------------------------------------------------------------
# Cropping, done to the HTML rather than to the browser
# ---------------------------------------------------------------------------
#
# HEADLESS CHROME CAPTURES FROM THE TOP OF THE LAYOUT AND IGNORES SCROLL. Both
# obvious approaches were tried and produce the first screen every time:
# `--evaluate-on-new-document` runs before the DOM exists, so the query finds
# nothing; deferring the same script to the `load` event finds the row, scrolls
# to it, and the capture still comes out at the top of the page.
#
# So the crop is done to the DOCUMENT. The page has exactly one table with one
# thead and one tbody, and rows come in pairs - a data row followed by its
# evidence row - so selecting a window of rows and rebuilding the document is
# both deterministic and inspectable. No timing, no scroll, and the styles and
# header come along unchanged because only the tbody is replaced.


class PageHasNoTable(ValueError):
    """The file loaded but holds no queue table.

    DISTINCT FROM A MISSING FILE, and deliberately so. An absent
    `reports/ui/queue.html` means the build never ran; a present but empty one
    means it ran and produced nothing, which is a different fault with a
    different fix. Cropping the second as though it were a queue would write a
    screenshot of a blank page over a real one and report success - the same
    shape as the Streamlit skeleton this module refuses to take.
    """


_TBODY_OPEN = "<tbody>"
_TBODY_CLOSE = "</tbody>"
_ROW = re.compile(r"<tr>.*?</tr>", re.S)


def _split(html: str) -> tuple[str, list[str], str]:
    """(everything up to and including <tbody>, body rows, the rest)."""
    if _TBODY_OPEN not in html or _TBODY_CLOSE not in html:
        raise PageHasNoTable(
            "the page has no <tbody>, so there is no queue to crop. An empty or "
            "truncated render is legitimate output from a broken build and must "
            "not be captured as if it were the evidence queue."
        )
    start = html.index(_TBODY_OPEN) + len(_TBODY_OPEN)
    end = html.index(_TBODY_CLOSE)
    return html[:start], _ROW.findall(html[start:end]), html[end:]


def _rebuild(html: str, rows: Sequence[str]) -> str:
    head, _, tail = _split(html)
    return head + "".join(rows) + tail


def crop_ai_verified(html: str) -> str:
    """The AI_VERIFIED rows, with two pairs of context above them.

    Selected by CONTENT, not by index: these sit at ranks 58-59 today and the
    whole reason the page stopped truncating at 40 was that they moved. A row
    offset would silently capture the wrong rows the next time one is added.
    """
    head, rows, tail = _split(html)
    hits = [index for index, row in enumerate(rows) if "AI_VERIFIED" in row]
    if not hits:
        return html
    first, last = min(hits), max(hits)
    window = rows[max(0, first - 4) : last + 2]
    return head + "".join(window) + tail


def crop_evidence_panel(html: str) -> str:
    """One abstained duplicate with its evidence panel already expanded.

    The panel is a <details>, which renders collapsed by default, so the shot
    the README wants does not exist until it is opened. Opening it here rather
    than clicking it in a browser is what makes the image reproducible.
    """
    head, rows, tail = _split(html)
    for index, row in enumerate(rows[:-1]):
        if "ABSTAINED" not in row or "DUPLICATE_CANDIDATE" not in row:
            continue
        panel = rows[index + 1]
        if "<details>" not in panel:
            continue
        return head + row + panel.replace("<details>", "<details open>", 1) + tail
    return html


@dataclass(frozen=True)
class Shot:
    """One screenshot: a page, a viewport, and how to reduce the document."""

    name: str
    source: str
    width: int
    height: int
    crop: Callable[[str], str] | None = None


SHOTS: tuple[Shot, ...] = (
    Shot("evidence-queue", "reports/ui/queue.html", 1500, 1500),
    Shot("queue-outage", "reports/ui/queue-outage.html", 1500, 1500),
    Shot("queue-ai-verified", "reports/ui/queue.html", 1500, 1000, crop_ai_verified),
    Shot("evidence-panel", "reports/ui/queue.html", 1400, 1150, crop_evidence_panel),
)


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return shutil.which("google-chrome") or shutil.which("chromium")


def capture(chrome: str, url: str, out: Path, shot: Shot, scale: int) -> bool:
    """One headless capture. Returns whether a PNG was written.

    CHROME IS STOPPED DELIBERATELY RATHER THAN WAITED ON. On this platform
    `--screenshot` writes the file correctly and then the browser does not
    exit - every headless variant, with and without a virtual time budget,
    sits there until killed. So the file is the completion signal: poll until
    it appears and its size stops changing, then terminate. Waiting for the
    process instead would hang the target, and a screenshot tool that leaves a
    browser running is a worse problem than a stale image.
    """
    if out.exists():
        out.unlink()
    with tempfile.TemporaryDirectory() as profile:
        command = [
            chrome,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            f"--force-device-scale-factor={scale}",
            f"--window-size={shot.width},{shot.height}",
            f"--screenshot={out}",
        ]
        command.append(url)
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic_ns() + CAPTURE_TIMEOUT_NS
            stable_since: int | None = None
            last_size = -1
            while time.monotonic_ns() < deadline:
                if process.poll() is not None and out.exists():
                    break
                size = out.stat().st_size if out.exists() else 0
                if size and size == last_size:
                    if stable_since is None:
                        stable_since = time.monotonic_ns()
                    elif time.monotonic_ns() - stable_since >= STABLE_FOR_NS:
                        break
                else:
                    stable_since = None
                last_size = size
                time.sleep(POLL_SLEEP_MS / 1000)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                    process.kill()
                    process.wait(timeout=10)
    return out.exists() and out.stat().st_size > 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "ui")
    parser.add_argument("--only", action="append", default=None, help="shot name; repeatable")
    parser.add_argument("--scale", type=int, default=2, help="device scale factor")
    args = parser.parse_args(argv)

    chrome = find_chrome()
    if chrome is None:
        print(
            "No Chrome or Chromium found. Looked in:\n  "
            + "\n  ".join(CHROME_CANDIDATES)
            + "\nInstall one, or pass a path by editing CHROME_CANDIDATES.",
            file=sys.stderr,
        )
        return 3
    print(f"browser: {chrome}")

    wanted = [shot for shot in SHOTS if args.only is None or shot.name in args.only]
    if not wanted:
        print("nothing selected", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for shot in wanted:
        out = args.out / f"{shot.name}.png"
        before = out.stat().st_size if out.exists() else 0
        source = REPO / shot.source
        if not source.exists():
            print(f"  {shot.name}: {shot.source} is missing; run `make ui-static` first")
            continue
        page = source
        if shot.crop is not None:
            cropped = Path(tempfile.mkdtemp()) / source.name
            cropped.write_text(shot.crop(source.read_text("utf-8")), "utf-8")
            page = cropped
        ok = capture(chrome, page.as_uri(), out, shot, args.scale)
        if not ok:
            print(f"  {shot.name}: FAILED")
            continue
        size = out.stat().st_size
        change = "new" if before == 0 else f"was {before:,}"
        print(f"  {shot.name:<20} {size:>9,} bytes  ({change})")
        written += 1

    print(f"\nwrote {written} of {len(wanted)} screenshot(s) to {args.out}")
    print(
        "The METHOD is reproducible; the BYTES are not - Chrome version, fonts and\n"
        "display scale all move them, like the durations in bench.md. Compare by\n"
        "looking, not by hashing."
    )
    return 0 if written == len(wanted) else 1


if __name__ == "__main__":
    sys.exit(main())
