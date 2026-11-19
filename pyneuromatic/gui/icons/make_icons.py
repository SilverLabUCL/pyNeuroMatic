"""Draw pyNeuroMatic's icon and wordmark (#545).

The mark is the NeuroMatic website logo's "M": two events on a baseline, in
its red. Run from the repository to remake every file in this folder:

    .venv/bin/python pyneuromatic/gui/icons/make_icons.py

Needs PyQt6 (to render PNGs and outline the wordmark's text) and Pillow (for
the .ico); the macOS .icns is made with macOS's iconutil, when there.

Files:
    icon.svg              the icon, filling its square (favicon, window, .ico)
    icon-macos.svg        on macOS's icon grid, with its margin (Dock, .icns)
    icon-<size>.png       icon.svg at 16 ... 1024 px
    icon-macos-<size>.png icon-macos.svg at 16 ... 1024 px
    pyneuromatic.icns     macOS app icon
    pyneuromatic.ico      Windows icon and website favicon (16, 32, 48, 64 px)
    wordmark.svg          "pyNeuroMatic" for light backgrounds, text as outlines
    wordmark-dark.svg     the same for dark backgrounds
"""
from __future__ import annotations

import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

RED = "#CB0000"  # the NeuroMatic website logo's red
RED_ON_DARK = "#FF4A4A"
SQUARE = "#FFFFFF"
EDGE = "#DADADA"  # the white square's edge, so it shows on white
GREY, GREY_ON_DARK = "#8C8C8C", "#9A9A9A"  # "py"
TEXT, TEXT_ON_DARK = "#111111", "#F2F2F2"
PNG_SIZES = (16, 32, 48, 64, 128, 256, 512, 1024)


# --- the trace -------------------------------------------------------------

def _events(x: float, peaks, rise: float, tau: float) -> float:
    """Events summed: each a linear rise over *rise*, then a decay *tau*."""
    y = 0.0
    for x0, height in peaks:
        dt = x - x0
        if -rise <= dt < 0:
            y += height * (dt + rise) / rise
        elif dt >= 0:
            y += height * math.exp(-dt / tau)
    return y


def trace_path(x0, x1, step, base, peaks, rise, tau, noise=0.0, seed=7) -> str:
    """SVG path data of a baseline at *base* with *peaks* [(x, height)],
    and a little correlated noise, like a recording."""
    rnd = random.Random(seed)
    points, x, n = [], x0, 0.0
    while x <= x1 + 1e-9:
        if noise:
            n = 0.55 * n + rnd.gauss(0, noise)
        points.append((x, base - _events(x, peaks, rise, tau) + n))
        x += step
    return "M" + " L".join(f"{px:.2f},{py:.2f}" for px, py in points)


# The icon's mark (in a 100 x 100 square): smooth and bold, the two events
# far enough apart to stay two at 16 px
MARK = trace_path(12, 88, 0.25, 74, [(40, 52), (57, 42)], 1.2, 4.0)
MARK_WIDTH = 6.5


def icon_svg(margin: float = 0.0, size: int = 512) -> str:
    """The icon: the mark on a white rounded square. *margin*: the space
    around the square, as a fraction of the icon (macOS's grid: 0.1)."""
    side = 100 * (1 - 2 * margin)
    offset = 100 * margin
    scale = side / 100
    radius = 22 * scale
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 100 100">'
        f'<rect x="{offset + 0.5 * scale:.3f}" y="{offset + 0.5 * scale:.3f}" '
        f'width="{side - scale:.3f}" height="{side - scale:.3f}" rx="{radius:.3f}" '
        f'fill="{SQUARE}" stroke="{EDGE}" stroke-width="{scale:.3f}"/>'
        f'<g transform="translate({offset:.3f} {offset:.3f}) scale({scale:.4f})">'
        f'<path d="{MARK}" fill="none" stroke="{RED}" stroke-width="{MARK_WIDTH}" '
        f'stroke-linejoin="round" stroke-linecap="round"/></g></svg>\n'
    )


# --- the wordmark ----------------------------------------------------------

FONT_SIZE = 40


def _text_path(text: str, x: float, baseline: float) -> str:
    """*text* in Arial as SVG path data (outlines: the same on every
    computer, with or without the font)."""
    from PyQt6 import QtGui

    font = QtGui.QFont("Arial")
    font.setPixelSize(FONT_SIZE)
    path = QtGui.QPainterPath()
    path.addText(x, baseline, font, text)
    parts, i, count = [], 0, path.elementCount()
    while i < count:
        e = path.elementAt(i)
        if e.isMoveTo():
            parts.append(f"M{e.x:.2f},{e.y:.2f}")
            i += 1
        elif e.isLineTo():
            parts.append(f"L{e.x:.2f},{e.y:.2f}")
            i += 1
        else:  # a cubic: this point and the two after it
            c1, c2 = path.elementAt(i), path.elementAt(i + 1)
            end = path.elementAt(i + 2)
            parts.append(f"C{c1.x:.2f},{c1.y:.2f} {c2.x:.2f},{c2.y:.2f} "
                         f"{end.x:.2f},{end.y:.2f}")
            i += 3
    return "".join(parts) + "Z" if parts else ""


def _advance(text: str) -> float:
    from PyQt6 import QtGui

    font = QtGui.QFont("Arial")
    font.setPixelSize(FONT_SIZE)
    return QtGui.QFontMetricsF(font).horizontalAdvance(text)


def wordmark_svg(dark: bool = False) -> str:
    """ "pyNeuroMatic": "py" in grey, the two events making the "M" and
    running on as the baseline beneath, as on the NeuroMatic website."""
    py_fill = GREY_ON_DARK if dark else GREY
    text_fill = TEXT_ON_DARK if dark else TEXT
    red = RED_ON_DARK if dark else RED
    baseline_text, baseline_trace = 52, 62
    x_py = 4.0
    x_neuro = x_py + _advance("py")
    x_m = x_neuro + _advance("Neuro")
    peaks = [(x_m + 3.5, 56), (x_m + 10.5, 45)]  # the M
    x_atic = x_m + 17
    right = x_atic + _advance("atic") + 6
    trace = trace_path(0, right, 0.5, baseline_trace, peaks, 0.6, 1.6, 0.45, seed=11)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {right:.0f} 70" '
        f'width="{right * 4:.0f}" height="280" role="img" aria-label="pyNeuroMatic">'
        f'<path d="{_text_path("py", x_py, baseline_text)}" fill="{py_fill}"/>'
        f'<path d="{_text_path("Neuro", x_neuro, baseline_text)}" fill="{text_fill}"/>'
        f'<path d="{_text_path("atic", x_atic, baseline_text)}" fill="{text_fill}"/>'
        f'<path d="{trace}" fill="none" stroke="{red}" stroke-width="1.1" '
        f'stroke-linejoin="round"/></svg>\n'
    )


# --- files -----------------------------------------------------------------

def _render_png(svg_path: Path, size: int, out: Path) -> None:
    from PyQt6 import QtCore, QtGui, QtSvg

    image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    QtSvg.QSvgRenderer(str(svg_path)).render(painter, QtCore.QRectF(0, 0, size, size))
    painter.end()
    image.save(str(out))


def _make_icns(out: Path) -> bool:
    """macOS's .icns from the icon-macos PNGs, with iconutil (macOS only)."""
    if shutil.which("iconutil") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "pyneuromatic.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            shutil.copy(HERE / f"icon-macos-{size}.png", iconset / f"icon_{size}x{size}.png")
            shutil.copy(HERE / f"icon-macos-{size * 2}.png",
                        iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)
    return True


def _make_ico(out: Path) -> None:
    from PIL import Image

    sizes = [(s, s) for s in (16, 32, 48, 64)]
    Image.open(HERE / "icon-64.png").save(out, sizes=sizes,
                                          append_images=[Image.open(HERE / f"icon-{s}.png")
                                                         for s, _ in sizes])


def main() -> int:
    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])  # noqa: F841
    (HERE / "icon.svg").write_text(icon_svg())
    (HERE / "icon-macos.svg").write_text(icon_svg(margin=0.1))
    for size in PNG_SIZES:
        _render_png(HERE / "icon.svg", size, HERE / f"icon-{size}.png")
        _render_png(HERE / "icon-macos.svg", size, HERE / f"icon-macos-{size}.png")
    _make_ico(HERE / "pyneuromatic.ico")
    made_icns = _make_icns(HERE / "pyneuromatic.icns")
    (HERE / "wordmark.svg").write_text(wordmark_svg())
    (HERE / "wordmark-dark.svg").write_text(wordmark_svg(dark=True))
    print("made the icons and wordmark in", HERE)
    if not made_icns:
        print("no iconutil (macOS only): pyneuromatic.icns not remade")
    return 0


if __name__ == "__main__":
    sys.exit(main())
