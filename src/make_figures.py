"""Render the two report figures (PIL only - matplotlib is not installed here).

  figures/fig1_two_stage_pipeline.png   - two-stage abstainer flowchart (report Figure 1)
  figures/fig2_coverage_quality_cost.png - coverage vs selective / behaviour accuracy
  figures/plot_data.csv                 - the exact points plotted (replot in Colab for PDF)

Usage (project root):  python src/make_figures.py
"""
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
CURVES = json.loads((ROOT / "results" / "coverage_accuracy_curves.json")
                    .read_text(encoding="utf-8"))

S = 2  # supersampling factor
INK = (33, 37, 41)
GRID = (224, 228, 232)
MUTED = (110, 117, 125)
C_M1 = (31, 119, 180)
C_M3 = (214, 96, 45)
C_TS = (35, 139, 69)
C_M4 = (142, 68, 173)

REG = Path(r"C:\Windows\Fonts\arial.ttf")
BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


def font(size, bold=False):
    path = BOLD if bold and BOLD.exists() else REG
    try:
        return ImageFont.truetype(str(path), int(size * S))
    except OSError:
        return ImageFont.load_default()


class Canvas:
    def __init__(self, w, h):
        self.img = Image.new("RGB", (w * S, h * S), (255, 255, 255))
        self.d = ImageDraw.Draw(self.img)

    def line(self, pts, fill=INK, width=2, dash=None):
        p = [(x * S, y * S) for x, y in pts]
        if len(p) < 2:
            return
        if not dash:
            self.d.line(p, fill=fill, width=max(1, int(width * S)), joint="curve")
            return
        for i in range(len(p) - 1):
            (x1, y1), (x2, y2) = p[i], p[i + 1]
            seg = math.hypot(x2 - x1, y2 - y1)
            if seg == 0:
                continue
            n = max(1, int(seg // (dash * S)))
            for k in range(n):
                a, b = k / n, min(1.0, (k + 0.6) / n)
                self.d.line([(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a),
                             (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)],
                            fill=fill, width=max(1, int(width * S)))

    def rect(self, box, fill=None, outline=INK, width=2, radius=10):
        x0, y0, x1, y1 = [v * S for v in box]
        self.d.rounded_rectangle([x0, y0, x1, y1], radius=radius * S, fill=fill,
                                 outline=outline, width=max(1, int(width * S)))

    def circle(self, xy, r, fill, outline=None, width=1.5):
        x, y = xy[0] * S, xy[1] * S
        self.d.ellipse([x - r * S, y - r * S, x + r * S, y + r * S], fill=fill,
                       outline=outline, width=max(1, int(width * S)))

    def diamond(self, xy, r, fill):
        x, y = xy[0] * S, xy[1] * S
        self.d.polygon([(x, y - r * S), (x + r * S, y), (x, y + r * S), (x - r * S, y)], fill=fill)

    def text(self, xy, s, size=18, fill=INK, anchor="la", bold=False, bg=None):
        f = font(size, bold)
        if bg:
            box = self.d.textbbox((xy[0] * S, xy[1] * S), s, font=f, anchor=anchor)
            pad = 3 * S
            self.d.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], fill=bg)
        self.d.text((xy[0] * S, xy[1] * S), s, font=f, fill=fill, anchor=anchor)

    def vtext(self, xy, s, size=17, fill=INK):
        f = font(size)
        tmp = Image.new("RGBA", (420 * S, 70 * S), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((210 * S, 35 * S), s, font=f, fill=fill, anchor="mm")
        tmp = tmp.rotate(90, expand=True)
        bbox = tmp.getbbox()
        if bbox:
            tmp = tmp.crop(bbox)
        self.img.paste(tmp, (int(xy[0] * S - tmp.width / 2), int(xy[1] * S - tmp.height / 2)), tmp)

    def arrow(self, pts, fill=INK, width=2, head=9):
        self.line(pts, fill=fill, width=width)
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        ang = math.atan2(y2 - y1, x2 - x1)
        a = (x2 - head * math.cos(ang - 0.42), y2 - head * math.sin(ang - 0.42))
        b = (x2 - head * math.cos(ang + 0.42), y2 - head * math.sin(ang + 0.42))
        self.d.polygon([(x2 * S, y2 * S), (a[0] * S, a[1] * S), (b[0] * S, b[1] * S)], fill=fill)

    def save(self, path):
        w, h = self.img.size
        self.img.resize((w // S, h // S), Image.LANCZOS).save(path, dpi=(300, 300))


# ---------------------------------------------------------------------------
# Figure 1: two-stage pipeline
# ---------------------------------------------------------------------------
def fig1():
    W, H = 1500, 780
    c = Canvas(W, H)
    blue_fill, blue_edge = (232, 240, 254), (74, 111, 165)
    green_fill, green_edge = (230, 244, 234), (52, 168, 83)
    red_fill, red_edge = (252, 232, 230), (197, 34, 31)

    def box(x, y, w, h, lines, fill, edge):
        c.rect((x, y, x + w, y + h), fill=fill, outline=edge, width=2, radius=12)
        ty = y + h / 2 - (len(lines) - 1) * 13
        for i, line in enumerate(lines):
            c.text((x + w / 2, ty + i * 26), line, size=19 if i == 0 else 17,
                   fill=INK, anchor="mm", bold=(i == 0))

    box(620, 40, 260, 70, ["Question + context"], blue_fill, blue_edge)
    box(590, 150, 320, 110, ["Stage 1: one greedy pass", "s1 = length-normalised",
                             "log-prob of the answer"], blue_fill, blue_edge)
    box(590, 420, 320, 110, ["Stage 2: N = 5 samples", "c = largest-cluster share",
                             "(consistency)"], blue_fill, blue_edge)
    box(170, 168, 220, 74, ["Abstain"], red_fill, red_edge)
    box(1160, 168, 220, 74, ["Answer"], green_fill, green_edge)
    box(1160, 438, 220, 74, ["Answer"], green_fill, green_edge)
    box(1160, 610, 220, 74, ["Abstain"], red_fill, red_edge)

    c.arrow([(750, 110), (750, 150)])
    c.arrow([(910, 205), (1160, 205)])
    c.text((1035, 184), "s1 ≥ τH", size=17, anchor="mm", fill=MUTED)
    c.arrow([(590, 205), (390, 205)])
    c.text((490, 184), "s1 ≤ τL", size=17, anchor="mm", fill=MUTED)
    c.arrow([(750, 260), (750, 420)])
    c.text((764, 340), "otherwise (borderline)", size=17, anchor="lm", fill=MUTED)
    c.arrow([(910, 475), (1160, 475)])
    c.text((1035, 454), "c ≥ τC", size=17, anchor="mm", fill=MUTED)
    c.arrow([(750, 530), (750, 647), (1160, 647)])
    c.text((1035, 626), "c < τC", size=17, anchor="mm", fill=MUTED)

    c.text((30, 740), "Stage 1 runs once per question (1 sequence); Stage 2 runs only on "
                      "borderline questions, so the average sample count is 1 + 5·p.",
           size=16, fill=MUTED)
    c.save(OUT / "fig1_two_stage_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 2: coverage vs accuracy
# ---------------------------------------------------------------------------
def step(points):
    pts = sorted(points, key=lambda p: p[0])
    out, prev = [], None
    for cov, val in pts:
        if prev is None:
            out.append((cov, val))
        else:
            out.append((cov, prev))
            out.append((cov, val))
        prev = val
    return out


class Panel:
    def __init__(self, c, box, xlim, ylim, title, ylabel, xticks, yticks):
        self.c, self.x0, self.y0, self.x1, self.y1 = c, *box
        self.xlim, self.ylim = xlim, ylim
        self.title, self.ylabel = title, ylabel
        self.xticks, self.yticks = xticks, yticks

    def X(self, v):
        return self.x0 + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.x1 - self.x0)

    def Y(self, v):
        return self.y1 - (v - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.y1 - self.y0)

    def frame(self):
        c = self.c
        for t in self.yticks:
            c.line([(self.x0, self.Y(t)), (self.x1, self.Y(t))], fill=GRID, width=1)
            c.text((self.x0 - 10, self.Y(t)), f"{t:.1f}", size=15, anchor="rm", fill=MUTED)
        for t in self.xticks:
            c.text((self.X(t), self.y1 + 9), f"{t:.1f}", size=15, anchor="ma", fill=MUTED)
        c.line([(self.x0, self.y1), (self.x1, self.y1)], fill=INK, width=1.5)
        c.line([(self.x0, self.y0), (self.x0, self.y1)], fill=INK, width=1.5)
        c.text(((self.x0 + self.x1) / 2, self.y1 + 40), "coverage", size=17, anchor="mm")
        c.vtext((self.x0 - 62, (self.y0 + self.y1) / 2), self.ylabel, size=17)
        c.text((self.x0, self.y0 - 20), self.title, size=19, anchor="lm", bold=True)

    def series(self, pts, color):
        if len(pts) >= 2:
            self.c.line([(self.X(x), self.Y(y)) for x, y in pts], fill=color, width=2.5)

    def marker(self, x, y, color, label=None, dx=0, dy=-16, yerr=None):
        if yerr:
            lo, hi = yerr
            self.c.line([(self.X(x), self.Y(lo)), (self.X(x), self.Y(hi))], fill=color, width=2)
            for yy in (lo, hi):
                self.c.line([(self.X(x) - 6, self.Y(yy)), (self.X(x) + 6, self.Y(yy))],
                            fill=color, width=2)
        self.c.circle((self.X(x), self.Y(y)), 6, fill=color)
        if label:
            self.c.text((self.X(x) + dx, self.Y(y) + dy), label, size=14, anchor="mm",
                        fill=color, bg=(255, 255, 255))

    def hline(self, y, color, label):
        self.c.line([(self.x0, self.Y(y)), (self.x1, self.Y(y))], fill=color, width=1.4, dash=7)
        self.c.text((self.x1 - 4, self.Y(y) - 8), label, size=14, anchor="rb", fill=color)

    def diamond_marker(self, x, y, color, label=None, dx=0, dy=-18):
        self.c.diamond((self.X(x), self.Y(y)), 9, color)
        if label:
            self.c.text((self.X(x) + dx, self.Y(y) + dy), label, size=14, anchor="mm",
                        fill=color, bg=(255, 255, 255))


def clean(pts, lo=0.03, hi=1.0):
    return [(x, y) for x, y in pts
            if lo <= x <= hi and y == y]


def fig2():
    W, H = 1600, 700
    c = Canvas(W, H)

    m1 = clean([(p["coverage"], p["selective_accuracy"]) for p in CURVES["M1"]])
    m3 = clean([(p["coverage"], p["selective_accuracy"]) for p in CURVES["M3_k5"]])
    m1b = clean([(p["coverage"], p["behaviour_accuracy"]) for p in CURVES["M1"]])
    m3b = clean([(p["coverage"], p["behaviour_accuracy"]) for p in CURVES["M3_k5"]])
    m1 = m1[::8] + [m1[-1]]
    m1b = m1b[::8] + [m1b[-1]]
    ts = [(0.298, 0.517, 0.538), (0.480, 0.463, 0.514)]
    m4 = None
    m24 = ROOT / "results" / "m2_m4_table.csv"
    if m24.exists():
        with m24.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row["method"] == "M4":
                    m4 = (float(row["test_cov"]), float(row["selective_acc"]),
                          float(row["behaviour_acc"]))
    CI = {}
    ci_path = ROOT / "results" / "ci_table.csv"
    if ci_path.exists():
        with ci_path.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row["method"] == "two-stage":
                    CI[row["target_coverage"]] = (
                        float(row["sel_ci_low"]), float(row["sel_ci_high"]),
                        float(row["beh_ci_low"]), float(row["beh_ci_high"]))

    pa = Panel(c, (140, 100, 770, 560), (0.0, 1.0), (0.2, 0.9),
               "(a) Selective accuracy vs. coverage", "selective accuracy",
               [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    pa.frame()
    pa.series(step(m1), C_M1)
    pa.series(step(m3), C_M3)
    pa.marker(ts[0][0], ts[0][1], C_TS, f"{ts[0][1]*100:.1f}", dx=-30, dy=-16,
              yerr=CI.get("0.3", (None, None))[:2] if "0.3" in CI else None)
    pa.marker(ts[1][0], ts[1][1], C_TS, f"{ts[1][1]*100:.1f}", dx=28, dy=2,
              yerr=CI.get("0.5", (None, None))[:2] if "0.5" in CI else None)
    if m4:
        pa.diamond_marker(m4[0], m4[1], C_M4, f"{m4[1]*100:.1f}", dx=26, dy=-12)

    pb = Panel(c, (980, 100, 1560, 560), (0.0, 1.0), (0.2, 0.7),
               "(b) Overall decision accuracy vs. coverage", "behaviour accuracy",
               [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    pb.frame()
    pb.series(step(m1b), C_M1)
    pb.series(step(m3b), C_M3)
    pb.marker(ts[0][0], ts[0][2], C_TS, f"{ts[0][2]*100:.1f}", dx=0, dy=-20,
              yerr=(CI["0.3"][2], CI["0.3"][3]) if "0.3" in CI else None)
    pb.marker(ts[1][0], ts[1][2], C_TS, f"{ts[1][2]*100:.1f}", dx=0, dy=-20,
              yerr=(CI["0.5"][2], CI["0.5"][3]) if "0.5" in CI else None)
    pb.hline(0.5, MUTED, "always abstain = 0.500")
    if m4:
        pb.diamond_marker(m4[0], m4[2], C_M4, f"{m4[2]*100:.1f}", dx=26, dy=-12)

    lx, ly = 140, 645
    for i, (color, label) in enumerate([(C_M1, "M1: log-prob"),
                                        (C_M3, "M3: 5-sample consistency"),
                                        (C_TS, "two-stage (0.3 / 0.5)"),
                                        (C_M4, "M4: LoRA tuning")]):
        x = lx + i * 360
        c.line([(x, ly), (x + 30, ly)], fill=color, width=3)
        c.text((x + 40, ly), label, size=15, anchor="lm")
    c.text((30, 24), "Test set: 1,000 questions (500 answerable / 500 unanswerable) · "
                     "Qwen2.5-0.5B-Instruct · thresholds tuned on val · "
                     "error bars: 95% paired bootstrap CI",
           size=15, fill=MUTED)
    c.save(OUT / "fig2_coverage_quality_cost.png")


def write_data():
    rows = []
    for key in ("M1", "M3_k5"):
        for p in CURVES[key]:
            rows.append({"series": key, "coverage": round(p["coverage"], 6),
                         "selective_accuracy": round(p["selective_accuracy"], 6)
                         if p["selective_accuracy"] == p["selective_accuracy"] else "",
                         "behaviour_accuracy": round(p["behaviour_accuracy"], 6),
                         "tau": p["tau"]})
    for cov, sel, beh in [(0.298, 0.517, 0.538), (0.480, 0.463, 0.514)]:
        rows.append({"series": "TwoStage", "coverage": cov, "selective_accuracy": sel,
                     "behaviour_accuracy": beh, "tau": ""})
    with (OUT / "plot_data.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["series", "coverage", "selective_accuracy",
                                          "behaviour_accuracy", "tau"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    fig1()
    fig2()
    write_data()
    print("wrote", OUT)
