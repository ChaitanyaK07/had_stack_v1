"""Side-by-side panel: each results figure of the bond paper next to the same notebook cell re-run here.
  python run_nb.py --save-plots
  python compare_figures.py <path/to/bond_paper.pdf>      -> paper_vs_rerun.png   (needs poppler's pdfimages)"""
import os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); PLOTS = os.path.join(HERE, "plots")
tmp = tempfile.mkdtemp(); subprocess.run(["pdfimages", "-png", sys.argv[1], os.path.join(tmp, "img")], check=True)
pairs = [("Fig. 3 (captioned AR(1)) vs cell 6", "img-000.png", "cell06_2.png"), ("Fig. 4 (captioned GBM) vs cell 8", "img-002.png", "cell08_1.png"),
         ("Fig. 5 Ensemble+Residual MLP vs cell 10", "img-004.png", "cell10_1.png"), ("Fig. 6 Hybrid ARIMA-LSTM vs cell 11", "img-006.png", "cell11_1.png"),
         ("Fig. 7 LSTM on Delta vs cell 12", "img-008.png", "cell12_1.png"), ("Fig. 8 Two-Head LSTM vs cell 13", "img-010.png", "cell13_1.png")]
W, H, hdr = 700, 300, 34
try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except OSError: f = ImageFont.load_default()
c = Image.new("RGB", (2 * W + 30, len(pairs) * (H + hdr) + 40), "white"); d = ImageDraw.Draw(c)
d.text((10, 8), "Left: figure as printed in the bond paper.   Right: same notebook cell re-run.", fill="black", font=f)
for k, (t, a, b) in enumerate(pairs):
    y = 40 + k * (H + hdr); d.text((10, y + 6), t, fill="black", font=f)
    for j, p in enumerate((os.path.join(tmp, a), os.path.join(PLOTS, b))):
        im = Image.open(p).convert("RGB"); im.thumbnail((W, H)); c.paste(im, (10 + j * (W + 10), y + hdr))
c.save(os.path.join(HERE, "paper_vs_rerun.png")); print("wrote paper_vs_rerun.png")
