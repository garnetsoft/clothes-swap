import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageOps
import threading
import numpy as np
import torch
import logging
logging.disable(logging.WARNING)

from datetime import datetime


# ── lazy-loaded globals ───────────────────────────────────────────────────────
seg_processor = None
seg_model      = None

CLOTHING_LABELS = {
    "Upper body (shirt/jacket)": [4, 7],
    "Lower body (pants/skirt)":  [6, 12],
    "Dress":                     [7],
    "All clothing":              [4, 6, 7, 12],
}


# ── color vocabulary ──────────────────────────────────────────────────────────
# (hue in [0,1], base_saturation in [0,1])
COLOR_MAP = {
    # reds
    "red":         (0.000, 0.90),
    "crimson":     (0.972, 0.95),
    "scarlet":     (0.028, 0.95),
    "maroon":      (0.000, 0.90),
    "rose":        (0.944, 0.70),
    "brick":       (0.028, 0.80),
    "rust":        (0.040, 0.85),
    # oranges
    "orange":      (0.083, 0.95),
    "coral":       (0.044, 0.80),
    "salmon":      (0.044, 0.55),
    "peach":       (0.080, 0.45),
    "terracotta":  (0.050, 0.75),
    # yellows
    "yellow":      (0.167, 0.95),
    "gold":        (0.142, 0.90),
    "amber":       (0.111, 0.90),
    "mustard":     (0.139, 0.85),
    "lemon":       (0.180, 0.80),
    "cream":       (0.150, 0.20),
    "ivory":       (0.150, 0.12),
    # greens
    "green":       (0.333, 0.85),
    "lime":        (0.250, 0.90),
    "olive":       (0.200, 0.70),
    "forest":      (0.333, 0.80),
    "mint":        (0.417, 0.45),
    "emerald":     (0.389, 0.90),
    "teal":        (0.500, 0.85),
    "sage":        (0.333, 0.35),
    "khaki":       (0.167, 0.40),
    # blues
    "blue":        (0.667, 0.90),
    "navy":        (0.667, 0.95),
    "sky":         (0.556, 0.55),
    "azure":       (0.556, 0.80),
    "cobalt":      (0.611, 0.90),
    "indigo":      (0.722, 0.85),
    "cyan":        (0.500, 0.90),
    "turquoise":   (0.481, 0.75),
    "denim":       (0.600, 0.65),
    "aqua":        (0.500, 0.80),
    "cerulean":    (0.556, 0.75),
    # purples
    "purple":      (0.750, 0.85),
    "violet":      (0.789, 0.85),
    "lavender":    (0.750, 0.35),
    "lilac":       (0.750, 0.30),
    "magenta":     (0.833, 0.90),
    "plum":        (0.833, 0.70),
    "mauve":       (0.833, 0.35),
    "wine":        (0.917, 0.80),
    "burgundy":    (0.944, 0.90),
    "eggplant":    (0.750, 0.80),
    # pinks
    "pink":        (0.917, 0.55),
    "hot pink":    (0.917, 0.95),
    "fuchsia":     (0.833, 0.95),
    # browns
    "brown":       (0.083, 0.75),
    "tan":         (0.100, 0.40),
    "beige":       (0.100, 0.25),
    "camel":       (0.100, 0.50),
    "chocolate":   (0.067, 0.80),
    "sienna":      (0.055, 0.72),
    "coffee":      (0.083, 0.70),
}

# Achromatic colors: name → target value (brightness) in [0,1]
ACHROMATIC_MAP = {
    "white":    0.95,
    "ivory":    0.90,
    "cream":    0.88,
    "silver":   0.75,
    "gray":     0.50,
    "grey":     0.50,
    "charcoal": 0.25,
    "black":    0.08,
}

# Modifiers: keyword → (sat_multiplier, val_multiplier)
MODIFIERS = {
    "dark":    (1.10, 0.60),
    "deep":    (1.20, 0.55),
    "rich":    (1.20, 0.75),
    "light":   (0.60, 1.30),
    "pale":    (0.35, 1.30),
    "pastel":  (0.40, 1.25),
    "bright":  (1.40, 1.05),
    "vivid":   (1.50, 1.00),
    "neon":    (1.80, 1.00),
    "muted":   (0.50, 0.85),
    "dull":    (0.50, 0.80),
    "faded":   (0.40, 0.95),
    "washed":  (0.35, 1.10),
    "warm":    (1.05, 1.00),
}


def parse_color_prompt(prompt: str):
    """
    Parse a natural-language color description.
    Returns (hue, base_sat, sat_mult, val_mult, is_achromatic, achromatic_v)
      hue / base_sat  in [0,1], None if achromatic
    Returns None if no recognisable color found.
    """
    text = prompt.lower().strip()

    # Collect modifiers
    sat_mult = val_mult = 1.0
    for mod, (sm, vm) in MODIFIERS.items():
        if mod in text:
            sat_mult *= sm
            val_mult *= vm

    # Achromatic check (longer names first to avoid "gray" shadowing "charcoal")
    for name in sorted(ACHROMATIC_MAP, key=len, reverse=True):
        if name in text:
            return None, None, sat_mult, val_mult, True, ACHROMATIC_MAP[name]

    # Chromatic: find longest matching name
    best_name, best_hue, best_s = None, None, None
    for name, (h, s) in COLOR_MAP.items():
        if name in text and (best_name is None or len(name) > len(best_name)):
            best_name, best_hue, best_s = name, h, s

    if best_hue is None:
        return None   # nothing found

    return best_hue, best_s, sat_mult, val_mult, False, None


def parsed_to_preview_hex(parse_result) -> str:
    """Convert parse result to a hex color string for the UI swatch."""
    if parse_result is None:
        return "#888888"
    hue, base_s, sat_mult, val_mult, is_achromatic, achromatic_v = parse_result
    import colorsys
    if is_achromatic:
        v = min(1.0, achromatic_v * val_mult)
        r, g, b = v, v, v
    else:
        s = min(1.0, base_s * sat_mult)
        v = min(1.0, 0.85 * val_mult)
        r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(status_cb):
    global seg_processor, seg_model
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    t0 = datetime.now()
    status_cb("Loading segmentation model...")
    seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model     = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model.eval()
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"Model loaded in {elapsed:.2f}s")
    status_cb(f"Model ready ({elapsed:.1f}s). Load an image to get started.")


# ── segmentation ──────────────────────────────────────────────────────────────
def segment_clothing(image: Image.Image, label_ids: list) -> np.ndarray:
    """Returns a binary uint8 mask (255 = clothing, 0 = background)."""
    inputs = seg_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = seg_model(**inputs)
    upsampled = torch.nn.functional.interpolate(
        outputs.logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    pred = upsampled.argmax(dim=1).squeeze().numpy()
    mask = np.zeros_like(pred, dtype=np.uint8)
    for lid in label_ids:
        mask[pred == lid] = 255
    return mask


# ── vectorised HSV helpers ────────────────────────────────────────────────────
def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    r, g, b  = img[..., 0], img[..., 1], img[..., 2]
    maxc     = np.maximum(r, np.maximum(g, b))
    minc     = np.minimum(r, np.minimum(g, b))
    delta    = maxc - minc
    v        = maxc
    s        = np.where(maxc > 0, delta / maxc, 0.0)
    h        = np.zeros_like(r)
    nz       = delta > 0
    mr, mg, mb = nz & (maxc == r), nz & (maxc == g), nz & (maxc == b)
    h[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6
    h[mg] = (b[mg] - r[mg]) / delta[mg] + 2
    h[mb] = (r[mb] - g[mb]) / delta[mb] + 4
    return np.stack([(h / 6.0) % 1.0, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h6 = hsv[..., 0] * 6
    s, v = hsv[..., 1], hsv[..., 2]
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p, q, t = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
    out = np.zeros(hsv.shape, dtype=np.float32)
    for k, (rv, gv, bv) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        m = i == k
        out[m] = np.stack([rv, gv, bv], axis=-1)[m]
    return out


# ── color application ─────────────────────────────────────────────────────────
def apply_color(image: Image.Image, mask: np.ndarray, parse_result,
                sat_slider: float, val_slider: float) -> Image.Image:
    """Apply parsed color to masked clothing pixels, preserving per-pixel shading."""
    hue, base_s, sat_mult, val_mult, is_achromatic, achromatic_v = parse_result

    img_arr = np.array(image, dtype=np.float32) / 255.0
    hsv     = _rgb_to_hsv(img_arr)
    cloth   = mask > 128

    eff_sat = min(sat_mult * sat_slider, 3.0)
    eff_val = val_mult * val_slider

    if is_achromatic:
        # Remove colour, shift brightness toward achromatic_v
        hsv[cloth, 1] = np.clip(hsv[cloth, 1] * 0.08 * eff_sat, 0, 1)
        # blend original V toward achromatic_v
        orig_v = hsv[cloth, 2]
        hsv[cloth, 2] = np.clip(orig_v * achromatic_v * 2 * eff_val, 0, 1)
    else:
        target_s = min(1.0, base_s * eff_sat)
        hsv[cloth, 0] = hue
        hsv[cloth, 1] = target_s
        hsv[cloth, 2] = np.clip(hsv[cloth, 2] * eff_val, 0, 1)

    return Image.fromarray((_hsv_to_rgb(hsv) * 255).astype(np.uint8))


# ── GUI ───────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clothing Color Swap")
        self.resizable(True, True)
        self.geometry("720x560")
        self.configure(bg="#1e1e1e")

        self.orig_image      = None
        self.orig_image_name = None
        self.result_image    = None
        self._parse_result   = None

        self._build_ui()
        self._load_model_async()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        PAD = 10
        BG  = "#1e1e1e"
        FG  = "#e0e0e0"
        ACC = "#4a9eff"
        BTN = {"bg": "#2d2d2d", "fg": FG, "activebackground": "#3a3a3a",
               "activeforeground": FG, "relief": "flat", "cursor": "hand2",
               "font": ("Segoe UI", 10)}

        ctrl = tk.Frame(self, bg=BG, padx=PAD, pady=PAD)
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew")

        # row 0 — action buttons + garment selector
        tk.Button(ctrl, text="Load Image", command=self._load_image,
                  **BTN, padx=12, pady=6).grid(row=0, column=0, padx=(0, 8))

        tk.Label(ctrl, text="Garment:", bg=BG, fg=FG,
                 font=("Segoe UI", 10)).grid(row=0, column=1, padx=(0, 4))
        self.garment_var = tk.StringVar(value=list(CLOTHING_LABELS)[0])
        ttk.Combobox(ctrl, textvariable=self.garment_var,
                     values=list(CLOTHING_LABELS), state="readonly",
                     width=24, font=("Segoe UI", 10)).grid(row=0, column=2, padx=(0, 12))

        self.run_btn = tk.Button(ctrl, text="Run", command=self._run,
                                 bg=ACC, fg="white", activebackground="#3a8ae0",
                                 activeforeground="white", relief="flat",
                                 cursor="hand2", font=("Segoe UI", 10, "bold"),
                                 padx=14, pady=6, state="disabled")
        self.run_btn.grid(row=0, column=3, padx=(0, 8))

        tk.Button(ctrl, text="Save Result", command=self._save,
                  **BTN, padx=10, pady=6).grid(row=0, column=4)

        # row 1 — prompt entry + color preview swatch
        tk.Label(ctrl, text="Color prompt:", bg=BG, fg=ACC,
                 font=("Segoe UI", 10, "bold")).grid(row=1, column=0, padx=(0, 6), pady=(8, 0), sticky="e")

        self.prompt_var = tk.StringVar(value="dark navy blue")
        prompt_entry = tk.Entry(ctrl, textvariable=self.prompt_var,
                                bg="#2d2d2d", fg="white", insertbackground="white",
                                font=("Segoe UI", 11), relief="flat",
                                highlightthickness=1, highlightcolor=ACC,
                                highlightbackground="#555")
        prompt_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(8, 0))
        prompt_entry.bind("<Return>", lambda e: self._update_swatch())
        prompt_entry.bind("<FocusOut>", lambda e: self._update_swatch())

        self.swatch = tk.Label(ctrl, width=4, relief="flat", bg="#001f5c")
        self.swatch.grid(row=1, column=3, padx=(8, 0), pady=(8, 0), sticky="w")

        self.parse_lbl = tk.Label(ctrl, text="", bg=BG, fg="#888",
                                  font=("Segoe UI", 8))
        self.parse_lbl.grid(row=1, column=4, padx=(6, 0), pady=(8, 0), sticky="w")

        # row 2 — saturation fine-tune
        tk.Label(ctrl, text="Saturation:", bg=BG, fg=FG,
                 font=("Segoe UI", 9)).grid(row=2, column=0, padx=(0, 6), pady=(6, 0), sticky="e")
        self.sat_var = tk.DoubleVar(value=1.0)
        tk.Scale(ctrl, variable=self.sat_var, from_=0.2, to=2.5, resolution=0.05,
                 orient="horizontal", length=200, bg=BG, fg=FG,
                 troughcolor="#333", highlightthickness=0,
                 font=("Segoe UI", 8)).grid(row=2, column=1, columnspan=2, sticky="w", pady=(6, 0))

        # row 3 — brightness fine-tune
        tk.Label(ctrl, text="Brightness:", bg=BG, fg=FG,
                 font=("Segoe UI", 9)).grid(row=3, column=0, padx=(0, 6), pady=(2, 0), sticky="e")
        self.val_var = tk.DoubleVar(value=1.0)
        tk.Scale(ctrl, variable=self.val_var, from_=0.3, to=1.5, resolution=0.05,
                 orient="horizontal", length=200, bg=BG, fg=FG,
                 troughcolor="#333", highlightthickness=0,
                 font=("Segoe UI", 8)).grid(row=3, column=1, columnspan=2, sticky="w", pady=(2, 0))

        ctrl.columnconfigure(2, weight=1)

        # image panels
        lbl_cfg   = {"bg": BG, "fg": "#888", "font": ("Segoe UI", 9)}
        panel_cfg = {"bg": "#121212", "width": 320, "height": 320, "relief": "flat"}

        tk.Label(self, text="Original", **lbl_cfg).grid(row=1, column=0, pady=(4, 0))
        tk.Label(self, text="Result",   **lbl_cfg).grid(row=1, column=1, pady=(4, 0))

        self.panel_orig   = tk.Label(self, **panel_cfg)
        self.panel_result = tk.Label(self, **panel_cfg)
        self.panel_orig.grid(  row=2, column=0, padx=(PAD, 4), pady=(2, PAD), sticky="nsew")
        self.panel_result.grid(row=2, column=1, padx=(4, PAD), pady=(2, PAD), sticky="nsew")

        self.columnconfigure(0, weight=1, uniform="panel")
        self.columnconfigure(1, weight=1, uniform="panel")
        self.rowconfigure(2, weight=1)
        self.bind("<Configure>", self._on_resize)

        # status bar
        self.status_var = tk.StringVar(value="Loading model...")
        tk.Label(self, textvariable=self.status_var, bg="#111", fg="#aaa",
                 font=("Segoe UI", 9), anchor="w", padx=8, pady=4
                 ).grid(row=3, column=0, columnspan=2, sticky="ew")

        # progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("dark.Horizontal.TProgressbar",
                        troughcolor="#2d2d2d", background=ACC, thickness=4)
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=640,
                                        style="dark.Horizontal.TProgressbar")
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew")

        # initialise swatch for default prompt
        self._update_swatch()

    # ── swatch update ─────────────────────────────────────────────────────────
    def _update_swatch(self):
        result = parse_color_prompt(self.prompt_var.get())
        self._parse_result = result
        if result is None:
            self.swatch.config(bg="#555")
            self.parse_lbl.config(text="(not recognised)", fg="#f66")
        else:
            hex_col = parsed_to_preview_hex(result)
            self.swatch.config(bg=hex_col)
            hue, base_s, sat_mult, val_mult, is_achromatic, achromatic_v = result
            if is_achromatic:
                self.parse_lbl.config(text=f"achromatic  v={achromatic_v:.2f}", fg="#888")
            else:
                self.parse_lbl.config(
                    text=f"h={hue:.3f}  s={base_s:.2f}  x{sat_mult:.1f}/{val_mult:.1f}",
                    fg="#888")

    # ── model loading ─────────────────────────────────────────────────────────
    def _load_model_async(self):
        self.progress.start(12)
        def _work():
            try:
                load_model(lambda msg: self.status_var.set(msg))
                self.after(0, self._on_model_ready)
            except Exception as e:
                self.after(0, lambda err=e: self._on_error(f"Model load failed: {err}"))
        threading.Thread(target=_work, daemon=True).start()

    def _on_model_ready(self):
        self.progress.stop()
        self.progress["value"] = 0
        if self.orig_image:
            self.run_btn.config(state="normal")

    # ── load image ────────────────────────────────────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if not path:
            return
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        self.orig_image      = img
        self.orig_image_name = os.path.basename(path)
        self._show(self.panel_orig, img)
        self.panel_result.config(image="", text="")
        self.result_image = None
        if seg_model is not None:
            self.run_btn.config(state="normal")
        self.status_var.set(f"Loaded: {self.orig_image_name}")

    # ── run pipeline ──────────────────────────────────────────────────────────
    def _run(self):
        if not self.orig_image:
            messagebox.showwarning("No image", "Please load an image first.")
            return

        self._update_swatch()
        if self._parse_result is None:
            messagebox.showwarning("Unknown color",
                                   f"Could not recognise a color in: {self.prompt_var.get()!r}\n\n"
                                   "Try something like: 'dark navy blue', 'bright red', 'pale green'.")
            return

        parse_result = self._parse_result
        label_ids    = CLOTHING_LABELS[self.garment_var.get()]
        sat_slider   = self.sat_var.get()
        val_slider   = self.val_var.get()

        self.run_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Segmenting clothing...")

        def _work():
            try:
                t0   = datetime.now()
                mask = segment_clothing(self.orig_image, label_ids)
                t1   = datetime.now()
                self.after(0, lambda: self.status_var.set("Applying color..."))
                result = apply_color(self.orig_image, mask, parse_result, sat_slider, val_slider)
                t2 = datetime.now()
                print(f"Segmentation: {(t1-t0).total_seconds():.2f}s  |  "
                      f"Color apply: {(t2-t1).total_seconds():.3f}s")
                self.result_image = result
                self.after(0, self._on_done, result)
            except Exception as e:
                self.after(0, lambda err=e: self._on_error(f"Pipeline error: {err}"))

        threading.Thread(target=_work, daemon=True).start()

    def _on_done(self, result: Image.Image):
        self.progress.stop()
        self._show(self.panel_result, result)
        self.run_btn.config(state="normal")
        self.status_var.set("Done  — use Save Result to export")

    # ── save ──────────────────────────────────────────────────────────────────
    def _save(self):
        if not self.result_image:
            messagebox.showinfo("Nothing to save", "Run the pipeline first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self.result_image.save(path)
            self.status_var.set(f"Saved -> {path}")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _show(self, panel: tk.Label, img: Image.Image):
        w = max(self.winfo_width() // 2, 100)
        h = max(panel.winfo_height() if panel.winfo_height() > 1 else self.winfo_height(), 100)
        self.panel_orig.config(width=w)
        self.panel_result.config(width=w)
        thumb = img.copy()
        thumb.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(thumb)
        panel.config(image=photo, text="", height=h)
        panel.image = photo

    def _on_resize(self, event):
        if event.widget is not self:
            return
        if hasattr(self, "_resize_job"):
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self._do_resize)

    def _do_resize(self):
        if self.orig_image:
            self._show(self.panel_orig, self.orig_image)
        if self.result_image:
            self._show(self.panel_result, self.result_image)

    def _on_error(self, msg: str):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status_var.set(f"Error: {msg}")
        messagebox.showerror("Error", msg)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
