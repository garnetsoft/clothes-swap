import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"]    = "error"

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageOps
import threading
import numpy as np
import torch
import logging
logging.disable(logging.WARNING)

import datetime 
from datetime import datetime 


# ── lazy-loaded globals ──────────────────────────────────────────────────────
seg_processor = None
seg_model = None
inpaint_pipe = None

CLOTHING_LABELS = {
    "Upper body (shirt/jacket)": [4, 7],   # upper-clothes, coat
    "Lower body (pants/skirt)":  [6, 12],  # pants, skirt
    "Dress":                     [7],
    "All clothing":              [4, 6, 7, 12],
}

# ── model loading ────────────────────────────────────────────────────────────
def load_models(status_cb):
    global seg_processor, seg_model, inpaint_pipe
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from diffusers import AutoPipelineForInpainting
    
    stime = datetime.now()
    print(f"Segmentation model loaded with labels: {datetime.now()}")

    status_cb("Loading segmentation model…")
    seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model     = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model.eval()

    print(f"Loading inpainting model (may take a minute)…{datetime.now()}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype  = torch.float16 if device != "cpu" else torch.float32

    status_cb(f"Loading inpainting model (may take a minute)… [{device}]")
    inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
    )
    inpaint_pipe.enable_sequential_cpu_offload()
    inpaint_pipe.enable_attention_slicing(1)
    inpaint_pipe.enable_vae_tiling()
    inpaint_pipe.set_progress_bar_config(disable=True)

    print(f"Segmentation model loaded with labels  — total load time: {(datetime.now() - stime).total_seconds():.2f}s")
    status_cb(f"Models ready ✓  ({(datetime.now() - stime).total_seconds():.1f}s). Please load an image to get started.")


# ── segmentation ─────────────────────────────────────────────────────────────
def segment_clothing(image: Image.Image, label_ids: list[int]) -> Image.Image:
    """Returns a binary PIL mask (white = clothing, black = background)."""
    inputs = seg_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = seg_model(**inputs)

    logits = outputs.logits                          # (1, num_labels, H, W)
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],                       # (H, W)
        mode="bilinear",
        align_corners=False,
    )
    pred = upsampled.argmax(dim=1).squeeze().numpy() # (H, W)

    mask = np.zeros_like(pred, dtype=np.uint8)
    for lid in label_ids:
        mask[pred == lid] = 255

    return Image.fromarray(mask).convert("RGB")


# ── inpainting ───────────────────────────────────────────────────────────────
def inpaint(image: Image.Image, mask: Image.Image, prompt: str,
            progress_cb=None) -> Image.Image:
    TARGET = 384
    orig_size = image.size
    total_steps = 8

    img_r  = image.resize((TARGET, TARGET), Image.LANCZOS)
    mask_r = mask.resize((TARGET, TARGET),  Image.NEAREST)

    def _step_cb(_pipe, step, _timestep, kwargs):
        if progress_cb:
            pct = round((step + 1) / total_steps * 100)
            progress_cb(pct)
        return kwargs

    result = inpaint_pipe(
        prompt=prompt,
        image=img_r,
        mask_image=mask_r,
        num_inference_steps=total_steps,
        guidance_scale=7.5,
        strength=0.99,
        callback_on_step_end=_step_cb,
    ).images[0]

    return result.resize(orig_size, Image.LANCZOS)


# ── GUI ───────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clothes Swap — local AI")
        self.resizable(True, True)
        self.geometry("720x480")
        self.configure(bg="#1e1e1e")

        self.orig_image      = None
        self.orig_image_name = None
        self.result_image    = None
        self._build_ui()
        self._load_models_async()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        PAD = 10
        BG  = "#1e1e1e"
        FG  = "#e0e0e0"
        ACC = "#4a9eff"
        BTN = {"bg": "#2d2d2d", "fg": FG, "activebackground": "#3a3a3a",
               "activeforeground": FG, "relief": "flat", "cursor": "hand2",
               "font": ("Segoe UI", 10)}

        # ── top controls ──
        ctrl = tk.Frame(self, bg=BG, padx=PAD, pady=PAD)
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew")

        tk.Button(ctrl, text="📂  Load Image", command=self._load_image, **BTN,
                  padx=12, pady=6).grid(row=0, column=0, padx=(0, 8))

        print(f"Available clothing labels: {list(CLOTHING_LABELS)}")

        tk.Label(ctrl, text="Garment:", bg=BG, fg=FG,
                 font=("Segoe UI", 10)).grid(row=0, column=1, padx=(0, 4))
        self.garment_var = tk.StringVar(value=list(CLOTHING_LABELS)[0])
        garment_menu = ttk.Combobox(ctrl, textvariable=self.garment_var,
                                    values=list(CLOTHING_LABELS), state="readonly",
                                    width=28, font=("Segoe UI", 10))
        garment_menu.grid(row=0, column=2, padx=(0, 12))

        self.run_btn = tk.Button(ctrl, text="▶  Run", command=self._run,
                                 bg=ACC, fg="white", activebackground="#3a8ae0",
                                 activeforeground="white", relief="flat",
                                 cursor="hand2", font=("Segoe UI", 10, "bold"),
                                 padx=14, pady=6, state="disabled")
        self.run_btn.grid(row=0, column=3, padx=(0, 8))

        # ── prompt row ──
        tk.Label(ctrl, text="New clothes prompt:", bg=BG, fg=ACC,
                 font=("Segoe UI", 10, "bold")).grid(row=1, column=0, padx=(0, 6), pady=(6, 0), sticky="e")
        self.prompt_var = tk.StringVar(value="a stylish red wool sweater")
        prompt_entry = tk.Entry(ctrl, textvariable=self.prompt_var,
                                bg="#2d2d2d", fg="white", insertbackground="white",
                                font=("Segoe UI", 11), relief="flat",
                                highlightthickness=1, highlightcolor=ACC,
                                highlightbackground="#555")
        prompt_entry.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        ctrl.columnconfigure(2, weight=1)

        tk.Button(ctrl, text="💾  Save Result", command=self._save,
                  **BTN, padx=10, pady=6).grid(row=0, column=6)

        # ── image panels ──
        panel_cfg = {"bg": "#121212", "width": 320, "height": 320,
                     "relief": "flat"}
        lbl_cfg   = {"bg": BG, "fg": "#888", "font": ("Segoe UI", 9)}

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

        # ── status bar ──
        self.status_var = tk.StringVar(value="Loading models…")
        tk.Label(self, textvariable=self.status_var, bg="#111", fg="#aaa",
                 font=("Segoe UI", 9), anchor="w", padx=8, pady=4
                 ).grid(row=3, column=0, columnspan=2, sticky="ew")

        # ── progress bar ──
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("dark.Horizontal.TProgressbar",
                        troughcolor="#2d2d2d", background=ACC, thickness=4)
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=640,
                                        style="dark.Horizontal.TProgressbar")
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew")

    # ── model loading (background thread) ────────────────────────────────────
    def _load_models_async(self):
        self.progress.start(12)
        def _work():
            try:
                print(f"Starting model load at {datetime.now()}")

                load_models(lambda msg: self.status_var.set(msg))
                self.after(0, self._on_models_ready)

                print(f"Model load completed at {datetime.now()}, please load an image to get started.")

            except Exception as e:
                err = e
                print(f"Model load failed: {err}")

                #self.after(0, lambda err=err: self._on_error(f"Model load failed: {err}"))
        threading.Thread(target=_work, daemon=True).start()

    def _on_models_ready(self):
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
        self.orig_image = img
        self.orig_image_name = path.split('/')[-1].split(chr(92))[-1]
        self._show(self.panel_orig, img)
        self.panel_result.config(image="", text="")
        self.result_image = None
        if seg_model is not None:
            self.run_btn.config(state="normal")
        self.status_var.set(f"Loaded: {path.split('/')[-1].split(chr(92))[-1]}")

    # ── run pipeline ──────────────────────────────────────────────────────────
    def _run(self):
        if not self.orig_image:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        prompt      = self.prompt_var.get().strip()
        label_ids   = CLOTHING_LABELS[self.garment_var.get()]
        print(f"Run clicked — image: {self.orig_image_name!r}, garment: {self.garment_var.get()!r}, prompt: {prompt!r}, label_ids: {label_ids!r}")
        self.run_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Segmenting clothing…")

        def _work():
            try:
                t0 = datetime.now()
                mask = segment_clothing(self.orig_image, label_ids)
                t1 = datetime.now()
                print(f"Segmentation done in {(t1 - t0).total_seconds():.2f}s")
                self.after(0, lambda: self.status_var.set("Inpainting… 0%"))
                def _on_progress(pct):
                    self.after(0, lambda p=pct: self.status_var.set(f"Inpainting… {p}%"))
                result = inpaint(self.orig_image, mask, prompt, progress_cb=_on_progress)
                t2 = datetime.now()
                print(f"Inpainting done in {(t2 - t1).total_seconds():.2f}s")
                print(f"Total pipeline time: {(t2 - t0).total_seconds():.2f}s")
                self.result_image = result
                self.after(0, self._on_done, result)
            except Exception as e:
                err = e
                self.after(0, lambda err=err: self._on_error(f"Pipeline error: {err}"))

        threading.Thread(target=_work, daemon=True).start()

    def _on_done(self, result: Image.Image):
        self.progress.stop()
        self._show(self.panel_result, result)
        self.run_btn.config(state="normal")
        self.status_var.set("Done ✓  —  use 💾 Save Result to export")

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
            self.status_var.set(f"Saved → {path}")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _show(self, panel: tk.Label, img: Image.Image):
        w = max(self.winfo_width() // 2, 100)
        h = max(panel.winfo_height() if panel.winfo_height() > 1 else self.winfo_height(), 100)
        # keep both columns the same width at all times
        self.panel_orig.config(width=w)
        self.panel_result.config(width=w)
        thumb = img.copy()
        thumb.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(thumb)
        panel.config(image=photo, text="", height=h)
        panel.image = photo          # keep reference

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
