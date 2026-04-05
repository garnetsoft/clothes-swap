import base64
from io import BytesIO
from pathlib import Path

import httpx
import streamlit as st
import yaml
from PIL import Image

_cfg_path = Path(__file__).parents[1] / "config.yaml"
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)

BACKEND = _cfg["server"]["backend_url"]

GARMENT_TYPES = {
    "Upper body (shirt / jacket)": "upper",
    "Lower body (pants / skirt)":  "lower",
    "Dress":                       "dress",
    "All clothing":                "all",
}

st.set_page_config(page_title="Clothes Swap", layout="wide")
st.title("Clothes Swap")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Garment Swap", "Text Prompt"], index=0)
    garment_label = st.selectbox("Garment region", list(GARMENT_TYPES))
    garment_type  = GARMENT_TYPES[garment_label]

    st.divider()
    if st.button("Check backend status"):
        try:
            r = httpx.get(f"{BACKEND}/status", timeout=5)
            st.json(r.json())
        except Exception as e:
            st.error(f"Backend unreachable: {e}")

# ── upload area ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Person photo")
    person_file = st.file_uploader(
        "Upload person image", type=["jpg", "jpeg", "png", "webp"], key="person"
    )
    if person_file:
        st.image(person_file, width='stretch')

with col2:
    if mode == "Garment Swap":
        st.subheader("Garment photo")
        garment_file = st.file_uploader(
            "Upload garment image", type=["jpg", "jpeg", "png", "webp"], key="garment"
        )
        if garment_file:
            st.image(garment_file, width='stretch')
    else:
        st.subheader("Text prompt")
        prompt = st.text_area(
            "Describe the new clothing",
            value="a stylish navy blue linen blazer",
            height=120,
        )
        st.caption("Text-prompt mode is Phase 2 — currently a stub.")

# ── run ───────────────────────────────────────────────────────────────────────
st.divider()

run_ready = bool(person_file) and (
    (mode == "Garment Swap" and garment_file) or
    (mode == "Text Prompt")
)

if st.button("Run Swap", disabled=not run_ready, type="primary"):
    with st.spinner("Running… this may take a minute on first run while models download."):
        try:
            if mode == "Garment Swap":
                resp = httpx.post(
                    f"{BACKEND}/swap/garment",
                    files={
                        "person_image":  ("person.jpg",  person_file.getvalue(),  "image/jpeg"),
                        "garment_image": ("garment.jpg", garment_file.getvalue(), "image/jpeg"),
                    },
                    data={"garment_type": garment_type},
                    timeout=300,
                )
            else:
                resp = httpx.post(
                    f"{BACKEND}/swap/text",
                    files={"person_image": ("person.jpg", person_file.getvalue(), "image/jpeg")},
                    data={"prompt": prompt, "garment_type": garment_type},
                    timeout=300,
                )

            if resp.status_code == 501:
                st.warning(resp.json().get("detail", "Not implemented yet."))
            elif resp.status_code != 200:
                st.error(f"Backend error {resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                result_img = Image.open(BytesIO(base64.b64decode(data["image_base64"])))

                st.subheader("Result")
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.caption("Original")
                    st.image(person_file, width='stretch')
                with rc2:
                    st.caption("Swapped")
                    st.image(result_img, width='stretch')

                buf = BytesIO()
                result_img.save(buf, format="PNG")
                st.download_button(
                    "Download result", buf.getvalue(),
                    file_name="swap_result.png", mime="image/png"
                )

                if "saved_path" in data:
                    st.caption(f"Saved to: {data['saved_path']}")

        except httpx.TimeoutException:
            st.error("Request timed out. Models may still be loading — try again shortly.")
        except Exception as e:
            st.error(f"Error: {e}")
