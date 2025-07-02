import streamlit as st
from PIL import Image
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

# ========== MODEL SETUP ==========
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model_path = "./BLIP-ECG-Capt-Final.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="ECG-BLIP", layout="wide")

# ========== HEADER ==========
header_col1, header_col2 = st.columns([0.07, 0.93])
with header_col1:
    logo_img = Image.open(r"C:\Users\62838\Downloads\logoekg.png")
    st.markdown("<div style='padding-top: 36px;'>", unsafe_allow_html=True)
    st.image(logo_img, width=90)
    st.markdown("</div>", unsafe_allow_html=True)
with header_col2:
    st.markdown(
        "<h1 style='margin: 0px; padding-top: 24px;'>ECG-BLIP: Automated Diagnostic Caption from Electrocardiogram Images</h1>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ========== INSTRUCTIONS & UPLOADER ==========
left_col, right_col = st.columns([1.2, 1])
with left_col:
    st.markdown("### Instructions:")
    st.markdown(
        """
        <div style="text-align: left; font-size:14px;">
        Using this ECG-BLIP, users can generate a complete diagnostic report by uploading a scanned image of a 12-lead ECG. 
        This tool is for demonstration only and is not intended for clinical use.
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "jpeg", "png"])

# ========== PREPROCESSING FUNCTION ==========
def adjust_brightness_contrast(img_array, target_brightness, target_contrast):
    mean_img = img_array.mean()
    std_img = img_array.std()
    if std_img == 0:
        return img_array.astype(np.uint8)
    norm = (img_array - mean_img) * (target_contrast / std_img) + target_brightness
    norm = np.clip(norm, 0, 255)
    return norm.astype(np.uint8)

# ========== MAIN PROCESSING ==========
with right_col:
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        width, height = image.size

        # Step 1: Crop 2200x1265 from bottom center
        crop_top = height - 1265
        crop_left = (width - 2200) // 2
        cropped = image.crop((crop_left, crop_top, crop_left + 2200, height))

        # Step 2: Resize to 224x224
        resized = cropped.resize((224, 224), resample=Image.LANCZOS)

        # Step 3: Convert to grayscale
        gray = resized.convert('L')
        arr = np.array(gray, dtype=np.float32)

        # Step 4: Brightness & contrast adjustment
        target_brightness = 219.36
        target_contrast = 20.0
        adjusted_arr = adjust_brightness_contrast(arr, target_brightness, target_contrast)
        adjusted_img = Image.fromarray(adjusted_arr).convert('RGB')  # convert back to RGB for model

        # Display preprocessed image
        st.image(adjusted_img, caption="Preprocessed ECG Image", use_container_width=False, width=250)

        # Step 5: Inference Caption with max_length=60
        inputs = processor(images=adjusted_img, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_length=60)
        generated_caption = processor.decode(output[0], skip_special_tokens=True)

        st.markdown("### Generated Caption:")
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6; padding:12px; border-radius:8px; font-size:14px;">
            {generated_caption}
            </div>
            """,
            unsafe_allow_html=True
        )