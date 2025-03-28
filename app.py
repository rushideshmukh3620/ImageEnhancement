import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Function to enhance image (Sharpening & Denoising)
def enhance_image(img):
    img = np.array(img)  # Convert to NumPy array

    # Sharpening
    sharpen_kernel = np.array([[0, -1, 0], 
                               [-1, 5, -1], 
                               [0, -1, 0]])
    img_sharp = cv2.filter2D(img, -1, sharpen_kernel)

    # Denoising
    img_denoise = cv2.fastNlMeansDenoisingColored(img_sharp, None, 10, 10, 7, 21)

    return img_denoise

# Streamlit UI
st.title("🖼️ Multi-Image Enhancement App")
st.write("Upload one or more images to enhance and compare before & after results.")

# Upload Multiple Images
uploaded_imgs = st.file_uploader("📤 Upload Images", type=["jpg", "jpeg", "png", "jfif", "bmp", "tiff", "webp"], accept_multiple_files=True)

if uploaded_imgs:
    for uploaded_img in uploaded_imgs:
        st.write(f"### {uploaded_img.name}")  # Display original filename

        # Load Image
        image = Image.open(uploaded_img)

        # Process Image (Enhance)
        enhanced_image = enhance_image(image)

        # Convert Enhanced Image back to PIL format
        enhanced_pil = Image.fromarray(enhanced_image)

        # Ensure the image is in RGB mode before saving as JPEG
        if enhanced_pil.mode != 'RGB':
            enhanced_pil = enhanced_pil.convert('RGB')

        # Convert Image to Bytes for Download
        img_bytes = io.BytesIO()
        enhanced_pil.save(img_bytes, format="JPEG")

        # Show Before → After Comparison
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Before", use_container_width=True)

        with col2:
            st.image(enhanced_image, caption="After", use_container_width=True)

        # Download Button for Enhanced Image
        st.download_button(
            label=f"📥 Download Enhanced {uploaded_img.name}",
            data=img_bytes.getvalue(),
            file_name=f"enhanced_{uploaded_img.name}",
            mime="image/jpeg"
        )
