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

# Function to reduce image size (Compress to ~1MB)
def compress_image(img, target_size_kb=1000):
    quality = 95  # Start with high quality
    while True:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=quality)
        size_kb = len(img_bytes.getvalue()) // 1024  # Convert to KB
        if size_kb <= target_size_kb or quality <= 10:
            break
        quality -= 5  # Reduce quality step by step
    return img_bytes

# Streamlit UI
st.title("🖼️ Multi-Image Enhancement App")
st.write("Upload one or more images to enhance and compare before & after results.")

# Upload Multiple Images
uploaded_imgs = st.file_uploader("📤 Upload Images", type=["jpg", "jpeg", "png", "jfif", "bmp", "tiff", "webp"], accept_multiple_files=True)

if uploaded_imgs:
    for idx, uploaded_img in enumerate(uploaded_imgs):
        st.write(f"##### {uploaded_img.name}")  # Image numbering

        # Load Image
        image = Image.open(uploaded_img)

        # Process Image (Enhance)
        enhanced_image = enhance_image(image)

        # Convert Enhanced Image back to PIL format
        enhanced_pil = Image.fromarray(enhanced_image)

        # Compress Image (~1MB)
        compressed_img_bytes = compress_image(enhanced_pil)

        # Show Before → After Comparison
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Before", use_container_width=True)

        with col2:
            st.image(enhanced_image, caption="After", use_container_width=True)

        # Download Button for Compressed Image
        st.download_button(
            label=f"📥 Download Enhanced {uploaded_img.name}",
            data=compressed_img_bytes.getvalue(),
            file_name=f"enhanced_{uploaded_img.name}.jpg",
            mime="image/jpeg"
        )
