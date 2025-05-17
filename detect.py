import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("counting coins from image (not very much accurate)")
st.write("Upload an image and the app will detect and count circles.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image and convert to array
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    output_img = img_array.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Optional: improve contrast
    gray = cv2.equalizeHist(gray)

    # Detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.02,
        minDist=251,
        param1=85,
        param2=150,        # Lowered threshold to detect more circles
        minRadius=80,     # Adjusted for large circle
        maxRadius=380     # Allow larger circles
    )

    # Show original image
    st.image(image, caption="Original Image", use_column_width=True)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_count = len(circles[0, :])

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Draw outer circle
            cv2.circle(output_img, center, radius, (0, 255, 0), 2)
            # Draw center of circle
            cv2.circle(output_img, center, 3, (0, 0, 255), -1)

        # Convert for Streamlit display
        output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        st.image(output_img_rgb, caption=f"Detected {circle_count} Circle(s)", use_column_width=True)
        st.success(f"🟢 Total Circles Detected: {circle_count}")
    else:
        st.warning("⚠️ No circles detected. Try improving contrast or adjusting radius settings.")

