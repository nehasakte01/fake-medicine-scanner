import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

st.set_page_config(
    page_title="AI Fake Medicine Scanner",
    page_icon="💊"
)

st.title("💊 AI-Based Fake Medicine Detection")
st.write("Preliminary screening of medicine packaging")

st.header("1. Genuine Reference Image")

reference_file = st.file_uploader(
    "Upload genuine medicine package",
    type=["jpg", "jpeg", "png"]
)

st.header("2. Medicine Image to Check")

test_file = st.file_uploader(
    "Upload medicine package to scan",
    type=["jpg", "jpeg", "png"]
)

if test_file:

    test_image = Image.open(test_file)

    st.image(
        test_image,
        caption="Medicine Package",
        use_container_width=True
    )

    st.header("3. OCR Text Analysis")

    detected_text = pytesseract.image_to_string(test_image)

    if detected_text.strip():
        st.write("Detected text:")
        st.code(detected_text)
    else:
        st.warning("No readable text detected.")

    if reference_file:

        reference_image = Image.open(reference_file)

        image1 = np.array(
            test_image.convert("RGB").resize((256, 256))
        )

        image2 = np.array(
            reference_image.convert("RGB").resize((256, 256))
        )

        image1 = cv2.cvtColor(
            image1, cv2.COLOR_RGB2GRAY
        )

        image2 = cv2.cvtColor(
            image2, cv2.COLOR_RGB2GRAY
        )

        score = cv2.matchTemplate(
            image1,
            image2,
            cv2.TM_CCOEFF_NORMED
        )[0][0]

        similarity = max(
            0,
            min(100, score * 100)
        )

        st.header("4. Screening Result")

        st.metric(
            "Packaging Similarity",
            f"{similarity:.1f}%"
        )

        if similarity >= 85:
            st.success(
                "🟢 Packaging is highly similar to the reference."
            )

        elif similarity >= 70:
            st.warning(
                "🟠 Some packaging differences detected."
            )

        else:
            st.error(
                "🔴 Significant packaging differences detected."
            )

        st.info(
            "This is a prototype screening tool. "
            "It cannot confirm medicine authenticity."
  )
