import datetime
import re
import cv2
import pytesseract
import argparse
import numpy as np
import math
import pprint
from PIL import Image
import deskew
import streamlit as st


@st.cache
def cleanAndConvertDate(date_str):
    if date_str == "":
        return date_str

    date_format = "%y%m%d"
    cleaned_date = date_str.replace("B", "8").replace("O", "0").replace("I", "1")
    return datetime.datetime.strptime(cleaned_date, date_format)


@st.cache
def read_mrz(mrz_text):
    mrz_details = {}
    line_one_pattern = r"P<NGA(\w+)<<(\w+<?\w*)<*"

    # Group 1: Surname, Group 2: Given Name (replace < with " ")
    line_one_matches = re.findall(line_one_pattern, mrz_text, flags=re.MULTILINE)
    if line_one_matches:
        surname, given_name = line_one_matches[0]
        given_name = given_name.replace("<", " ")
        mrz_details["surname"] = surname.title().strip()
        mrz_details["given_name"] = given_name.title().strip()

    line_two_pattern = r"(\w+)\wNGA([OBI\d]{6})..([OBI\d]{6}).(\w+)?"
    line_two_matches = re.findall(line_two_pattern, mrz_text, flags=re.MULTILINE)
    if line_two_matches:
        (
            mrz_details["passport_number"],
            mrz_details["date_of_birth"],
            mrz_details["date_of_expiration"],
            mrz_details["national_id_number"],
        ) = line_two_matches[0]

    if "date_of_birth" in mrz_details:
        mrz_details["date_of_birth"] = cleanAndConvertDate(mrz_details["date_of_birth"])

    if "date_of_expiration" in mrz_details:
        mrz_details["date_of_expiration"] = cleanAndConvertDate(
            mrz_details["date_of_expiration"]
        )

    if (
        "national_id_number" in mrz_details
        and len(mrz_details["national_id_number"]) < 2
    ):
        del mrz_details["national_id_number"]

    return mrz_details


if __name__ == "__main__":
    st.title("Nigerian Passport OCR [Demo]")
    uploaded_file = st.file_uploader("Passport Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        st.markdown("Processing uploaded image")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
    else:
        st.markdown("**No file uploaded.** Using sample image.")
        image = cv2.imread("sample.jpeg", cv2.IMREAD_COLOR)

    info_dict = {}

    image = deskew.deskew(image)
    if image is None:
        st.write("Not a valid image of a Nigerian Passport")
    else:
        imageForFaceDetection = cv2.resize(
            image, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_CUBIC
        )
        image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(
            imageForFaceDetection, minSize=(60, 60), scaleFactor=1.1, minNeighbors=9
        )
        gray = cv2.cvtColor(imageForFaceDetection, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in faces:
            cv2.rectangle(
                imageForFaceDetection,
                (int(x - 0.1 * w), int(y - 0.25 * h)),
                (int(x + 1.1 * w), int(y + 1.25 * h)),
                (255, 0, 0),
                2,
            )
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = imageForFaceDetection[y : y + h, x : x + w]

        st.image(
            imageForFaceDetection,
            caption="Face detection result",
            use_column_width=True,
            channels="BGR",
        )

        if image is not None:
            image = cv2.bilateralFilter(image, 9, 75, 75)
            thresh = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)[1]
            morph_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            thresh = cv2.erode(thresh, morph_struct, anchor=(-1, -1), iterations=1)
            thresh = cv2.dilate(thresh, morph_struct, anchor=(-1, -1), iterations=1)

            custom_config = r"-c tessedit_char_whitelist='ABCDEÉFGHIJKLMNOPQRSTUVWXYZ/-<1234567890 ' --psm 6 --tessdata-dir ."
            text = pytesseract.image_to_string(image, config=custom_config, lang="ocrb")

            if len(text) < 30:
                st.write("MRZ malformed or unreadable.")
            else:
                text_split = text.split()

                mrz = []
                for line in text_split:
                    if len(line) > 20 and "<" in line:
                        mrz.append(line)
                if len(mrz) < 2:
                    st.write("MRZ malformed or unreadable.")
                else:
                    mrz_text = "\n".join(mrz)
                    info_dict = read_mrz(mrz_text)
                    if mrz_text:
                        if mrz_text[0] == "P":
                            info_dict["document_type"] = "Passport"
                        info_dict["mrz_text"] = mrz_text

                # Looks for the passport number in the entire OCR result, in case it can't be read from the MRZ properly
                if (
                    "passport_number" not in info_dict
                    or len(info_dict["passport_number"]) < 8
                ):
                    matches = re.findall(
                        r"([A-C]{1}[0-9]{8})", text, flags=re.MULTILINE
                    )
                    if matches:
                        info_dict["passport_number"] = matches[0]

        if info_dict:
            st.subheader("OCR Results")
            for key, val in info_dict.items():
                if type(val) is datetime.datetime:
                    st.markdown("**" + key + "**: " + val.strftime("%dth %B %Y"))
                else:
                    st.markdown("**" + key + "**: " + val)
        else:
            st.subheader("Error: Not a valid image of a Nigerian Passport")
