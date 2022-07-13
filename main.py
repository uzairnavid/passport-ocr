import datetime
import re
import cv2
import pytesseract
import argparse
import numpy as np
import math
import pprint
from PIL import Image
import streamlit as st
from math import sqrt
import matplotlib.pyplot as plt

def detect_blur_fft(image, size=60, thresh=15, vis=False):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))
        # display the original input image
        (fig, ax) = plt.subplots(
            1,
            2,
        )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size : cY + size, cX - size : cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean <= thresh


def cleanAndConvertDate(date_str):
    if date_str == "":
        return date_str

    date_format = "%y%m%d"
    cleaned_date = date_str.replace("B", "8").replace("O", "0").replace("I", "1")
    return datetime.datetime.strptime(cleaned_date, date_format)

def detectPassport(image):
    img1 = cv2.imread("passport_template.jpeg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    boost_factor = img1.shape[1] / img2.shape[1]
    if boost_factor > 1.0:
        img2 = cv2.resize(img2, None, fx=boost_factor, fy=boost_factor, interpolation=cv2.INTER_CUBIC)
    else:
        boost_factor = 1.0

    # # histogram equalize
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(16,16))
    # img1 = clahe.apply(img1)
    img2 = clahe.apply(img2)

    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        exit(0)
        
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.65*n.distance:
            good.append(m)
            
    MIN_MATCH_COUNT = 50
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    st.image(
            img3,
            caption="Object matching (SIFT) results. Good matches: {0}".format(len(good)),
            use_column_width=True,
            channels="BGR",
        )

    if matchesMask is None:
        return None

    im_dst = cv2.warpPerspective(image, np.linalg.inv(M), (img1.shape)[::-1])
    st.image(
            im_dst,
            caption="Transformed image".format(len(good)),
            use_column_width=True,
            channels="BGR"
        )
    return im_dst

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

    if image is None:
        st.write("Not a valid image of a Nigerian Passport")
    else:
        extractedImage = detectPassport(image)
        if extractedImage is None:
            st.write("Not a valid image of a Nigerian Passport")
        else:
            image = extractedImage
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
