import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Age Detection AI",
    page_icon="🎯",
    layout="centered"
)

# -------------------- UI --------------------
st.markdown("""
    <style>
    .title {text-align: center; font-size: 40px; font-weight: bold; color: #00ffcc;}
    .subtitle {text-align: center; color: #bbbbbb; font-size: 18px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎯 Age Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and detect age instantly</div>', unsafe_allow_html=True)
st.write("")

# -------------------- MODEL PATHS --------------------
face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"

age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"

# -------------------- LOAD MODELS --------------------
@st.cache_resource()
def load_models():
    face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    return face_net, age_net

face_net, age_net = load_models()

# -------------------- AGE LABELS --------------------
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# -------------------- FACE DETECTION --------------------
def detect_faces(net, frame, conf_threshold=0.7):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104, 117, 123), swapRB=False, crop=False
    )

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]

        if conf > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            boxes.append([x1, y1, x2, y2])

    return boxes

# -------------------- UPLOAD --------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(image, use_container_width=True)

    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # -------------------- PROCESS --------------------
    with st.spinner("🔍 Detecting faces & predicting age..."):
        boxes = detect_faces(face_net, frame)
        count = 0

        for box in boxes:
            x1, y1, x2, y2 = box

            # ------------------ FIX 1: Padding ------------------
            padding = 25
            x1_p = max(0, x1 - padding)
            y1_p = max(0, y1 - padding)
            x2_p = min(frame.shape[1], x2 + padding)
            y2_p = min(frame.shape[0], y2 + padding)

            face = frame[y1_p:y2_p, x1_p:x2_p]

            # ------------------ FIX 2: Skip invalid ------------------
            if face.size == 0:
                continue

            if face.shape[0] < 60 or face.shape[1] < 60:
                continue

            # ------------------ FIX 3: Resize ------------------
            face = cv2.resize(face, (227, 227))

            # ------------------ FIX 4: Brightness improve ------------------
            face = cv2.convertScaleAbs(face, alpha=1.3, beta=20)

            # ------------------ FIX 5: Blur noise reduce ------------------
            face = cv2.GaussianBlur(face, (3, 3), 0)

            # ------------------ FIX 6: Blob ------------------
            blob = cv2.dnn.blobFromImage(
                face,
                scalefactor=1.0,
                size=(227, 227),
                mean=(78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )

            age_net.setInput(blob)
            preds = age_net.forward()[0]

            # ------------------ FIX 7: Normalize ------------------
            preds = preds / preds.sum()

            age = age_list[np.argmax(preds)]
            confidence = np.max(preds)

            count += 1

            # ------------------ DRAW ------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{age} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

    # Convert back
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("🎯 Result")
        st.image(frame, use_container_width=True)

    if count == 0:
        st.warning("⚠️ No clear face detected")
    else:
        st.success(f"✅ Detected {count} face(s)")