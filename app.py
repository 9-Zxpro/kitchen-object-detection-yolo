import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import pandas as pd
import io

# model = YOLO(r"C:\Users\zz-xy\OneDrive\Desktop\object detection\training\weights\best.pt")
# model = YOLO(r"C:\Users\zz-xy\Downloads\best.pt")
YOLO("best.pt")

st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("YOLOv8 Object Detection")

# st.sidebar.header("Detection Settings")
# conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
# iou = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.6, 0.01)

uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

img_v = st.empty()
video_v = st.empty()


def extract_classes(results):
    names = results[0].names
    classes = results[0].boxes.cls
    confidences = results[0].boxes.conf
    class_names = [names[int(cls)] for cls in classes]
    data = {
        "Class": class_names,
        "Confidence": [f"{conf:.2f}" for conf in confidences],
    }
    return pd.DataFrame(data)
    

if uploaded_file:
    file_type = uploaded_file.type
     

    if file_type.startswith("image"):
        video_v.empty()

        with img_v.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploaded Image")
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_container_width=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp.name)
                results = model.predict(source=tmp.name, conf=0.5, iou=0)

                with col2:  
                    st.subheader("Detection Results")
                    results[0].save(filename="output.jpg")
                    st.image("output.jpg", caption="Detected Output", use_container_width=True)
            
                st.markdown("Detected Objects")
                st.dataframe(extract_classes(results), use_container_width=True)

    elif file_type.startswith("video"):
        img_v.empty()

        with video_v.container():
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = "output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            all_detections = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_text = st.empty()
            progress_bar = st.progress(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=0.5, iou=0, verbose=False)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                names = results[0].names
                classes = results[0].boxes.cls
                confs = results[0].boxes.conf
                for cid, c in zip(classes, confs):
                    all_detections.append((names[int(cid)], f"{c:.2f}")) 

                frame_count += 1
                if total_frames > 0:
                    progress_percent = min(100, int(frame_count / total_frames * 100))
                    progress_bar.progress(progress_percent)
                    progress_text.text(f"Processed {frame_count} of {total_frames} frames. {progress_percent}% complete.") 

            cap.release()
            out.release()
            progress_text.empty()
            progress_bar.empty()

            st.subheader("Processed Video")

            with open(output_path, 'rb') as f:
                video_bytes = f.read()
            st.video(io.BytesIO(video_bytes))

            st.markdown("Detected Objects")
            if all_detections:
                df = pd.DataFrame(all_detections, columns=["Class","Confidence"])
                df_summary = df.groupby("Class")\
                               .agg({"Confidence":"max"})\
                               .reset_index()\
                               .sort_values(by="Confidence", ascending=False)
                st.dataframe(df_summary, use_container_width=True)
            else:
                st.write("No objects detected.")

    else:
        img_v.empty()
        video_v.empty()
        st.warning("Unsupported file format.")
