import streamlit as st
from PIL import Image
import io
import os

from inference.img_detect import detect_img
from inference.vid_detect import detect_vid

st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("YOLOv8 Object Detection")

uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "png", "mp4", "avi"])

img_v = st.empty()
video_v = st.empty()

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

            output_path, df = detect_img(uploaded_file)
            with col2:  
                st.subheader("Detection Results")
                st.image(output_path, caption="Detected Output", use_container_width=True)
        
            st.markdown("Detected Objects")
            st.dataframe(df, use_container_width=True)

    elif file_type.startswith("video"):
        img_v.empty()

        with video_v.container():
            progress_text = st.empty()
            progress_bar = st.progress(0)

        def pcb(frame_count, total_frames):
            progress_percent = min(100, int(frame_count / total_frames * 100))
            progress_bar.progress(progress_percent)
            progress_text.text(f"Processed {frame_count} of {total_frames} frames. {progress_percent}% complete.") 

        video_bytes, df, e_msg = detect_vid(uploaded_file.read(), progress_callback=pcb)
        
        progress_text.empty()
        progress_bar.empty()

        if e_msg:
            st.error(e_msg)
        
        if video_bytes:
            st.subheader("Processed Video")
            st.video(io.BytesIO(video_bytes), format="video/mp4")
            # st.download_button(
            #     label="Download Processed Video",
            #     data=video_bytes,
            #     file_name="detected_output.mp4",
            #     mime="video/mp4"
            # )
        
        else:
            if not e_msg:
                st.warning("Video processing failed or resulted in no output video data.")
            
        st.markdown("Detected Objects")
        if not df.empty: 
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No objects detected or summary not available.")
               
    else:
        img_v.empty()
        video_v.empty()
        st.warning("Unsupported file format.")
