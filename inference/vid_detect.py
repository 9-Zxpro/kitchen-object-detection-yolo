import cv2
import os
import tempfile
import pandas as pd
from ultralytics import YOLO
import ffmpeg
import time

model = YOLO("training/weights/best.pt")

def detect_vid(video_bytes, progress_callback=None):
    iop_path = "iop.mp4"
    final_out_path="output.mp4"
    input_video_path = None
    video_data = None
    cap = None
    out = None
    df_summary = pd.DataFrame(columns=["Class", "Confidence"])
    error_message = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(video_bytes)
            input_video_path = tfile.name
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open input video file at {input_video_path}.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(iop_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise RuntimeError(f"Could not initialize video writer for '{iop_path}' with FOURCC '{fourcc}'. This likely indicates a codec or FFMPEG configuration issue on the server.")

            all_detections = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=0.5, iou=0.6, verbose=False)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                names = results[0].names
                classes = results[0].boxes.cls
                confs = results[0].boxes.conf
                for cid, c in zip(classes, confs):
                    all_detections.append((names[int(cid)], f"{c:.2f}")) 

                frame_count += 1
                if progress_callback:
                    progress_callback(frame_count, total_frames)

            cap.release()
            out.release()
            time.sleep(0.3)
                    
            if os.path.exists(iop_path) and os.path.getsize(iop_path) > 0:
                with open(iop_path, "rb") as f:
                    video_data = f.read()
            else:
                raise RuntimeError(f"Processed video file '{iop_path}' was not created or is empty after processing.")
            
            ffmpeg.input(iop_path).output(final_out_path, vcodec="libx264", preset="medium").overwrite_output().run()

            if os.path.exists(final_out_path) and os.path.getsize(final_out_path) > 0:
                with open(final_out_path, "rb") as f:
                    video_data = f.read()
            else:
                raise RuntimeError(f"Final MP4 video file '{final_out_path}' was not created or is empty after re-encoding. Check ffmpeg output or logs.")
            
            if all_detections:
                df = pd.DataFrame(all_detections, columns=["Class","Confidence"])
                df_summary = df.groupby("Class")\
                            .agg({"Confidence":"max"})\
                            .reset_index()\
                            .sort_values(by="Confidence", ascending=False)
                
    except Exception as e:
        error_message = f"An error occurred during video processing: {e}"
        video_data = None
        df_summary = pd.DataFrame(columns=["Class", "Confidence"])
    finally:
        cv2.destroyAllWindows()
        if input_video_path and os.path.exists(input_video_path):
            os.remove(input_video_path)
        if final_out_path and os.path.exists(final_out_path):
            os.remove(final_out_path)

    return video_data, df_summary, error_message
