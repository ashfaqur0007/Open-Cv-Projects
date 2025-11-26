import cv2
import os
import mediapipe as mp
import argparse

def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box

            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)
            img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w],(30, 30))

    return img

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="webcam")
parser.add_argument("--filepath", default=None)
args = parser.parse_args()

out_dir = "./output"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5) as face_detection:

    if args.mode == "image":
        if args.filepath is None:
            raise ValueError("You must provide --filepath for image mode")

        img = cv2.imread(args.filepath)
        if img is None:
            raise ValueError("Invalid image path!")

        img = process_img(img, face_detection)
        cv2.imwrite(os.path.join(out_dir, "output.png"), img)

    elif args.mode == "video":
        if args.filepath is None:
            raise ValueError("You must provide --filepath for video mode")

        cap = cv2.VideoCapture(args.filepath)
        if not cap.isOpened():
            raise ValueError("Cannot open video file!")

        ret, frame = cap.read()
        output_video = cv2.VideoWriter(
            os.path.join(out_dir, "video_out.mp4"),
            cv2.VideoWriter_fourcc(*"MP4V"),
            25,
            (frame.shape[1], frame.shape[0])
        )

        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
    elif args.mode == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not access webcam!")

        ret, frame = cap.read()

        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow("frame", frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()
