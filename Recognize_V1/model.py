import os
import sys
import face_recognition
from cv2 import cv2

def train_model_by_img(name):
    if not os.path.exists("../data"):
        print("[ERROR] Not exists dataset")
        sys.exit()

    known_encodings = []
    images = os.listdir("../data")
    for (i, image) in enumerate(images):
        print(f"[+] load image.. {i + 1}/{len(images)}")
        face_img = face_recognition.load_image_file(f"../data/{image}")
        face_enc = face_recognition.face_encodings(face_img)[0]

        if len(known_encodings) == 0:
            known_encodings.append(face_enc)
        else:
            for item in range(0, len(known_encodings)):
                result = face_recognition.compare_faces([face_enc], known_encodings[item])

                if result[0]:
                    known_encodings.append(face_enc)
                    print("Same person!")
                    break
                else:
                    print("Another person!")
                    break

    data = {
        'name': name,
        "encodings": known_encodings,
    }
    return data

def take_screenshots(video_path):
    if not os.path.exists("../data_video"):
        print("[ERROR] Not exists dataset")
        sys.exit()

    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        multiplier = fps * 3

        if ret:
            frame_id = int(round(cap.get(1)))
            cv2.imshow("frame", frame)
            k = cv2.waitKey(20)
            if frame_id % multiplier == 0:
                cv2.imwrite(f"../data_video/screen_{count}.jpg", frame)
                count += 1
            if k == ord("q"):
                cv2.imwrite(f"../data_video/extra_screen_{count}.jpg", frame)
                count += 1
        else:
            print('[ERROR] Cant get video')
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    train_model_by_img("Tayler")