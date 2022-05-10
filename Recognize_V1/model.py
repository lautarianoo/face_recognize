import os
import sys
import face_recognition

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


if __name__ == "__main__":
    train_model_by_img("Tayler")