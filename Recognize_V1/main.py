import face_recognition
from PIL import Image, ImageDraw
import pickle, cv2

def face_rec():
    face = face_recognition.load_image_file("../data/510.jpg")
    face_location = face_recognition.face_locations(face)

    pil_img = Image.fromarray(face)
    draw_img = ImageDraw.Draw(pil_img)
    for (top, right, bottom, left) in face_location:
        draw_img.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=5)
    del draw_img

    pil_img.save("../data/new_510.jpg")

def extracting_face(filepath):
    faces = face_recognition.load_image_file(filepath)
    face_locations = face_recognition.face_locations(faces)

    for id, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_img = faces[top:bottom, left:right]

        pil_img = Image.fromarray(face_img)
        pil_img.save(f"../data/{id}_510.jpg")

    return "Successful"

def compare_faces(filepath1, filepath2):
    face_1 = face_recognition.load_image_file(filepath1)
    face_2 = face_recognition.load_image_file(filepath2)
    face_1_encoding = face_recognition.face_encodings(face_1)[0]
    face_2_encoding = face_recognition.face_encodings(face_2)[0]

    result = face_recognition.compare_faces([face_1_encoding], face_2_encoding)
    return result


def detect_person_in_video(name_dataset, video_path):
    data = pickle.loads(open(f'{name_dataset}', 'rb').read())
    video = cv2.VideoCapture(video_path)

    while True:
        ret, image = video.read()
        locations = face_recognition.face_locations(image, model="cnn")
        encodings = face_recognition.face_encodings(image, locations)
        for encoding, location in zip(encodings, locations):
            result = face_recognition.compare_faces(data["encodings"], encoding)
            match = None

            if True in result:
                match = data["name"]
                print(f"Match found! {match}")
            else:
                print('Alarm!')
            left_top = (location[3], location[0])
            right_bottom = (location[1], location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, left_top, right_bottom, color)

        cv2.imshow("running", image)
        k = cv2.waitKey(20)
        if k == ord("q"):
            break


def main():
    extracting_face("../data/510.jpg")

if __name__ == "__main__":
    main()