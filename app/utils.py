import face_recognition
import cv2
import os
import app.CONFIG as CONFIG

# ============================== Config ========================================

IMAGE_FORMAT = ['jpg', 'png', 'jpeg']  # Image Format of your data [jpg / png / jpeg]

# ============================== Config ========================================


def image_resize(img, width=None):
    dim = None
    (h, w) = img.shape[:2]

    if w > h:
        h = int((width * h) / w)
        dim = (width, h)
    else:
        w = int((width * w) / h)
        dim = (w, width)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def learn(image_path):
    image = face_recognition.load_image_file(image_path)
    # loaded_image = face_recognition.load_image_file(image_path)
    return face_recognition.face_encodings(image)[0]


def test(image_path, known_face_encodings, known_people):

    image = cv2.imread(image_path)
    image = image_resize(image, 720)
    name = image_path.split('\\')[-1]
    # finding locations by upsampling the image twice
    face_locations_unknown = face_recognition.face_locations(image,
                                                             number_of_times_to_upsample=2)
    # finding the encodings in the area where the faces have been detected
    unknown_face_encodings = face_recognition.face_encodings(image,
                                                             known_face_locations=face_locations_unknown)
    print(len(face_locations_unknown), len(unknown_face_encodings))
    faces_found = {}

    for unknown_face_encoding, face_location in zip(unknown_face_encodings, face_locations_unknown):
        top, right, bottom, left = face_location
        results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding,
                                                 tolerance=0.6)

        print(len(results))
        if any(results):
            for index, result in enumerate(results):
                if result:
                    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
                    t_size = cv2.getTextSize(known_people[index], cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                    c3 = left + t_size[0] + 3, top + t_size[1] + 4
                    cv2.rectangle(image, (left, top), c3, (255, 0, 0), -1)
                    cv2.putText(image, known_people[index], (left, top + t_size[1] + 4),
                                cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], 2)
                    faces_found["Person_" + str(index)] = known_people[index]

        else:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 3)
            t_size = cv2.getTextSize("Unknown", cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            c3 = left + t_size[0] + 3, top + t_size[1] + 4
            cv2.rectangle(image, (left, top), c3, (0, 0, 255), -1)
            cv2.putText(image, "Unknown", (left, top + t_size[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], 2)

    print(os.path.join(CONFIG.OUTPUT_PATH, name))
    cv2.imwrite(os.path.join(CONFIG.OUTPUT_PATH, name), image)
    return faces_found
