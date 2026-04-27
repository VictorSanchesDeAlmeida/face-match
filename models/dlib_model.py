import face_recognition

class DlibModel:
    def get_embedding(self, image_path):
        img = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) == 0:
            raise ValueError("Nenhum rosto detectado")

        return encodings[0]