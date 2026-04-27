import cv2
from insightface.app import FaceAnalysis

class InsightFaceModel:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def get_embedding(self, image_path):
        img = cv2.imread(image_path)
        faces = self.app.get(img)

        if len(faces) == 0:
            raise ValueError("Nenhum rosto detectado")

        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        return face.embedding