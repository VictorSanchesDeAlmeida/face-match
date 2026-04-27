from .insightface_model import InsightFaceModel
from .dlib_model import DlibModel
from .facenet_model import FaceNetModel

def get_model(model_name):
    if model_name == "insightface":
        return InsightFaceModel()
    elif model_name == "dlib":
        return DlibModel()
    elif model_name == "facenet":
        return FaceNetModel()
    else:
        raise ValueError("Modelo inválido")