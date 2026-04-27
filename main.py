from config import MODEL_NAME
from models import get_model
from services.face_service import FaceService

def get_image_path():
    folder = input("Digite o nome da pasta onde estão as imagens: ")
    img1 = f"test/{folder}/face1.png"
    img2 = f"test/{folder}/face2.png"
    img3 = f"test/{folder}/face3.png"
    return img1, img2, img3

if __name__ == "__main__":
    model = get_model(MODEL_NAME)
    service = FaceService(model)

    try:
        for i in range(9):
            img1 = f"test/{i + 1}/face1.png"
            img2 = f"test/{i + 1}/face2.png"
            img3 = f"test/{i + 1}/face3.png"

            score1, score2, score3 = service.compare_faces(img1, img2, img3)

            print(f"\n================================= Teste {i + 1} =================================")
            print(f"Modelo usado: {MODEL_NAME}")
            print(f"face1 vs face2: {score1}/100")
            print(f"face1 vs face3: {score2}/100")
            print(f"face2 vs face3: {score3}/100")
            print("================================================================================\n")

    except Exception as e:
        print(f"Erro: {e}")
