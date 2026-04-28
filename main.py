import pathlib

from models import get_model
from services.face_service import FaceService

def get_image_path():
    folder = input("Digite o nome da pasta onde estão as imagens: ")
    img1 = f"test/{folder}/face1.png"
    img2 = f"test/{folder}/face2.png"
    img3 = f"test/{folder}/face3.png"
    return img1, img2, img3

def get_model_name():
    print(f"\n==================================================================\n")

    print("Escolha o modelo de reconhecimento facial:")
    print("1. InsightFace")
    print("2. Dlib")
    print("3. FaceNet")

    print(f"\n==================================================================\n")

    choice = input("Digite o número do modelo desejado: ")

    print (f"\n==================================================================\n")

    if choice == "1":
        return "insightface"
    elif choice == "2":
        return "dlib"
    elif choice == "3":
        return "facenet"
    else:
        raise ValueError("Opção inválida")

def get_number_of_tests():
    try:
        folders = pathlib.Path("test").glob("*")
        return len(list(folders))
    except Exception as e:        
        print(f"Erro ao contar pastas de teste: {e}")
        return 0
if __name__ == "__main__":
    model_name = get_model_name()
    model = get_model(model_name)
    service = FaceService(model)

    try:
        num_tests = get_number_of_tests()
        
        for i in range(num_tests):
            img1 = f"test/{i + 1}/face1.png"
            img2 = f"test/{i + 1}/face2.png"
            img3 = f"test/{i + 1}/face3.png"

            score1, score2, score3 = service.compare_faces(img1, img2, img3)

            print(f"\n================================= Teste {i + 1} ======================================")
            print(f"Modelo usado: {model_name}")
            print(f"face1 vs face2: {score1}/100")
            print(f"face1 vs face3: {score2}/100")
            print(f"face2 vs face3: {score3}/100")
            print("================================================================================\n")

    except Exception as e:
        print(f"Erro: {e}")
