import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

class FaceNetModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Para usar GPU, caso disponível 
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])

    def get_embedding(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(img)

        return emb.cpu().numpy()[0]