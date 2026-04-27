from utils.similarity import cosine_similarity, similarity_to_score

class FaceService:
    def __init__(self, model):
        self.model = model

    def compare_faces(self, img1, img2, img3):
        emb1 = self.model.get_embedding(img1)
        emb2 = self.model.get_embedding(img2)
        emb3 = self.model.get_embedding(img3)

        sim1 = cosine_similarity(emb1, emb2)
        sim2 = cosine_similarity(emb1, emb3)
        sim3 = cosine_similarity(emb2, emb3)

        return (
            similarity_to_score(sim1),
            similarity_to_score(sim2),
            similarity_to_score(sim3)
        )