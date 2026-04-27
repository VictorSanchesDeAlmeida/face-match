# Face Match

Comparador de similaridade facial em Python usando embeddings com três backends:

- InsightFace
- Dlib (face_recognition)
- FaceNet (facenet-pytorch)

O projeto calcula a similaridade por cosseno entre vetores faciais e converte o resultado para uma pontuacao de 0 a 100.

## Como Funciona

1. Cada imagem e convertida em embedding pelo modelo selecionado.
2. A similaridade entre dois embeddings e calculada com cosseno.
3. A similaridade e normalizada para score percentual:

$$
score = \frac{sim + 1}{2} \times 100
$$

## Requisitos

- Python 3.10+ (recomendado)
- pip atualizado

Dependencias principais:

- numpy
- opencv-python
- pillow
- insightface
- onnxruntime
- face_recognition
- torch
- torchvision
- facenet-pytorch

## Instalacao

1. Crie e ative um ambiente virtual.
2. Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Configuracao

Selecione o modelo em `config.py` pela variavel `MODEL_NAME`:

- `insightface`
- `dlib`
- `facenet`

Exemplo:

```python
MODEL_NAME = "insightface"
```

## Execucao

Rode o script principal:

```bash
python main.py
```

O fluxo atual executa uma bateria de 9 testes e imprime, para cada caso:

- modelo utilizado
- score de `face1 vs face2`
- score de `face1 vs face3`
- score de `face2 vs face3`

## Estrutura (Arquivos Versionados)

```text
face-match/
	config.py
	main.py
	requirements.txt
	models/
		__init__.py
		dlib_model.py
		facenet_model.py
		insightface_model.py
	services/
		__init__.py
		face_service.py
	utils/
		__init__.py
		similarity.py
```

## Observacoes

- Se nenhum rosto for detectado em uma imagem, o processo gera erro com a mensagem "Nenhum rosto detectado".
- O backend InsightFace roda em CPU na configuracao atual (`ctx_id=-1`).
