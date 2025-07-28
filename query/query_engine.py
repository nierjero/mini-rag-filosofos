import chromadb
import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="citas_filosoficas")

def embed_text(text: str) -> list:
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return response['data'][0]['embedding']

def buscar_citas_similares(pregunta: str, k=3):
    embedding = embed_text(pregunta)

    resultados = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    citas = resultados["documents"][0]
    autores = resultados["metadatas"][0]
    scores = resultados["distances"][0]

    contexto = ""
    for i in range(k):
        cita = citas[i]
        autor = autores[i]["autor"]
        contexto += f"- {autor} dijo: \"{cita}\"\n"

    return contexto

def preguntar_a_filosofo(pregunta):
    # Respuesta de ejemplo, reemplaza por tu lógica real
    return "Esta es una respuesta de ejemplo a la pregunta: " + pregunta
