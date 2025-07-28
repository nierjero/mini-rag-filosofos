from db.chroma_setup import collection
from embeddings.embeddings_openai import embed_text  # Cambiado a OpenAI

def buscar_similares(frase_usuario, k=3):
    embedding = embed_text(frase_usuario)
    resultados = collection.query(query_embeddings=[embedding], n_results=k)
    return resultados
