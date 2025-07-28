from dotenv import load_dotenv
load_dotenv()

import os
from config import OPENAI_API_KEY
from db.chroma_setup import cargar_citas
from query.query_engine import preguntar_a_filosofo

def main():
    # Cargar citas filosóficas (solo la primera vez)
    cargar_citas()

    while True:
        pregunta = input("🧠 Pregúntale algo al Filósofo (o 'salir'): ")
        if pregunta.lower() == "salir":
            break

        respuesta = preguntar_a_filosofo(pregunta)
        print("\n💬 El sabio responde:\n", respuesta, "\n")

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY no está definido en las variables de entorno.")
    main()

