import chromadb
from chromadb.api.types import EmbeddingFunction
from embeddings.embeddings_openai import embed_text  # Cambiamos a la nueva función OpenAI

class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts):
        return [embed_text(t) for t in texts]

client = chromadb.Client()
embedding_fn = OpenAIEmbeddingFunction()
collection = client.get_or_create_collection(name="filosofia", embedding_function=embedding_fn)

# Citas (idem a antes, no cambia)

citas = [
    {"autor": "Platón", "texto": "El alma del hombre es inmortal y eterna."},
    {"autor": "Platón", "texto": "El conocimiento verdadero viene del interior."},
    {"autor": "Platón", "texto": "La filosofía es una meditación sobre la muerte."},
    {"autor": "Platón", "texto": "La mayor declaración de amor es la que no se hace."},
    {"autor": "Platón", "texto": "El cuerpo es la cárcel del alma."},
    {"autor": "Platón", "texto": "La educación es el encendido de una llama, no el llenado de un recipiente."},
    {"autor": "Platón", "texto": "La belleza del cuerpo es un reflejo del alma."},
    {"autor": "Platón", "texto": "La opinión sin conocimiento es el mayor de los males."},
    
    {"autor": "Aristóteles", "texto": "La felicidad depende de nosotros mismos."},
    {"autor": "Aristóteles", "texto": "El todo es más que la suma de sus partes."},
    {"autor": "Aristóteles", "texto": "El hombre es un animal político."},
    {"autor": "Aristóteles", "texto": "Somos lo que hacemos repetidamente. La excelencia, entonces, no es un acto, sino un hábito."},
    {"autor": "Aristóteles", "texto": "La amistad es un alma que habita en dos cuerpos."},
    {"autor": "Aristóteles", "texto": "La inteligencia consiste no solo en el conocimiento, sino también en la destreza de aplicar los conocimientos en la práctica."},
    {"autor": "Aristóteles", "texto": "El fin último del hombre es la felicidad."},
    {"autor": "Aristóteles", "texto": "El sabio no dice todo lo que piensa, pero siempre piensa todo lo que dice."},

    {"autor": "Nietzsche", "texto": "Dios ha muerto."},
    {"autor": "Nietzsche", "texto": "El hombre es una cuerda tendida entre el animal y el superhombre."},
    {"autor": "Nietzsche", "texto": "Lo que no me mata me hace más fuerte."},
    {"autor": "Nietzsche", "texto": "Sin música, la vida sería un error."},
    {"autor": "Nietzsche", "texto": "La fe no mueve montañas, pero arrastra muchas cabezas."},
    {"autor": "Nietzsche", "texto": "La libertad es el derecho a decir no."},
    {"autor": "Nietzsche", "texto": "La moral es la mejor de todas las mentiras."},
    {"autor": "Nietzsche", "texto": "El individuo ha luchado siempre para no ser absorbido por la tribu."},

    {"autor": "Kant", "texto": "El cielo estrellado sobre mí y la ley moral en mí."},
    {"autor": "Kant", "texto": "Actúa de tal manera que trates a la humanidad siempre como un fin y nunca como un medio."},
    {"autor": "Kant", "texto": "La libertad es la autonomía de la voluntad."},
    {"autor": "Kant", "texto": "La ilustración es la salida del hombre de su minoría de edad autoimpuesta."},
    {"autor": "Kant", "texto": "La razón pura es práctica por sí misma."},
    {"autor": "Kant", "texto": "La moralidad no es la doctrina de cómo hacernos felices, sino de cómo debemos ser dignos de la felicidad."},
    {"autor": "Kant", "texto": "La experiencia sin teoría es ciega, pero la teoría sin experiencia es mero juego intelectual."}
]

def cargar_citas():
    for idx, cita in enumerate(citas):
        collection.add(
            documents=[cita["texto"]],
            metadatas=[{"autor": cita["autor"]}],
            ids=[f"cita_{idx}"]
        )
