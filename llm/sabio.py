import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def responder_como_sabio(frase_usuario, citas_resultado):
    contexto = ""
    autores = set()

    for doc, meta in zip(citas_resultado["documents"][0], citas_resultado["metadatas"][0]):
        contexto += f"- {doc} ({meta['autor']})\n"
        autores.add(meta['autor'])

    prompt = f"""
Eres un sabio antiguo. Una persona te dice: "{frase_usuario}"
Contéstale usando estas citas filosóficas:

{contexto}

Sé breve, profundo y responde con sabiduría.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un sabio antiguo que responde con sabiduría y brevedad."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7,
        n=1
    )

    respuesta_texto = response.choices[0].message.content.strip()

    return {
        "frase_usuario": frase_usuario,
        "autores": list(autores),
        "respuesta": respuesta_texto
    }
