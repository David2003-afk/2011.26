import uuid

import pandas as pd
import requests
import streamlit as st
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

st.set_page_config(page_title="Analizador de reseñas bancarias", page_icon="🏦", layout="centered")

ETIQUETAS = {"positive": "Positivo", "neutral": "Neutral", "negative": "Negativo", "mixed": "Mixto"}

EJEMPLOS = {
    "Escribir mi propio texto": "",
    "Reclamo": "La app del banco se cae cada vez que intento transferir y nadie responde mis reclamos. Muy decepcionado.",
    "Elogio": "Excelente atención en la agencia, el asesor fue muy amable y resolvió mi consulta rápido.",
    "Mixto": "La atención fue excelente y el personal muy amable. Sin embargo, las comisiones son abusivas y la app es lenta.",
}


@st.cache_resource
def get_language_client():
    return TextAnalyticsClient(
        endpoint=st.secrets["LANGUAGE_ENDPOINT"],
        credential=AzureKeyCredential(st.secrets["LANGUAGE_KEY"]),
    )


def analizar_sentimiento(texto: str):
    r = get_language_client().analyze_sentiment([texto], language="es", show_opinion_mining=True)[0]
    if r.is_error:
        raise RuntimeError(r.error.message)
    return r


def traducir(texto: str) -> str:
    resp = requests.post(
        "https://api.cognitive.microsofttranslator.com/translate",
        params={"api-version": "3.0", "from": "es", "to": ["en"]},
        headers={
            "Ocp-Apim-Subscription-Key": st.secrets["TRANSLATOR_KEY"],
            "Ocp-Apim-Subscription-Region": st.secrets["TRANSLATOR_REGION"],
            "Content-Type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4()),
        },
        json=[{"text": texto}],
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()[0]["translations"][0]["text"]


st.title("🏦 Analizador de reseñas bancarias")
st.caption("Azure AI Language (sentimiento y aspectos) + Azure Translator (español → inglés)")

ejemplo = st.selectbox("Ejemplo", list(EJEMPLOS.keys()))
texto = st.text_area(
    "Comentario del cliente (español)",
    value=EJEMPLOS[ejemplo],
    height=140,
    max_chars=1000,
    key=f"texto_{ejemplo}",
)

if st.button("Analizar", type="primary"):
    if not texto.strip():
        st.warning("Escribe un comentario primero.")
        st.stop()

    try:
        with st.spinner("Consultando Azure..."):
            r = analizar_sentimiento(texto)
            traduccion = traducir(texto)
    except Exception as e:
        st.error(f"Error al llamar a Azure: {e}")
        st.stop()

    st.subheader("1. Sentimiento (Azure AI Language)")
    c = r.confidence_scores
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("General", ETIQUETAS.get(r.sentiment, r.sentiment))
    col2.metric("Positivo", f"{c.positive:.0%}")
    col3.metric("Neutral", f"{c.neutral:.0%}")
    col4.metric("Negativo", f"{c.negative:.0%}")

    filas = []
    for oracion in r.sentences:
        for op in oracion.mined_opinions:
            filas.append(
                {
                    "Aspecto": op.target.text,
                    "Sentimiento": ETIQUETAS.get(op.target.sentiment, op.target.sentiment),
                    "Opiniones": ", ".join(a.text for a in op.assessments),
                }
            )
    st.markdown("**Aspectos detectados**")
    if filas:
        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
    else:
        st.caption("No se detectaron aspectos específicos en este texto.")

    st.subheader("2. Traducción al inglés (Azure Translator)")
    st.info(traduccion)
