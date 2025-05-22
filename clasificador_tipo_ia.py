# Clasificador de tipo de IA basado en descripciones

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset de ejemplo
frases = [
    "Un asistente de voz que responde preguntas",             # IA débil
    "Un robot que aprende a caminar por ensayo y error",     # Refuerzo
    "Un sistema que analiza correos para detectar spam",      # Supervisado
    "Una IA que aprende sin etiquetas, agrupando datos",      # No supervisado
    "Una IA capaz de razonar como un humano en cualquier tema" # AGI
]

etiquetas = [
    "IA debil",
    "IA refuerzo",
    "IA supervisada",
    "IA no supervisada",
    "IA general"
]

# Vectorización
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(frases)

# Modelo
modelo = MultinomialNB()
modelo.fit(X, etiquetas)

# Prueba
caso = input("Describe un caso de uso de IA: ")
caso_vec = vectorizer.transform([caso])
prediccion = modelo.predict(caso_vec)

print("Tipo de IA predicho:", prediccion[0])
