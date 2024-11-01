from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint
from langchain_community.document_loaders import DirectoryLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import os

app = Flask(__name__)

# Configuración de Swagger
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'  # archivo de documentación
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "ContextualChat-AI"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Carga el modelo de embeddings de Sentence Transformers
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Inicializa el índice FAISS para almacenar los embeddings
dimension = 384  # Dimensión de los embeddings
index = faiss.IndexFlatL2(dimension)

# Carga el modelo y tokenizador de GPT-2
gpt2_model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)

# Endpoint para cargar documentos
@app.route('/upload_documents', methods=['POST'])
def upload_documents():
    try:
        loader = DirectoryLoader('./documents')  # Cambia la ruta según tu estructura
        documents = loader.load()
        texts = [doc.page_content for doc in documents]

        # Vectorización de documentos
        embeddings = model.encode(texts)
        index.add(embeddings)  # Agrega los embeddings al índice FAISS

        return jsonify({"message": f"Se cargaron {len(texts)} documentos y se añadieron al índice."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500    

@app.route('/')
def home():
    return jsonify({"message": "API de chat con IA y RAG"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    # Validación de entrada
    if not data or 'query' not in data:
        return jsonify({"error": "Por favor, proporciona una consulta válida."}), 400

    user_question = data['query']  # Obtiene la pregunta del usuario

    # Vectoriza la pregunta del usuario para buscar en el índice FAISS
    user_embedding = model.encode([user_question])

    # Busca los k-nearest neighbors en el índice FAISS
    K = 1  # Número de documentos a recuperar
    distances, indices = index.search(user_embedding, K)

    # Recupera el documento más relevante
    if indices.size > 0:
        relevant_doc_index = indices[0][0]  # Obtiene el índice del documento más relevante
        
        # Carga el contenido del documento correspondiente
        try:
            with open(f'./documents/document_{relevant_doc_index}.txt', 'r') as f:  # Cambia la forma en que accedes al documento según tu carga
                relevant_text = f.read()
        except FileNotFoundError:
            relevant_text = "Lo siento, el documento relevante no fue encontrado."
    else:
        relevant_text = "Lo siento, no pude encontrar información relevante."

    # Combina la pregunta del usuario con el texto relevante para el modelo
    context = f"{relevant_text}\n\nPregunta: {user_question}\nRespuesta:"
    inputs_id = tokenizer.encode(context, return_tensors='pt')

    # Generación de respuesta usando GPT-2
    output = gpt2_model.generate(inputs_id, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Ajuste de la respuesta para quitar la pregunta del texto
    response = response.replace(context, '').strip()

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
