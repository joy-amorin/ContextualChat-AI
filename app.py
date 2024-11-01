from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint
from langchain_community.document_loaders import DirectoryLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import torch
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

# Carga el modelo de embeddings de Sentence Transformers (ligero)
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Inicializa el índice FAISS para almacenar los embeddings
dimension = 384  # Dimensión de los embeddings
index = faiss.IndexFlatL2(dimension)

# Carga el modelo y tokenizador de DistilGPT-2 (modelo generativo ligero)
gpt_model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # Agregar token de padding
gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_name, output_hidden_states=True,  return_dict_in_generate=True)

class ChatApi():
    def __init__(self):
        self.texts = []
api = ChatApi()

# Endpoint para cargar documentos
@app.route('/upload_documents', methods=['POST'])
def upload_documents():
    try:
        print(f"Buscando documentos en: {os.path.abspath('./documents')}")
        
        # Verifica que la carpeta de documentos exista
        if not os.path.exists('./documents'):
            return jsonify({"error": "El directorio './documents' no existe."}), 404
        
        print(f"Documentos disponibles: {os.listdir('./documents')}")  # Muestra los documentos en el directorio

        loader = DirectoryLoader('./documents')  # Ruta donde están los documentos
        documents = loader.load()
        
        # Verifica que se hayan cargado documentos
        if not documents:
            return jsonify({"error": "No se encontraron documentos."}), 404  # Manejo de caso sin documentos

        api.texts = [doc.page_content for doc in documents]

        print(f"Documentos cargados: {len(api.texts)}")
        if api.texts:
            print(f"Texto de ejemplo de documentos cargados: {api.texts[0][:100]}")

        # Vectorización de documentos
        embeddings = embedding_model.encode(api.texts, convert_to_tensor=True).cpu()  # Asegúrate de que esté en tensor
        print(f"Tamaños de los embeddings: {[embedding.shape for embedding in embeddings]}")

        # Verifica la dimensión de los embeddings
        for embedding in embeddings:
            print(f"Dimensión del embedding: {embedding.shape}")  # Imprime la forma de cada embedding

        # Asegúrate de que la dimensión coincida
        if embeddings.shape[1] != dimension:
            return jsonify({"error": f"La dimensión de los embeddings ({embeddings.shape[1]}) no coincide con la dimensión del índice FAISS ({dimension})."}), 500

        index.add(embeddings.numpy().astype('float32'))  # Agrega los embeddings al índice FAISS
        
        # Verificar el tamaño del índice FAISS después de agregar los embeddings
        print(f"Cantidad de vectores en el índice FAISS: {index.ntotal}")
        if index.ntotal == 0:
            return jsonify({"error": "No se añadieron embeddings al índice FAISS."}), 500

        return jsonify({"message": f"Se cargaron {len(api.texts)} documentos y se añadieron al índice."})
    except Exception as e:
        print(f"Error al cargar documentos: {str(e)}")  # Imprime el error en la consola
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "API de chat con IA y RAG"})

# Endpoint para chat con RAG
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('query')

    if not question:
        return jsonify({"error": "Invalid question provided"}), 400

    # Genera el embedding de la pregunta
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)  # Utiliza SentenceTransformer para el embedding

    # Asegúrate de que question_embedding sea 2D
    question_embedding = question_embedding.reshape(1, -1)  # Cambia a la forma correcta

    print(f"Forma del embedding de la pregunta: {question_embedding.shape}")

    # Busca el embedding en el índice FAISS
    distances, indices = index.search(question_embedding.numpy(), k=1)
    if distances is None or indices is None:
        return jsonify({"error": "Search faled"}), 500
    
    if indices[0][0] != -1:
        document_index = indices[0][0]
        response_text = api.texts[document_index]
        return jsonify({"response": response_text})
    
    #return jsonify({"err0r": "No se encontró documento relevante"}), 400

    # Procesa las distancias e índices
    response = {
        "distances": distances.tolist(),
        "indices": indices.tolist()
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
