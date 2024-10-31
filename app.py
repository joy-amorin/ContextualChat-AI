from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

# Configuración de Swagger
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'  # Aquí estará tu archivo de documentación
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "ContextualChat-AI"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    return jsonify({"response": "Hola, estoy listo para responder tus preguntas."})

if __name__ == '__main__':
    app.run(debug=True)
