# ai_server.py
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    base_url="https://api.vectorengine.ai/v1",
    api_key="sk-PPf5Hu1S6cbhPhv9A5jXMS03auJGbcL0aZw695izGrNRTehn"
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    messages = data.get('messages')
    model = data.get('model', 'gpt-4.1-nano')
    if not messages:
        return jsonify({'error': 'Missing messages'}), 400
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        content = response.choices[0].message.content
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)