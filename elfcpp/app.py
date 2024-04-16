from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/model', methods=['POST'])
def call_model():
    data = request.json
    prompt = data['prompt']
    
    # Replace 'your_executable' with the path to your compiled C++ executable
    command = ['./elf', prompt]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return jsonify({"output": result.stdout}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Model execution failed", "details": e.stderr}), 500

if __name__ == '__main__':
    app.run(debug=True)

