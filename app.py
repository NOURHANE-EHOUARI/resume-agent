"""
Application web Flask : interface pour uploader un fichier et obtenir
un résumé généré par Llama (hébergé sur Colab, exposé via ngrok).

Usage :
    python app.py
Puis ouvrir http://localhost:5000 dans le navigateur.
"""

import os
import tempfile

import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# URL ngrok de l'API Colab qui héberge Llama.
# À mettre à jour à chaque redémarrage de la session Colab.
COLAB_API_URL = "https://surging-disrupt-levitator.ngrok-free.dev/summarize"


def extract_text_from_file(filepath, filename):
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "txt":
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        import docx
        doc = docx.Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Format de fichier non supporté : .{ext}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize_route():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = extract_text_from_file(tmp_path, file.filename)
        if not text.strip():
            return jsonify({"error": "Impossible d'extraire du texte de ce fichier"}), 400

        response = requests.post(COLAB_API_URL, json={"text": text}, timeout=120)
        response.raise_for_status()
        summary = response.json()["summary"]

        return jsonify({
            "summary": summary,
            "original_length": len(text.split()),
            "summary_length": len(summary.split()),
        })
    except requests.exceptions.RequestException:
        return jsonify({
            "error": "Impossible de contacter le modèle. Vérifie que la session Colab est toujours active."
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)