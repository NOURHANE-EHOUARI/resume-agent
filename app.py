"""
Application web Flask : interface pour uploader un fichier et obtenir
un résumé généré par le modèle entraîné.

Usage :
    python app.py
Puis ouvrir http://localhost:5000 dans le navigateur.
"""

import os
import tempfile

from flask import Flask, request, jsonify, render_template

from src.generate import load_model, summarize
from src.train import get_device

app = Flask(__name__)

device = get_device()
_model = None
_tokenizer = None


def get_model():
    """Charge le modèle une seule fois, puis le réutilise (mis en cache en mémoire)."""
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load_model(device)
    return _model, _tokenizer


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

        model, tokenizer = get_model()
        summary = summarize(text, model, tokenizer, device)

        return jsonify({
            "summary": summary,
            "original_length": len(text.split()),
            "summary_length": len(summary.split()),
        })
    except FileNotFoundError:
        return jsonify({
            "error": "Aucun modèle entraîné trouvé. Lance d'abord l'entraînement (python -m src.train)."
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)