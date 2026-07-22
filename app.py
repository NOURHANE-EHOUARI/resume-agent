"""
Application web Flask : interface pour uploader un fichier et obtenir
un résumé généré par Llama, exécuté localement via Ollama.

Usage :
    python app.py
Puis ouvrir http://localhost:5000 dans le navigateur.

Prérequis : Ollama doit être installé et lancé en arrière-plan
(il démarre automatiquement après installation), avec le modèle
llama3.2:3b déjà téléchargé (ollama pull llama3.2:3b).
"""

import os
import tempfile

import ollama
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

OLLAMA_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "Tu es un outil de résumé automatique strict. "
    "Réponds UNIQUEMENT avec le résumé, en français, sans introduction ni commentaire. "
    "RÈGLE ABSOLUE : n'utilise QUE les informations explicitement présentes dans le texte fourni. "
    "N'invente JAMAIS de détails, de lieux, de dates, d'établissements ou de faits qui n'y figurent pas. "
    "Si le texte est une liste ou un document structuré (CV, fiche, tableau...), "
    "résume-le fidèlement sous forme de synthèse factuelle, sans transformer les éléments "
    "en une histoire ou un récit inventé. "
    "Ne te fixe aucune longueur prédéfinie : garde uniquement l'idée générale et les points "
    "réellement importants, et sois aussi concis que le permet le contenu. "
    "N'ajoute aucun détail secondaire ou superflu juste pour allonger le résumé. "
    "Il est impératif de toujours terminer ton résumé complètement : "
    "ne t'arrête jamais au milieu d'une phrase, d'un mot ou d'une section. "
    "Si le document a plusieurs parties, assure-toi de conclure sur toutes, "
    "même brièvement, plutôt que de laisser le résumé incomplet."
)


def summarize_with_llama(text):
    word_count = len(text.split())
    max_tokens = min(int(word_count * 1.5) + 150, 1200)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        options={
            "temperature": 0.2,
            "num_predict": max_tokens,
        },
    )
    return response["message"]["content"].strip()


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

        summary = summarize_with_llama(text)

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)