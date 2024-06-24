# Update app.py to fix variable scope issue
from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Define a dictionary to map target languages to their respective model names
model_names = {
    'es': 'Helsinki-NLP/opus-mt-en-es',
    'fr': 'Helsinki-NLP/opus-mt-en-fr',
    'de': 'Helsinki-NLP/opus-mt-en-de',
    'zh': 'Helsinki-NLP/opus-mt-en-zh'
}

# Load the models and tokenizers for each target language
models = {lang: MarianMTModel.from_pretrained(model_name) for lang, model_name in model_names.items()}
tokenizers = {lang: MarianTokenizer.from_pretrained(model_name) for lang, model_name in model_names.items()}

def translate_text(text, target_language):
    tokenizer = tokenizers[target_language]
    model = models[target_language]

    inputs = tokenizer.encode(text, return_tensors='pt')
    translated = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    target_language = request.form['target_language']
    if not text:
        flash('No text provided!')
        return redirect(url_for('home'))
    translation = translate_text(text, target_language)
    return render_template('result.html', original_text=text, translated_text=translation, target_language=target_language)

if __name__ == '__main__':
    app.run(debug=True)
