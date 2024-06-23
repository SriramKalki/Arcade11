from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load pre-trained model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text, target_language):
    # Update the tokenizer and model to use the correct target language
    tokenizer.src_lang = 'en'
    model_name = f'Helsinki-NLP/opus-mt-en-{target_language}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

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