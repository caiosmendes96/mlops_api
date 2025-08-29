from flask import Flask
from textblob import TextBlob
from transformers import pipeline
from transformers import MBartForConditionalGeneration
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast #tradutor modelo BART50

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

app = Flask(__name__)

@app.route('/')
def home():
    return "Minha primeira API2."


@app.route('/sentimento/<frase>')
def sentimento(frase):  
    
    tokenizer.src_lang = "pt_XX"
    inputs = tokenizer(frase, return_tensors="pt")
    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    frase_en = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    tb = TextBlob(frase_en)
    #return "teste"
    if tb.sentiment.polarity > 0.1:
        return 'Positivo'
    elif tb.sentiment.polarity == 0:
        return 'Neutro'
    else:
        return 'Negativo'

    
app.run(debug=True)

#para executar, abrir cmd e executar "python main.py"