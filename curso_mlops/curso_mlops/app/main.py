from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import os
from dotenv import load_dotenv

from transformers import pipeline
from transformers import MBartForConditionalGeneration
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast #tradutor modelo BART50

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

colunas = ['tamanho','ano','garagem']
#modelo = pickle.load(open('../../models/modelo.sav','rb'))
modelo = pickle.load(open('curso_mlops\models\modelo.sav','rb'))


app = Flask(__name__)
load_dotenv()

####
#ativar venv: minhavenv/Scripts/activate
####
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
@basic_auth.required
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

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True, host='0.0.0.0')