import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers import AbstractSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline,T5ForConditionalGeneration, T5Tokenizer
from summarizer.sbert import SBertSummarizer
from summarizer import Summarizer
from nltk.corpus import stopwords 
import wikipedia

st.markdown("<h1 style='text-align: center;'>MiniWiki</h1>", unsafe_allow_html=True)

st.markdown('"MiniWiki" is a mini version of Wikipedia for people who want to just short summary on any wikipedia page instead of reading  all the information provided')
userInput = st.text_input('Topic Searching for : ')
userNumber = st.slider('Number of Sentence you want to summarize it : ', 1,10)
try:
    wikisearch = wikipedia.page("m"+userInput)
except wikipedia.exceptions.DisambiguationError as e:
    st.write("Error: {0}".format(e))

wikicontent = wikisearch.content[:2500]
parser=PlaintextParser.from_string(wikicontent,Tokenizer("english"))

options = st.multiselect(
     'Text Summarization algorithm you want to apply : ',
     ['LexRank','Luhn','KL-Sum','BART-Tranformer','LSA-Summarizer','T5-Transformer','BERT'])

for i in range(len(options)):
	if(options[i] == 'LexRank'):
		st.markdown("<h5>LexRank Algorithm : </h5>", unsafe_allow_html=True)
		summarizer = LexRankSummarizer()
		summary = summarizer(parser.document,userNumber)
		for sentence in summary:
		   	st.markdown(sentence)
	elif(options[i] == 'Luhn'):
		st.markdown("<h5>Luhn Algorithm : </h5>", unsafe_allow_html=True)
		summarizer = LuhnSummarizer()
		summary = summarizer(parser.document,userNumber)
		for sentence in summary:
		    st.markdown(sentence)

	elif(options[i] == 'KL-Sum'):
		st.markdown("<h5>KL-Sum Algorithm : </h5>", unsafe_allow_html=True)
		summarizer = KLSummarizer()
		summary = summarizer(parser.document,userNumber)
		for sentence in summary:
		    st.markdown(sentence)

	elif(options[i] == 'BART-Tranformer'):
		st.markdown("<h5>BART-Transformer : </h5>", unsafe_allow_html=True)
		summarizer = pipeline("summarization")
		summarized = summarizer(wikicontent, min_length=75, max_length=100)
		st.write(summarized[0]['summary_text'])

	elif(options[i] == 'LSA-Summarizer'):
		st.markdown("<h5>LSA-Summarizer : </h5>", unsafe_allow_html=True)
		summarizer_2 = LsaSummarizer()
		summary_2 =summarizer_2(parser.document,userNumber)
		for sentence in summary_2:
		    st.markdown(sentence)

	elif(options[i] == 'T5-Transformer'):
		st.markdown("<h5>T5-Transformer : </h5>", unsafe_allow_html=True)
		model = T5ForConditionalGeneration.from_pretrained("t5-base")
		tokenizer = T5Tokenizer.from_pretrained("t5-base")
		inputs = tokenizer.encode("summarize: " + wikicontent, return_tensors="pt", max_length=512, truncation=True)
		outputs = model.generate(
	    inputs, 
	    max_length=150, 
	    min_length=40, 
	    length_penalty=2.0, 
	    num_beams=4, 
	    early_stopping=True)
		st.markdown(tokenizer.decode(outputs[0]).replace("<pad>","").replace("</s>",""))

	elif(options[i] == 'BERT'):
		st.markdown("<h5>BERT-Transformer : </h5>", unsafe_allow_html=True)
		body = wikicontent
		model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
		result = model(body, num_sentences=userNumber)
		st.markdown(result)




