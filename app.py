# Untuk web server dan bootstrap
from flask import Flask, render_template, request, url_for

# untuk menghitung waktu prediksi
import time

# untuk analisis dan memanipulasi data
import pandas as pd

# untuk operasi matematika
import numpy as np

# untk import image
from PIL import Image

# untuk membuat wordcloud
from wordcloud import WordCloud

# untuk membuat plot
import matplotlib.pyplot as plt

# untuk fungsi regex
import re

# untuk mengambil punctuation data
import string

# Untuk translate
from googletrans import Translator

# Untuk tokenize 
from nltk.tokenize import word_tokenize

# Untuk stemming dan stopwords
from nltk.stem import PorterStemmer

# Untuk Analisa Sentmen menggunakan VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Untuk encoding label
from sklearn.preprocessing import LabelEncoder

# Untuk proses TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Untuk algoritma SVM
from sklearn.svm import LinearSVC

# Untuk algoritma Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Untuk dump/load model
import pickle

# Untuk upsampling menggunakan SMOTE
from imblearn.over_sampling import SMOTE
from imblearn import pipeline

# Load Flask
app = Flask(__name__)

# Define fungsi untuk clean tweet
def clean_tweet(tweet):
	# Case folding
	tweet = tweet.lower()

	# Cleansing (Remove URL)
	tweet = re.sub('http\S+|\S+co\S+', ' ', tweet)

	# Cleansing (Remove Mention)
	tweet = re.sub("@[A-Za-z0-9\S]+", "", tweet)

	# Cleansing (Remove Hastag)
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

	# Cleansing (Convert Emoticon)
	emotion	= [emot.strip('\n').strip('\r') for emot in open('static/data/emotion.txt')]
	dic={}
	token = tweet.split()
	for i in emotion:
		(key,val)=i.split('\t')
		dic[str(key)]=val
	tweet = ' '.join(str(dic.get(word, word)) for word in token)

	# Cleansing (Remove Number and Punctuation)
	wrem_list = ('rt')
	exclude = set (string.punctuation)
	rem_list = []
	token = tweet.split()
	for w in token:
		if w not in wrem_list:
			for x in w:
				if x in exclude or x.isdigit():
					x=""
					rem_list.append(x)
				else:
					rem_list.append(x)
			rem_list.append(" ")
	tweet = "".join(rem_list)
	
	# Replace karakter berulang
	def hapus_katadouble(tweet):
		pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
		return pattern.sub(r"\1\1", tweet)  

	tweet=hapus_katadouble(tweet)
	tweet = re.sub('[\s]+|[, ]+', ' ', tweet.strip())
	return tweet

# Define fungsi untuk stemming
def preprocessing_en_stem(tweet):
	stemmer = PorterStemmer()

	# Masukkan stopword tambahan
	file = open('static/data/stop_tambah.txt')
	stoptambah = file.read()

	# Memecah kata menggunakan word_tokenize
	token_words = word_tokenize(clean_tweet(tweet))
	sentence = []
	for word in token_words:
		# kondisi untuk filter length dan stopwords tambahan
		if word not in stoptambah and len(word) > 1 and len(word) < 25:
			sentence.append(stemmer.stem(word))
	return sentence

# Define fungsi untuk translate ke bahasa inggris
def gtrans_tweet_en(tweet):
	translator = Translator()
	translator = translator.translate(tweet,dest='en')
	translator = translator.text
	return translator

# Define fungsi untuk labeling menggunakan VADER
def sentiment_Vader(tweet):
	analysis = SentimentIntensityAnalyzer()
	analysis = analysis.polarity_scores(tweet)
	comm = analysis['compound']
	if (comm >= 0.05):
	    return "Positive"
	elif ((comm > -0.05) and (comm < 0.05)):
	    return "Neutral"
	else:
	    return "Negative"

# Define fungsi word_cloud
def generate_wordcloud(filename, color, words_tem):
    wine_mask = np.array(Image.open("static/images/mosque.png"))
    def transform_format(val):
        if val == 0:
            return 255
        else:
            return val
    transformed_wine_mask = np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)
    for i in range(len(wine_mask)):
        transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))

    word_cloud = WordCloud(colormap=color, width = 512, height = 512, background_color='white', mode="RGBA", mask=transformed_wine_mask).generate_from_frequencies(words_tem)
    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filename, format="png")

# Define fungsi untuk prediksi data
def predict_data(tweet):
	# Ubah tweet menjadi array
	print("transform tweet")
	tweet = tfidfconverter.transform([tweet]).toarray()
	
	print("predict tweet")
	tweet = model.predict(tweet)
	
	print("decode label")
	le = LabelEncoder().fit(["Positive", "Negative"])
	tweet = le.inverse_transform(tweet)
	print("result")
	return tweet

# Define fungsi cek bobot
def cek_bobot(tweet):
	# Ubah tweet menjadi array
	print("transform tweet")
	tweet = tfidfconverter.transform([tweet]).toarray()
	df_tfidf = pd.DataFrame(tweet,columns=tfidfconverter.get_feature_names())
	# Menghitung bobot
	print("menghitung bobot kata")

	words = df_tfidf.sum(axis=0)#.sort_values(ascending=False)
	result = pd.DataFrame({'Count': words}).reset_index()#[:10].to_numpy()
	result = result[result.Count > 0].to_numpy()
	result = ','.join(str(v) for v in result)
	return result


# Read Dataset
print("Read Dataset")
file_name = ('static/data/Dataset.csv')
df= pd.read_csv(file_name)

# Mengambil data tweet
#print("Mengambil data tweet")
#X = df.iloc[:, 0].values

# Encode Data Label dan mengambilnya
print("Encode Data Label")
le = LabelEncoder().fit(["Positive", "Negative"])
y = le.transform(df['VADER'])

# Load the model
print("load model")
model = pickle.load(open('static/data/model.pkl','rb'))
print(model)

# Melakukan proses TF-IDF
print("TF-IDF")
tfidfconverter = TfidfVectorizer(min_df=2, max_df=0.7, ngram_range=(1,3), stop_words=word_tokenize('english'), tokenizer= preprocessing_en_stem)
X_vect = tfidfconverter.fit_transform(df['Clean Tweet']).toarray()
#model = model.fit(X_vect,y)

# Melakukan upsampling menggunakan SMOTE
print("SMOTE")
#sm_combine = SMOTE(sampling_strategy='minority',random_state=10)
#X_vect,y = sm_combine.fit_sample(X_vect,y)

# Mengambil bobot data
print("Get feature name")
df_tfidf = pd.DataFrame(X_vect,columns=tfidfconverter.get_feature_names())
df_tfidf['Sentiment']= y


# Load index page
@app.route('/')
def index():
    return render_template('index.html')

# Load dataset page
@app.route('/dataset')
def dataset():	
	# Cek jumlah positive, negative dan neutral
	positives = df[df['VADER'] == 'Positive']
	negatives = df[df['VADER'] == 'Negative']
	positive = ('Total Label Positive VADER   : {}'.format(len(positives)))
	negative = ('Total Label Negative VADER   : {}'.format(len(negatives)))
	totaldata = ('Total Data                  : {}'.format(df.shape[0]))
	
	print("Count all feature")
	allwords = df_tfidf.drop(['Sentiment'], axis=1).sum(axis=0).sort_values(ascending=False)
	allwords = pd.DataFrame({'Count': allwords}).reset_index()

	dataset = df.to_html(table_id="Table1", classes="display table-bordered table-hover")
	bobotdata = allwords.to_html(table_id="Table2", classes="display table-bordered table-hover")
	
	return render_template('data.html', dataset = dataset, positive = positive, negative = negative, totaldata = totaldata, bobotdata = bobotdata)


# Load generate wordcloud
@app.route('/gen_wordcloud')
def gen_wordcloud():
	print("Count positive feature")
	pos_allwords = df_tfidf[(df_tfidf['Sentiment']==1).values].drop(['Sentiment'], axis=1).sum(axis=0).sort_values(ascending=False)

	print("Count negative feature")
	neg_allwords = df_tfidf[(df_tfidf['Sentiment']==0).values].drop(['Sentiment'], axis=1).sum(axis=0).sort_values(ascending=False)

	print("export wordcloud positive")
	generate_wordcloud('static/images/positive.png','Blues',pos_allwords)

	print("export wordcloud negative")
	generate_wordcloud('static/images/negative.png','Reds',neg_allwords)
	
	return render_template('index.html')


# Load analisa page
@app.route('/analisa', methods=['POST'])
def analisa():
	# Memulai perhitungan waktu
	start = time.time()
	if request.method == 'POST':
		# Mengambil isi text
		rawtext = request.form['rawtext']
		
		# Memanggil fungsi untuk translate
		trans_text = gtrans_tweet_en(rawtext)
		
		# Memanggil fungsi untuk cleansing
		clean_text = clean_tweet(trans_text)
		
		# Mengecek jika mendapat pilihan stem
		prep = preprocessing_en_stem(clean_text)
		
		# Menggabungkan tweet yang telah dipecah untuk ditampilkan
		prolist=[]
		for i in prep:
			prolist.append(i)
			prolist.append(" ")
		prep_text = "".join(prolist)
		
		# Menjalankan analisa sentimen menggunakan vader
		result_vader = (str(sentiment_Vader(clean_text)))
		
		# Menjalankan analisa sentimen menggunakan Support Vector Machines
		result_svm = (str(predict_data(clean_text))[2:-2])

		# Mengecek untuk distribusi frekuensi kata
		#fdist = nltk.FreqDist(prep)
		#freq_data = (str(fdist.most_common(n=20))[1:-1])
		bobot_kata = (str(cek_bobot(clean_text)))

		# Stop menghitung waktu dan menghitungnya
		end = time.time()
		final_time = ("{} detik".format(round((end-start),3)))		
		
	return render_template('index.html', received_text=rawtext, trans_text=trans_text, prep_text = prep_text, clean_text=clean_text, result_vader=result_vader, 
	result_svm=result_svm, final_time=final_time, bobot_kata=bobot_kata)
		

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=88)
