# untuk fungsi regex
import re

# untuk mengambil punctuation data
import string

# untuk analisis dan memanipulasi data
import pandas as pd

# Untuk encoding label
from sklearn.preprocessing import LabelEncoder

# Untuk tokenize 
from nltk.tokenize import word_tokenize

# Untuk stemming, lemmatize dan stopwords
from nltk.stem import PorterStemmer

# Untuk proses TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Untuk proses SMOTE
from imblearn.over_sampling import SMOTE

# Untuk algoritma SVM
from sklearn.svm import LinearSVC

# Untuk export model
import pickle



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
	emotion	= [emot.strip('\n').strip('\r') for emot in open('emotion.txt')]
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
	file = open('stop_tambah.txt')
	stoptambah = file.read()

	# Memecah kata menggunakan word_tokenize
	token_words = word_tokenize(clean_tweet(tweet))
	sentence = []
	for word in token_words:
		# kondisi untuk filter length dan stopwords tambahan
		if word not in stoptambah and len(word) > 1 and len(word) < 25:
			sentence.append(stemmer.stem(word))
	return sentence


# Read Dataset
print("Read Dataset")
file_name = ('Dataset_islamophobia.csv')
df= pd.read_csv(file_name)

# Encode Data Label
print("Encode Label")
le = LabelEncoder().fit(["Positive", "Negative"])
y = le.transform(df['Sentiment'])

# Melakukan proses TF-IDF
print("TF-IDF")
tfidfconverter = TfidfVectorizer(min_df=2, max_df=1.0, ngram_range=(1,3), stop_words=word_tokenize('english'), tokenizer= preprocessing_en_stem)
X_vect = tfidfconverter.fit_transform(df['Tweet']).toarray()
print(X_vect.shape)

# Penggunaan SMOTE
print("SMOTE")
sm_combine = SMOTE(sampling_strategy='minority',random_state=7)
X_vect,y = sm_combine.fit_sample(X_vect,y)

print("algoritma")
# Algoritma SVM
clr = LinearSVC(C=10.0)
clr.fit(X_vect,y)

print("dump")
# Save model
pickle.dump(clr, open('model.pkl','wb'))

print("load")
# Loading model untuk komparasi
model = pickle.load( open('model.pkl','rb'))

print("predict")
tweet = "he is a good boy"
tweet = tfidfconverter.transform([tweet]).toarray()
print(model.predict(tweet))