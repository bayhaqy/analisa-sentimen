# Analisa-Sentimen

This application is used to perform sentiment analysis using VADER and Supervise Vector Machine (SVM) with SMOTE. The dataset used is derived from a tweet by querying "islamophobia since: 2019-03-15 until: 2019-03-16". The aim is to see a sentiment analysis about Islamophobia based on the attack of the mosque in Christcurch, New Zealand. Below are the steps to run it:
1. Download the source or do the command "git clone https://github.com/bayhaqy/analisa-sentimen.git"
2. Run "pip install -r requirements.txt" (Python 2), or "pip3 install -r requirements.txt" (Python 3)
3. After that, run "python app.py" (Python 2), or "python3 app.py" (Python 3)


If you want to create from your own dataset, you can build it with step in the below:
1. Open Sentiment_Analysis.ipynb
2. Import your raw data
3. Running Preprocessing step 1
4. Run to Give a label with VADER
5. Run to select only column tweet and label from VADER
6. Running Preprocessing step 2 with TF-IDF
7. Evaluation and Validation Model
8. Export model with best performance result to directory static/data as model.pkl
9. Copy dataset with label from VADER to directory static/data as Dataset.csv
10. Open application and make sure it's run..


See the paper on:
http://dx.doi.org/10.12928/telkomnika.v18i4.14179
