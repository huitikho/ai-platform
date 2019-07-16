# Sentiment Analysis

Sentiment Analysis Model

Sentimemt Analysis

File Sentiment.csv which contains the sentiment information for different tweets is downloaded from Twitter.

The model will divide the file into the testing and training datasets, and try to train the data according to the words / features exits in the tweet. It deploys with Python NLTK Naive Bayes Classifier for training purpose. After going through the training process, it will test the model with the use of the test dataset, and accuracy is calculated based on the True Positive and True Negative rating for this !

Running Example

mlflow run. -e sentiment_analysis_model
