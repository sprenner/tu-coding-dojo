# Social Media Message Classification

Text classification is the process of assigning tags or categories to text according to its content. It’s one of the fundamental tasks in Natural Language Processing (NLP) with broad applications such as sentiment analysis, topic labeling, spam detection, and intent detection.

Unstructured data in the form of text is everywhere: emails, chats, web pages, social media, support tickets, survey responses, and more. Text can be an extremely rich source of information, but extracting insights from it can be hard and time-consuming due to its unstructured nature. 
Businesses are turning to text classification for structuring text in a fast and cost-efficient way to enhance decision-making and automate processes. But message classification is also a perfect tool for surveillance initiatives like [PRISM](https://de.wikipedia.org/wiki/PRISM) or [CMS](https://en.wikipedia.org/wiki/Central_Monitoring_System).
This is a short example of message classification/sentiment analysis based on social media messages.

# Project Structure
The project is divided into two parts:
1. Preprocessing Pipeline
Load and clean the messages dataset and store the resulting records into a .csv file so that we can use it in the next step to train a supervised model.
2. ML Pipeline
The machine learning pipeline is concerned with training the model and evaluating it. It includes a text processing part because the algorithm must be able to deal with text input.


Sentiment analysis is predestined to monitor voter opinions during political events, a famous example of this is [Brexit](https://brexit.foraction.gr/).

# Problem Formulation
Given a new incoming message, we want to assign it to one of 3 categories: positive, neutral or negative. The classifier makes the assumption that each new message is assigned to one and only one category. This is called a multi-class text classification problem. 

# Data Exploration and Preprocessing
Before diving into training machine learning models, you should look at the available data first.
* Open the jupyter notebook preprocessing.ipynb.
* Use pandas to load the prepared dataset into a dataframe, then explore the available columns and inspect a couple of examples.
* Use matplotlib or seaborn to plot the  the number of records in each class.
* Think about possible data cleaning steps to reduce the number of artifacts or inherent bias.
* You might also want to visualize word frequencies in your training data before and after cleaning (use nltk, matplotlib or seaborn).
* Finally, save the preprocessed dataset. It should include just 2 columns: `text` and `label`.

# Text Representation and ML Pipeline
Open the second jupyter notebook training_pipeline.ipynb to work with the preprocessed dataset.

* Use stopwords to remove less-meaningful words (we have prepared a list of German stop words for you). The logic of removing stop words has to do with the fact that these words don't carry a lot of meaning, and they appear a lot in most text
* Split the dataset into a train set and a test set (e.g.: 80/20, 90/10 ratio)
* The classifiers and learning algorithms can not directly process the text documents in their original form, as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length. Therefore, during the preprocessing step, the texts are converted to a more manageable representation.
* One common approach for extracting features from text is to use the bag of words model: a model where for each document, the presence (and often the frequency) of words is taken into consideration, but the order in which they occur is ignored.

* Create a pipeline consisting of 3 steps (use [scikit-learn](https://scikit-learn.org/stable/documentation.html) for this):
  1. CountVectorizer
  2. TfidfTransformer
  3. RandomForestClassifier

* Evaluate your model regarding accuracy, recall, etc. and plot a confusion matrix.
* Test your model manually using these three example texts (and play around with it a little more if you like):

Example 1:
> Bin rundum begeistert. Insbesondere die Musikwiedergabe ist eine tolle Sache. 
> Die Bedienung lässt keine Wünsche übrig und ist kinderleicht. Beste Grüße, Max

Example 2:
> Startet leider nicht mehr. Ich hatte den Echo leider nur ca. 3 Stunden einwandfrei in Betrieb, danach ging er leider nicht mehr. 
> Kann mir vl jemand sagen wie ich ihn wieder zum Laufen bringe oder ob er einfach nur kaputt ist?

Example 3:
> Gut, aber im Alltag noch nicht ausgereift. Amazon Echo ist eine praktische Sache allerdings 
> wenn man kein Smart-Home hat ist der Einsatzbereich begrenzt.
