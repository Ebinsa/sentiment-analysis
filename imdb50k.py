#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
import graphviz
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns


# ### Data preprocessing from first 10,000 set of data

# In[2]:


pdata = pd.read_excel("/Users/ebinsam/Downloads/IMDB-Dataset.xlsx")


# In[3]:


pdata.to_csv("reviews.csv", index=False)


# In[4]:


pdata


# In[5]:


pdata.head(10000)


# In[6]:


# Convert the 'sentiment' column to numerical labels
pdata['sentiment'] = pdata['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)


# In[7]:


# Combine all reviews into a single text
corpus = ' '.join(pdata['review'].astype(str))


# In[8]:


# Remove unwanted characters, numbers, and symbols
corpus = re.sub(r'\d+', '', corpus)
corpus = re.sub(r'[^\w\s]', '', corpus)


# In[9]:


words = corpus.lower().split()


# In[10]:


nltk.download('stopwords')


# In[11]:


from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
print(stop_words)


# In[12]:


stemmer = PorterStemmer()
words = [stemmer.stem(word) for word in words]


# In[13]:


corpus = ' '.join(words)


# In[14]:


corpus


# In[15]:


pdata


# In[35]:


pdata.head(10000)


# In[ ]:


#tfidf_vectorizer = TfidfVectorizer()tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
#tfidf_matrix = tfidf_vectorizer.fit_transform(pdata['review'])


# In[ ]:


#tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
#correlation_matrix = tfidf_df.corr()


# ### Splitting and training the first 10000 datas 

# In[77]:


X = pdata['review']
y = pdata['sentiment']

# Split the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# In[69]:


X_train


# In[70]:


X_test


# In[71]:


y_train


# In[72]:


y_test


# In[73]:


vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# ### Perform  the three best algorithms for the first 10,000 datas

# In[74]:


accuracy_scores = []
classification_reports = []
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Naive Bayes', MultinomialNB()),
    ('Random Forest', RandomForestClassifier())
]


# In[75]:


for name, model in models:
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    classification_reports.append(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# In[78]:


plt.figure(figsize=(8, 6))
plt.bar(range(len(models)), accuracy_scores)
plt.xticks(range(len(models)), [name for name, _ in models])
plt.title('Accuracy Scores')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()


# In[27]:


from graphviz import Source
from sklearn.tree import export_graphviz

if isinstance(model, RandomForestClassifier):
    estimator = model.estimators_[0]
    export_graphviz(estimator, out_file=f'{name}_tree.dot', 
                    class_names=['negative', 'positive'],
                    feature_names=vectorizer.get_feature_names_out(),
                    impurity=False, filled=True, rounded=True)
    
    # Convert DOT file to a Graphviz object
    dot_graph = Source.from_file(f'{name}_tree.dot')
    
    # Display the graph
    dot_graph.view()


# In[79]:


if isinstance(model, RandomForestClassifier):
    importances = model.feature_importances_
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(importances)), importances)
    plt.xticks(range(len(importances)), vectorizer.get_feature_names_out(), rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'{name} Feature Importances')
    plt.show()


# In[81]:


from sklearn.metrics import precision_score, recall_score, f1_score

precisions = []
recalls = []
f1_scores = []

for name, model in models:
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_test_vectorized)
    
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Visualization
plt.figure(figsize=(8, 6))
plt.bar(range(len(models)), [scores[0] for scores in precisions], label='Negative Precision')
plt.bar(range(len(models)), [scores[1] for scores in precisions], label='Positive Precision')
plt.xticks(range(len(models)), [name for name, _ in models])
plt.title('Precision Scores')
plt.xlabel('Model')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Similar plots for recall and F1-score


# In[55]:


from sklearn.model_selection import GridSearchCV

# Example for Logistic Regression
lr_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
lr_grid_search = GridSearchCV(LogisticRegression(), lr_params, cv=5, scoring='accuracy')
lr_grid_search.fit(X_train_vectorized, y_train)

# Visualize the results
plt.figure(figsize=(8, 6))
scores = lr_grid_search.cv_results_['mean_test_score']
plt.plot(lr_params['C'], scores[::2], label='L1 Penalty')
plt.plot(lr_params['C'], scores[1::2], label='L2 Penalty')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy')
plt.title('Hyperparameter Tuning: Logistic Regression')
plt.legend()
plt.show()


# In[58]:


from sklearn.metrics import auc

for name, model in models:
    model.fit(X_train_vectorized, y_train)
    y_pred_proba = model.predict_proba(X_test_vectorized)[:, 1]

    # Convert string labels to numerical labels
    y_test_numeric = np.where(y_test == 'positive', 1, 0)

    # Check if there are any positive samples in the test set
    if np.sum(y_test_numeric) == 0:
        print(f"{name} has no positive samples in the test set.")
        tpr = np.linspace(0, 1, 100)  # Dummy true positive rates
        fpr = np.linspace(0, 1, 100)  # Dummy false positive rates
        roc_auc = 0.5  # AUC for a random classifier
    else:
        # Sort the predicted probabilities in descending order
        sorted_indices = np.argsort(-y_pred_proba)
        y_pred_proba_sorted = y_pred_proba[sorted_indices]
        y_test_numeric_sorted = y_test_numeric[sorted_indices]

        # Calculate the true positive rates and false positive rates
        tpr = np.cumsum(y_test_numeric_sorted) / np.sum(y_test_numeric)
        fpr = np.cumsum(1 - y_test_numeric_sorted) / np.sum(1 - y_test_numeric)

        # Calculate the AUC
        roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


# In[54]:


for name, report in zip([name for name, _ in models], classification_reports):
    print(f'{name} Classification Report:\n', report, '\n')

