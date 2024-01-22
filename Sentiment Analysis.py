import pandas as pd

#File paths
raw_fname = 'reviews.csv'
train_fname = 'train.csv'
valid_fname = 'valid.csv'

data = pd.read_csv(raw_fname, delimiter='\t')

#Let's get rid of the Name and Datepublsihed columns
cleaned_df = data.drop(["Name", "DatePublished"], axis=1)

#Feature engineer sentiment grouping
def into_sentiment_group(rating):
    if rating in [1,2]:
        return 0
    elif rating == 3:
        return 1
    elif rating in [4,5]:
        return 2
    else:
        return 'OutOfRange'

cleaned_df['Sentiment'] = cleaned_df['RatingValue'].apply(into_sentiment_group)

#Great, now let's get rid of the RatingValue column
cleaned_df = cleaned_df.drop('RatingValue', axis=1)

#Now let's count how many observations we have for each Sentiment value
sentiment_counts = cleaned_df['Sentiment'].value_counts()

#We want to have an equal number of all sentiment types so we will reduce the number of observations for the 1st and 2nd most popular
#sentiment type so that their number of observations is equal to that of the least popular sentient type
lowest_count_value = sentiment_counts.idxmin()
highest_count_value = sentiment_counts.idxmax()
#Sort the counts in ascending order
sorted_counts = sentiment_counts.sort_values()
#Find the middle value
middle_index = len(sorted_counts) // 2
middle_value = sorted_counts.index[middle_index]

#To balance the data we will get rid of some the highest occuring sentiment type so it is the same number of observations as the lowest occuring type.
num_to_drop = (sentiment_counts[lowest_count_value] - sentiment_counts[highest_count_value]) * -1
dropped_count = 0

#Loop to drop rows where 'Sentiment' is of the highest occuring type until the desired number is reached
index_to_drop = cleaned_df[cleaned_df['Sentiment'] == highest_count_value].index
for idx in index_to_drop:
    cleaned_df = cleaned_df.drop(idx)
    dropped_count += 1
    if dropped_count == num_to_drop:
        break

#Loop to drop rows where 'Sentiment' is of the middle occuring type until the desired number is reached
num_to_drop_2 = (sentiment_counts[lowest_count_value] - sentiment_counts[middle_value]) * -1
dropped_count_2 = 0

index_to_drop = cleaned_df[cleaned_df['Sentiment'] == middle_value].index
for idx in index_to_drop:
    cleaned_df = cleaned_df.drop(idx)
    dropped_count_2 += 1
    if dropped_count_2 == num_to_drop_2:
        break

# Let's make sure that our data is set up in the same way as mentioned in the assignment
#Reset the index
cleaned_df.reset_index(drop=True, inplace=True)
#Add a number column
cleaned_df['Number'] = range(1, len(cleaned_df) + 1)
#Rearrange columns
cleaned_df = cleaned_df[['Number', 'Sentiment', 'Review']]

#We will store 10% of the data in the train set and the rest in teh validation set

# Initialize counters for each sentiment
count_0, count_1, count_2 = 0, 0, 0

# Initialize empty DataFrames for 'valid' and 'train'
valid = pd.DataFrame()
train = pd.DataFrame()

# Loop through the entire cleaned_df
for index, row in cleaned_df.iterrows():
    sentiment = row['Sentiment']

    # Check conditions and add to 'valid' or 'train' accordingly
    if sentiment == 0 and count_0 < 16:
        valid = pd.concat([valid, row.to_frame().transpose()], ignore_index=True)
        count_0 += 1
    elif sentiment == 0 and count_0 >= 16:
        train = pd.concat([train, row.to_frame().transpose()], ignore_index=True)

    elif sentiment == 1 and count_1 < 16:
        valid = pd.concat([valid, row.to_frame().transpose()], ignore_index=True)
        count_1 += 1
    elif sentiment == 1 and count_1 >= 16:
        train = pd.concat([train, row.to_frame().transpose()], ignore_index=True)

    elif sentiment == 2 and count_2 < 16:
        valid = pd.concat([valid, row.to_frame().transpose()], ignore_index=True)
        count_2 += 1
    elif sentiment == 2 and count_2 >= 16:
        train = pd.concat([train, row.to_frame().transpose()], ignore_index=True)

#Let's create 2 csv files for our test and validation sets
valid.to_csv('valid.csv', index=False)
train.to_csv('train.csv', index=False)

# Let us load in our training and validation data
training_data = pd.read_csv(train_fname)
validation_data = pd.read_csv(valid_fname)

# Let's import everything needed for our models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
docs_test = validation_data["Review"]

# Let's make pipelines for our multinomialnb and sdg classifiers
MultinomialNB_text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

SGDClassifier_text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

from sklearn.model_selection import GridSearchCV
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

# Let's run our multinomialNB model using the best params
MultinomialNB_gs_clf = GridSearchCV(MultinomialNB_text_clf, parameters, cv=5, n_jobs=-1)
MultinomialNB_gs_clf = MultinomialNB_gs_clf.fit(training_data["Review"], training_data["Sentiment"])
MultinomialNB_predicted = MultinomialNB_gs_clf.predict(docs_test)
MultinomialNB_accuracy = np.mean(MultinomialNB_predicted == validation_data["Sentiment"])

# Let's run our SGDClassifier model using the best params
SGDClassifier_gs_clf = GridSearchCV(SGDClassifier_text_clf, parameters, cv=5, n_jobs=-1)
SGDClassifier_gs_clf = SGDClassifier_gs_clf.fit(training_data["Review"], training_data["Sentiment"])
SGDClassifier_predicted = SGDClassifier_gs_clf.predict(docs_test)
SGDClassifier_accuracy = np.mean(SGDClassifier_predicted == validation_data["Sentiment"])

# We will know use the better model to display the performance metrics as that is the model that we
# will want to employ for our usecase. 
print("The accuracy of the multinomicalnb model is " + str(MultinomialNB_accuracy))
print("The accuracy of the SGD classifier model is " + str(SGDClassifier_accuracy))

if (MultinomialNB_accuracy > SGDClassifier_accuracy):
    print("The better model is MultinomialNB")
    better_model = MultinomialNB_predicted
    better_model_accuracy = MultinomialNB_accuracy
elif (MultinomialNB_accuracy < SGDClassifier_accuracy):
    print("The better model is SGDClassifier_accuracy")
    better_model = SGDClassifier_predicted
    better_model_accuracy = SGDClassifier_accuracy

from sklearn import metrics
print(metrics.classification_report(validation_data["Sentiment"], better_model,
    target_names=['0','1','2']))

from sklearn.metrics import f1_score
print("Format from assignment")
print("Accuracy on the test set: " + str(better_model_accuracy))
print("Macro average f1-score on the test set " + str(f1_score(validation_data["Sentiment"], better_model, average='macro')))
print("Class-wise F1 scores:")
print("negative: " + str(f1_score(validation_data["Sentiment"], better_model, labels=[0], average=None)[0]))
print("neutral: " + str(f1_score(validation_data["Sentiment"], better_model, labels=[1], average=None)[0]))
print("positive: " + str(f1_score(validation_data["Sentiment"], better_model, labels=[2], average=None)[0]))

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(validation_data["Sentiment"], better_model)

# Print the confusion matrix with labels
labels = ['Negative', 'Neutral', 'Positive']
print("Confusion Matrix:")
print(pd.DataFrame(conf_matrix, index=labels, columns=labels))


# Calculate normalized confusion matrix
normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Print the normalized confusion matrix with labels
print("Normalized Confusion Matrix:")
print(pd.DataFrame(normalized_conf_matrix, index=labels, columns=labels))