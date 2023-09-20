#-------------------------------------------------------------------------
# AUTHOR: Chi Le
# FILENAME: search_py
# SPECIFICATION:
# This program performs search engine tasks such as text transformation, indexing, scoring on documents.
# It returns:
# 1. Transformed documents
# 2. Index terms
# 3. tf-idf weights matrix
# 4. Document score for each document
# 5. Hits, Noises, Misses, Rejected, Recall & Precision
# FOR: CS 4250- Assignment #1
# TIME SPENT: 4 hours;
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard arrays

#importing some Python libraries
import csv
import math

documents = []
labels = []

#reading the data in a csv file
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])
            labels.append(row[1])

#Conduct stopword removal.
#--> add your Python code here
stopWords = {'I', 'and', 'She', 'They', 'her', 'their'}

#Tokenization
transformed_documents = [list(doc.split()) for doc in documents]

#Stopping
for doc in transformed_documents:
    doc[:] = [word for word in doc if word not in stopWords]

#Conduct stemming.
#--> add your Python code here
steeming = {
  "cats": "cat",
  "dogs": "dog",
  "loves": "love",
}

for doc in transformed_documents:
    doc[:] = [steeming.get(word, word) for word in doc]

print("After text transformation (tokenizing, stopping, steeming):")
for i, doc in enumerate(transformed_documents, 1):
    print(f'Document {i}: {doc}')

#Identify the index terms.
#--> add your Python code here
terms = []

for document in transformed_documents:
    for word in document:
        if word not in terms:
            terms.append(word)

print(f"\nIndex terms: {terms}")

#Build the tf-idf term weights matrix.
#--> add your Python code here
docMatrix = []
term_count = {}
total_term_count = {}
tf = {}
df = {term: 0 for term in terms}
idf = {term: 0 for term in terms}
tf_idf_weight = {}

#Step 1: Term count
for i, document in enumerate(transformed_documents):
    term_count[i] = {}
    total_term_count[i] =  0
    for term in terms:
        term_count[i][term] = document.count(term)
        total_term_count[i] += term_count[i][term]
#Step 2: Calculate document frequency: df(t, D) = occurrence of t in documents D
for document in term_count.keys():
    for term in terms:
        if(term_count[document][term] > 0):
            df[term] += 1

#Step 3: Calculate inverse document frequency: idf(t, D) = log(|D| / df(t,D))
for document in term_count.keys():
    for term in terms:
        idf[term] = round(math.log10(len(term_count)/df[term]),4)

#Step 4: Calculate term frequency for each document: f(t,d) = count of t in d / number of terms in d
for document in term_count.keys():
    tf[document] = {}
    for term in terms:
        tf[document][term] = round(term_count[document][term] / total_term_count[document],4)

#Step 5: Calculate tf-idf weight: tf-idf(t, d, D) = tf(t, d) * idf(t, D)
for document in tf.keys():
    tf_idf_weight[document] = {}
    for term in terms:
        tf_idf_weight[document][term] = round(tf[document][term] * idf[term],4)

#Step 6: Build and print matrix
for i, document in enumerate(tf_idf_weight, 1):
    row = [f'Document {i}']
    for term in terms:
        row.append(tf_idf_weight[document].get(term, 0.0))
    docMatrix.append(row)

print("\nTf-idf term weights matrix:")
header = " ".join(["{:<15}".format(term) for term in terms])
print("{:<15} {}".format("", header))
for row in docMatrix:
    print("{:<15} {:<15} {:<15} {:<15}".format(*row))

#Calculate the document scores (ranking) using document weigths (tf-idf) calculated before and query weights (binary - have or not the term).
#--> add your Python code here
docScores = []
query_weight = {term: 0 for term in terms}

#With provided query q = "cat and dogs"
q = "cat and dogs"

#Step 1: Query transformation
transformed_q = q.split()
transformed_q = [word for word in transformed_q if word not in stopWords]
transformed_q = [steeming.get(word, word) for word in transformed_q]

#Step 2: Calculate query weight of each term
for query_term in transformed_q:
    for term in terms:
        if query_term == term:
            query_weight[term] = 1

#Step 3: Calculate document store of each document
for document in tf_idf_weight.keys():
    document_score = 0
    for term, weight in query_weight.items():
        # If the term exists in the document's TF-IDF weights, calculate the contribution to the score
        if term in tf_idf_weight[document]:
            document_score += tf_idf_weight[document][term] * weight

    #Append the document score to the docScores list
    docScores.append((document, document_score))

print("\nWith query = \"cat and dogs\":")
for document, score in docScores:
    print(f"Document {document+1} has document score of {score}")
#Calculate and print the precision and recall of the model by considering that the search engine will return all documents with scores >= 0.1.
#--> add your Python code here

#Step 1: Calculate hits, misses, noise, rejected
evaluation = {'hits': 0, 'noises': 0, 'misses': 0, 'rejected': 0}
for document, score in docScores:
    retrieved = 1 if score >= 0.1 else 0
    relevant = 1 if labels[document] == " R" else 0

    if retrieved == 1 and relevant == 1:
        evaluation['hits'] += 1
    elif retrieved == 1 and relevant == 0:
        evaluation['noises'] += 1
    elif retrieved == 0 and relevant == 1:
        evaluation['misses'] += 1
    elif retrieved == 0 and relevant == 0:
        evaluation['rejected'] += 1

print(f'\nHits: {evaluation["hits"]}')
print(f'Noises: {evaluation["noises"]}')
print(f'Misses: {evaluation["misses"]}')
print(f'Rejected: {evaluation["rejected"]}')

#Step 2: Calculate recall and percision
recall = (evaluation["hits"]/(evaluation["hits"]+evaluation["misses"]))*100
precision = (evaluation["hits"]/(evaluation["hits"]+evaluation["noises"]))*100
print(f'\nRecall: {recall:.2f}%')
print(f'Precision: {precision:.2f}%')
