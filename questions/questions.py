import nltk
import sys
import os
import string
import numpy as np

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    contents = dict()

    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file)) as f:
                content = f.read()
                contents[file] = content

    return contents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    
    tokens = [token.lower() for token in nltk.word_tokenize(document)
                    if not any(char in string.punctuation for char in token)]

    tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words("english")]

    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    idfs = dict()
    all_words = set()

    total_docs = len(documents.keys())

    for doc in documents.keys():
        for word in documents[doc]:
            all_words.add(word)
    
    # compute the idfs
    for word in all_words:

        docs_containing_word = 0

        for doc in documents.keys():
            if word in documents[doc]:
                docs_containing_word += 1

        idfs[word] = np.log(total_docs / docs_containing_word)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    # rank files
    tf_idf_sum = dict()
    matches_query = dict()

    for file in files:

        tf_idf_sum[file] = 0
        matches_query[file] = False

        for word in query:
            if word in files[file]:

                tf = files[file].count(word)
                tf_idf_sum[file] += tf * idfs[word]
        
        if all((word in file) for word in query):
            matches_query[file] = True

    sorted_files = [k for k in sorted(
        files.keys(),
        key=lambda file: (
            # first key
            matches_query[file],
            # second key
            tf_idf_sum[file]
        ), reverse = True
    )]

    return sorted_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    matching_word_measure = dict()
    query_term_density = {
        s:dict() for s in sentences
    }

    # convert query to lowercase
    query = {word.lower() for word in query}

    for s in sentences:

        matching_word_measure[s] = 0
        query_term_density[s]["added"] = 0
        query_term_density[s]["in_query"] = 0
        query_term_density[s]["not_in_query"] = 0

    for s, words in sentences.items():

        for word in query:
            if word in words:
                matching_word_measure[s] += idfs[word]
        
        for word in words:
            if word in query:
                query_term_density[s]["in_query"] += 1
            else:
                query_term_density[s]["not_in_query"] += 1
    
    for s in sentences.keys():
        total_words = query_term_density[s]["in_query"] + query_term_density[s]["not_in_query"]
        query_term_density[s] = query_term_density[s]["in_query"] / total_words
    
    sorted_sentences = sorted(
        list(sentences.keys()),
        key=lambda s: (matching_word_measure[s], query_term_density[s]),
        reverse = True
    )
    
    return sorted_sentences[:n]


if __name__ == "__main__":
    main()
