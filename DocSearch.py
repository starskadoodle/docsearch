import math
import numpy as np

def build_word_list(documents):
    all_words = set()
    for document in documents:
        words_in_doc = document.split()
        all_words.update(words_in_doc)
    return all_words

def create_index(documents):
    word_index = {}
    for doc_id, document in enumerate(documents, start=1):
        words_in_doc = document.split()
        for word in words_in_doc:
            if word not in word_index:
                word_index[word] = []
            word_index[word].append(doc_id)
    return word_index

def calculate_angle_similarity(query_vector, document_vector):
    dot_product = np.dot(query_vector, document_vector)
    query_magnitude = np.linalg.norm(query_vector)
    doc_magnitude = np.linalg.norm(document_vector)
    cos_sim = dot_product / (query_magnitude * doc_magnitude)
    angle_degrees = math.degrees(math.acos(cos_sim))
    return angle_degrees

def search(documents, user_queries):
    word_list = build_word_list(documents)
    index = create_index(documents)

    print("Total words:", len(word_list))

    word_to_index = {word: idx for idx, word in enumerate(word_list)}

    for query in user_queries:
        print("Search query:", query)
        query_vector = np.zeros(len(word_list))
        query_words = query.split()

        for word in query_words:
            if word in word_to_index:
                query_vector[word_to_index[word]] = 1

        possible_matches = set(range(1, len(documents) + 1))
        for word in query_words:
            if word in index:
                possible_matches.intersection_update(index[word])
            else:
                possible_matches.clear() 
                break

        print("Potential matches:", *possible_matches)

        doc_similarities = {}
        for doc_id in possible_matches:
            doc_text = documents[doc_id - 1]
            doc_vector = np.zeros_like(query_vector)
            for term in doc_text.split():
                if term in word_to_index:
                    doc_vector[word_to_index[term]] += 1

            similarity = calculate_angle_similarity(query_vector, doc_vector)
            doc_similarities[doc_id] = similarity

        ranked_results = sorted(doc_similarities.items(), key=lambda item: item[1])

        for doc_id, similarity in ranked_results:
            print(doc_id, "{:.2f}".format(similarity))
        print()  

# Read data
with open("docs.txt", "r") as docs_file, open("queries.txt", "r") as queries_file:
    documents = docs_file.read().splitlines()
    user_queries = queries_file.read().splitlines()

# Execute search
search(documents, user_queries)
