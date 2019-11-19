
'''
Introduction to Python Programming (aka Programmierkurs I, aka Python I)
Software Assignment
'''

import sys
import string
import math
from stemming import porter2
from collections import Counter, defaultdict

def tokenise(sentence):
    "Takes a string as input and returns a list of stemmed, punctuation removed tokens"
    tokens = sentence.lower().split()
    
    for token in tokens:
        tokens[tokens.index(token)] = token.translate(str.maketrans('', '', string.punctuation))
        if token in string.whitespace:
            tokens.remove(token)
    
    tokens = [porter2.stem(x) for x in tokens]
    while '' in tokens:
        tokens.remove('')
    return tokens
    
def dot(query, doc, terms):
    "A helper function to get the dot product of two vectors; the query and the given document. Also takes in a list of all terms in the document collection"
    sum = 0
    for term in terms:
        if query[term] == 0 or doc[term] == 0:
            continue
        sum += query[term] * doc[term]
    return sum

def norm(vec):
    "Gets the norm of a given vector"
    sum = 0
    for term in vec:
        sum += vec[term] ** 2
    return math.sqrt(sum)

class SearchEngine:

    def __init__(self, collectionName, create):
        '''
        Initialize the search engine, i.e. create or read in index. If
        create=True, the search index should be created and written to
        files. If create=False, the search index should be read from
        the files. The collectionName points to the filename of the
        document collection (without the .xml at the end). Hence, you
        can read the documents from <collectionName>.xml, and should
        write / read the idf index to / from <collectionName>.idf, and
        the tf index to / from <collectionName>.tf respectively. All
        of these files must reside in the same folder as THIS file. If
        your program does not adhere to this "interface
        specification", we will subtract some points as it will be
        impossible for us to test your program automatically!
        '''
        self.idf_scores = {}
        self.tf_scores = {}
        self.tf_idf_scores = {}
        self.docs_containing_term_count = Counter()
        self.doc_count = 0
        self.word_vectors = {}
        self.doc_ids = []
        self.terms = []

        if create == True:
            with open(collectionName + '.xml') as f:
                line = f.readline()
                f.readline()
                
                while True:
                    word_counts = Counter()

                    # Scanning forward until the start of the documents
                    while '<DOC id=' not in line:
                        line = f.readline()

                    # Making the assumption that the id is the same length every time
                    doc_id = line[9:30]
                    
                    self.doc_ids.append(doc_id)
                    line = f.readline()

                    if '<HEADLINE>' in line:
                        line = f.readline()
                        words = tokenise(line)
                        word_counts.update(words)

                    while '<TEXT>' not in line:
                        line = f.readline()
                    
                    # Skipping over the line with just '<TEXT>'
                    line = f.readline()

                    while '</TEXT>' not in line:
                        if '<P>' in line or '</P>' in line:
                            line = f.readline()
                            continue
                        words = tokenise(line)
                        word_counts.update(words)
                        line = f.readline()
                        
                    # Wrapping a Counter object in list() returns a list of unique elements, so this
                    self.docs_containing_term_count.update(list(word_counts))
                    self.word_vectors[doc_id] = word_counts
                    
                    line = f.readline()
                    line = f.readline()
                    
                    if '<DOC id=' not in line:
                        break     

            self.doc_count = len(self.word_vectors)
            
            # Although it does seem more sensible to do collect all terms while actually processing the text, I tested both methods on the nytsmall
            # corpus, and it is empirically almost a second faster to do it after the fact like this.
            self.terms = []
            for doc in self.doc_ids:
                self.terms.extend(list(self.word_vectors[doc]))
            
            self.terms = sorted(set(self.terms))

            # Calculating tf_scores
            for term in self.terms:
                self.idf_scores[term] = math.log(self.doc_count / self.docs_containing_term_count[term])
                
            # Calculating idf scores & tf.idf scores
            for id in self.doc_ids:
                term_scores = defaultdict(float)
                term_tf_scores = defaultdict(float)

                for term in self.word_vectors[id]:
                    term_tf_scores[term] = self.word_vectors[id][term] / self.word_vectors[id].most_common(1)[0][1]
                    term_scores[term] = term_tf_scores[term] * self.idf_scores[term]

                self.tf_scores[id] = term_tf_scores
                self.tf_idf_scores[id] = term_scores

            # Writing to the index files       
            with open(collectionName + '.idf', mode = 'w') as f:
                for term in self.terms:
                    f.write(term)
                    f.write('\t')
                    f.write(str(self.idf_scores[term]))
                    f.write('\n')

            with open(collectionName + '.tf', mode = 'w') as f:
                for id in self.doc_ids:
                    for term in sorted(self.tf_scores[id]):
                        f.write(id)
                        f.write('\t')
                        f.write(term)
                        f.write('\t')
                        f.write(str(self.tf_scores[id][term]))
                        f.write('\n')
      
        else:
            print("Reading index from file...")
            with open(collectionName + '.idf') as f:
                while True:
                    line = f.readline()
                    if line == '':
                        break
                    self.idf_scores[line.split()[0]] = float(line.split()[1])
            
            with open(collectionName + '.tf') as f:
                while True:
                    line = f.readline()
                    if line == '':
                        break

                    doc_id = line.split()[0]
                    tf_term_scores = {}
                    while doc_id in line:
                        tf_term_scores[line.split()[1]]  = float(line.split()[2])
                        line = f.readline()
                    self.tf_scores[doc_id] = tf_term_scores

            # Collecting some extra stats which are ordinarily calculated in the course of the text processing above. If the index has already been created though, we need to
            # find them manually.
            # Namely, the total number of docs, the number of docs each term appears in and a list of all terms and of document ids
            self.doc_count = len(self.tf_scores)
            self.doc_ids = self.tf_scores.keys()
            self.terms = self.idf_scores.keys()

            # An overly complicated way of getting a dictionary of the number of documents in which each term appears in just one line
            self.docs_containing_term_count = {term : len(list(filter(lambda x: term in x[1], self.tf_scores.items()))) for term in self.terms}

            # Calculating the tf.idf scores themselves
            for id in self.doc_ids:
                term_scores = defaultdict(float)
                for term in self.tf_scores[id]:
                    term_scores[term] = self.tf_scores[id][term] * self.idf_scores[term]
                
                self.tf_idf_scores[id] = term_scores

            print("Done.")

    def executeQuery(self, queryTerms):
        '''
        Input to this function: List of query terms

        Returns the 10 highest ranked documents together with their
        tf.idf-sum scores, sorted score. For instance,

        [('NYT_ENG_19950101.0001', 0.07237004260325626),
         ('NYT_ENG_19950101.0022', 0.013039249597972629), ...]

        May be less than 10 documents if there aren't as many documents
        that contain the terms.
        '''

        query_scores = defaultdict(float)

        query_counts = Counter(queryTerms)

        for word in set(queryTerms):
            if word not in self.terms:
                continue
            else:
                query_scores[word] = math.log(self.doc_count / self.docs_containing_term_count[word]) * (query_counts[word] / max([query_counts[x] for x in queryTerms]))

        similarities = {}

        for doc in self.tf_idf_scores:
            if len(query_scores) == 0:
                similarities[doc] = 0
            else:
                similarities[doc] = dot(query_scores, self.tf_idf_scores[doc], self.terms) / (norm(query_scores) * norm(self.tf_idf_scores[doc]))

        docs = list(filter(lambda x: similarities[x] != 0, self.doc_ids))
        results = [(doc, similarities[doc]) for doc in sorted(docs, key = lambda x: similarities[x], reverse = True)]
        if len(results) > 10:
            return results[:10]
        else:
            return results

    def executeQueryConsole(self):
        '''
        When calling this, the interactive console should be started,
        ask for queries and display the search results, until the user
        simply hits enter.
        '''
        query = input("Please enter query terms, seperated by whitespace:")

        while query != '':
            query_terms = tokenise(query)
            results = self.executeQuery(query_terms)
            if len(results) == 0:
                print("Sorry, I didn't find any documents for this query.")
            else:
                print("I found the following documents:")
                for result in results:
                    print(result[0], '\t', result[1])
            query = input("Please enter query terms, seperated by whitespace:")
    
if __name__ == '__main__':
    '''
    write your code here:
    * load index / start search engine
    * start the loop asking for query terms
    * program should quit if users enters no term and simply hits enter
    '''

    '''
    # Example for how we might test your program:
    # Should also work with nyt199501 !
    searchEngine = SearchEngine("nytsmall", create=False)
    print(searchEngine.executeQuery(['hurricane', 'philadelphia']))
    '''
    searchEngine = SearchEngine('nyt199501', create=False)
    searchEngine.executeQueryConsole()
