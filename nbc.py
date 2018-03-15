import pickle
from math import log

class NBC:
    def __init__(self, documents = None):
        """ Initializes NBC and immediately trains the model if documents are passed"""
        self.classes = set() # Available classes
        self.fc = {} # Class frequencies
        self.fw = {} # Word per Class frequencies
        self.pc = {} # Class probabilities
        self.pw = {} # Word per CLass probabilities
        self.num_documents = 0
        self.vocabulary = set()

        if documents:
            self.train(documents)

    def train(self, documents, k = 1):
        """ Requires a dicts with tokenized, normalized and space seperated words and a gold standard class """
        for doc in documents:
            self.num_documents += 1

            # Gather classes
            c = doc['class']
            if c not in self.classes:
                self.classes.add(c)
                self.fc[c] = 1
                self.fw[c] = {}
                for word in self.vocabulary:
                    self.fw[c][word] = k
            else:
                self.fc[c] += 1

            # Gather frequencies for words
            for word in doc['words'].split():

                if word not in self.vocabulary:
                    self.vocabulary.add(word)

                    # New words must be found at least k times for all classes (Laplace smoothing)
                    for klass in self.classes:
                        self.fw[klass][word] = k
                self.fw[c][word] += 1

        # Frequencies gathered
        # Time to do probability to them
        for c in self.classes:
            # In order to calculate the probability for each word
            # on a class basis, we first need to
            # sum number of occurences for all words in vocabulary
            num_words = sum(self.fw[c][w] for w in self.vocabulary)

            self.pw[c] = {}
            for w, freq in self.fw[c].items():
                self.pw[c][w] = log(freq/num_words)

            # Calculate probability for class itself
            self.pc[c] = log(self.fc[c]/self.num_documents)

    def classify(self, document):
        """ Classify the document """
        max_prob = -float('inf')
        max_name = ''
        for c in self.classes:
            prob = self.pc[c]

            for w in document:
                if w in self.vocabulary:
                    prob += self.pw[c][w]
                else:
                    prob += self.handle_unknown_word(c, w)

            if max_prob < prob:
                max_prob = prob
                max_name = c

        return max_name

    def handle_unknown_word(self, c, w):
        """ What should be done for unknown words? """
        return 0

    def confusion_matrix(self, documents, baseline = None):
        """ Builds a confusion matrix for the passed documents, baseline could be set to a class"""
        matrix = {gold:{pred:0 for pred in self.classes} for gold in self.classes}
        for doc in documents:
            pred = self.classify(doc['words']) if not baseline else baseline
            gold = doc['class']
            matrix[gold][pred] += 1
        return matrix

    def accuracy(self, matrix):
        """ Calculates accuracy for given confusion matrix """
        correct = sum(matrix[x][x] for x in self.classes)
        total = sum(matrix[g][p] for g in self.classes for p in self.classes)
        return correct/total if total > 0 else float('NaN')

    def precision(self, matrix, key):
        """ Calculates precision for given key in confusion matrix """
        correct = matrix[key][key]
        total = sum(matrix[x][key] for x in self.classes)
        return correct/total if total > 0 else float('NaN')

    def recall(self, matrix, key):
        """ Calculates recall for given key in confusion matrix """
        correct = matrix[key][key]
        total = sum(matrix[key][x] for x in self.classes)
        return correct/total if total > 0 else float('NaN')

    def f1(self, matrix, key):
        """ Calculates f1, the harmonic mean, for given key in confusion matrix """
        prec = self.precision(matrix, key)
        rec = self.recall(matrix, key)
        total = prec + rec
        return 2 * ( prec * rec)/ total if total > 0 else float('NaN')

    def save(self, filename):
        """ Pickle this model to given filename """
        with open(filename, 'wb') as f:
            pickle.dump(self,f)

    @classmethod
    def load(cls, filename):
        """ Loads a pickled NBC """
        with open(filename, 'rb') as f:
            return pickle.load(f)

if __name__ == '__main__':
    documents = [
        {'class':'A','words':'Hej jag heter knasen'},
        {'class':'B','words':'Nej du är knasen'},
        {'class':'B','words':'Alberta mahogany bordsben'},
        {'class':'A','words':'Sicket bordsben du är'},
        {'class':'A','words':'du är inget bordsben'},
        {'class':'B','words':'Herre jisses kors vilken knasen'}
    ]
    a = NBC(documents)
    print(a.classify('jisses vad du är bordsben'.split()))
