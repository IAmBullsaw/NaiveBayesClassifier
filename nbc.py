import pickle  # Saving a trained model
from math import log  # Using log probabilities
import time  # timing the training.


class NBC:
    def __init__(self, documents=None, k=1, class_k='class', words_k='words'):
        """ Initializes NBC and immediately trains the model if documents are passed"""
        self.classes = set()  # Available classes
        self.fc = {}  # Class frequencies
        self.fw = {}  # Word per Class frequencies
        self.pc = {}  # Class probabilities
        self.pw = {}  # Word per Class probabilities
        self.num_documents = 0
        self.num_tokens = 0
        self.vocabulary = set()

        if documents:
            self.train(documents, k=k, class_k=class_k, words_k=words_k)

    def train(self, documents, k=1, class_k='class', words_k='words'):
        """
        Requires a dicts with tokenized, normalized and space seperated words and a gold standard class
        It's possible to train it again and again and again.
        """
        num_documents_train = len(documents)

        # STATUS PRINTING
        print('Started training on ' + str(num_documents_train) + ' documents...')
        start = time.time()
        printed_75 = False
        printed_50 = False
        printed_25 = False
        docs_left = num_documents_train
        #
        for doc in documents:
            self.num_documents += 1
            # Gather classes
            c = doc[class_k]
            if c not in self.classes:
                self.classes.add(c)
                self.fc[c] = 1
                self.fw[c] = {}
                for word in self.vocabulary:
                    self.fw[c][word] = k
            else:
                self.fc[c] += 1

            # Gather frequencies for words
            for word in doc[words_k]:
                self.num_tokens += 1
                if word not in self.vocabulary:
                    self.vocabulary.add(word)

                    # New words must be found at least k times for all classes (Laplace smoothing)
                    for klass in self.classes:
                        self.fw[klass][word] = k
                self.fw[c][word] += 1

            # Status printing
            docs_left -= 1
            if round(docs_left/num_documents_train * 100) == 75 and not printed_75:
                print('{0:.3f}s\t75% left...'.format(time.time() - start))
                printed_75 = True
            elif round(docs_left/num_documents_train * 100) == 50 and not printed_50:
                print('{0:.3f}s\t50% left...'.format(time.time() - start))
                printed_50 = True
            elif round(docs_left/num_documents_train * 100) == 25 and not printed_25:
                print('{0:.3f}s\t25% left...'.format(time.time() - start))
                printed_25 = True

        print('{0:.3f}s\t0% left...'.format(time.time() - start))
        print('Calculating probabilities...')
        #
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
        print('done.')

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

    def confusion_matrix(self, documents, baseline=None, class_k='class', words_k='words'):
        """ Builds a confusion matrix for the passed documents, baseline could be set to a class"""
        matrix = {gold: {pred: 0 for pred in self.classes} for gold in self.classes}
        for doc in documents:
            pred = self.classify(doc[words_k]) if not baseline else baseline
            gold = doc[class_k]
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
        return 2 * (prec * rec) / total if total > 0 else float('NaN')

    def save(self, filename):
        """ Pickle this model to given filename """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """ Loads a pickled NBC """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def p_matrix(cls, matrix):
        classes = matrix.keys()
        for c in classes:
            print('\t' + c, end='')
        print('')
        for outer_c in classes:
            print(outer_c, end='')
            for inner_c in classes:
                print('\t' + str(matrix[outer_c][inner_c]), end='')
            print('')
