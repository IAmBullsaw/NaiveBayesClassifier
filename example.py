from nbc import NBC
import json
import sys

def read_data(filename):
    with open(filename) as f:
        return json.load(f)

def main(training, testing):
    documents = read_data(training)
    test_docs = read_data(testing)
    nbc = NBC(documents)
    matrix = nbc.confusion_matrix(documents)
    NBC.p_matrix(matrix)
    print('Accuracy:', nbc.accuracy(matrix))
    matrix = nbc.confusion_matrix(test_docs)
    NBC.p_matrix(matrix)
    print('Accuracy:', nbc.accuracy(matrix))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2])
