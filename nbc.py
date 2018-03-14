class NBC:
    def __init__(self, documents = None):
        self.classes = set()
        self.data = {}
        if documents:
            self.train(documents)

    def train(self, documents, k = 1):
        for doc in documents:
            c = doc['class']
            if c not in self.classes:
                self.classes.add(c)
                self.data[c] = {}

            for word in doc['words'].split():
                if word not in self.data[c]:
                    self.data[c][word] = k
                else:
                    self.data[c][word] += 1

    def print_data(self):
        for d in self.data.items():
            print(d)

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
    a.print_data()
