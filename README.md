# NaiveBayesClassifier
This is one of those.

It can train on data like this:
```python
documents = [
  {'words':['yada', 'yada', 'yada'], 'class':'yada'},
  {'words':['bla', 'bla', 'bla'], 'class':'bla'}
]
nbc = NBC(documents)
nbc.save('trainedclassifier.pickle')
```

and classify new documents:
```python
document = ['yada', 'yada', 'bla']
nbc = NBC.load('trainedclassifier.pickle')
result = nbc.classify(document)
```

and give reports on:
* Accuracy
* Precision
* Recall
* F1 measure

and print during training:
```
Started training on 12928 documents...
0.277s  75% left...
0.594s  50% left...
0.931s  25% left...
1.277s  0% left...
Calculating probabilities...
done.
```

and also train again and again and again
```python
nbc = NBC()
for documents in documents_collection:
  nbc.train(documents)
# wait some time ...
```
