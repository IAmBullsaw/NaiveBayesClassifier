# NaiveBayesClassifier
This is one of those.

It can train on data like this:
```python
[
  {'words':['yada', 'yada', 'yada'], 'class':'yada'},
  {'words':['bla', 'bla', 'bla'], 'class':'bla'}
]
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
