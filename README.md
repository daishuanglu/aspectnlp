Aspect Detection NLP Toolkit
=============

Aspect Detection NLP toolkit is a Python package that performs various NLP tasks based on aspect detection and aspect based sentiment analysis.

Installation
-------------

As this package requires cpu-pytorch, please install with the find link arguments
```
pip install aspect-x.x.x-py3-none-any.whl -f https://download.pytorch.org/whl/torch_stable.html
```


Usage
--------
To use this package, 
```python
from aspect.aspect_detector import aspectDetector
from aspect.aspect_sentiment import AspectSentimentScorer
from aspect.vae_topic_model import VAETopic
from aspect.fuzzy_matcher import semanticMatcher


corpus=[
    "While there's a decent menu, it shouldn't take ten minutes to get your drinks and 45 for a dessert pizza.",
    'Faster than old EUSE, less crashing, definitely smoother experience.',
    'The product is really good but performance with things like sharing deals with partners is terrible and takes forever.',
    "Mostly still learning the tool. Things tend to change randomly from time to time... so that can be tricky. But its improving steadily!",
    "MSX is improving, but there is a lot more we could do to make MSX work for us and model best practices to our customers for how to use a CRM.",
    "but it's not working now. Showing 'error loading accounts'.",
    "changes in the new version are in the right direction, but performance still feels mediocre at best. \
There are various places where it could prevent page-loading delays. Salesforce is still ahead.",
    "there are many ways that dynamics fails as a crm compared to our competitors. here are a few: \
    data cleanliness is bad, there should only ever be 1 account per customer, and all activities performed on that account by any person,\
     becuase we dont enforce this well enough i often see multiple accounts, and can even create duplicate accounts,\
      for a customer that already exists. this is bad- and leads to other problems with msx that currently exist."
    ]

asp_detector=aspectDetector()
sent_asp=asp_detector.detect(corpus,disp=True)


# Topic modeling based on aspect words
corpus=[i['aspect'] for i in sent_asp]
topic_model=VAETopic(corpus,n_topics=10)
topic_model.fit()
top_words,topics,topic_ids=topic_model.get_top_words_and_topics(disp=True)

# Aspect based on aspect words
analyzer=AspectSentimentScorer(absa='aspect')
for sentId,sample in enumerate(sent_asp):
    content = list(analyzer.generateInputContent(sample))
    sentiment, scores = analyzer.scoreABSA(content)
    sent_asp[sentId]['sentiment']=sentiment
    sent_asp[sentId]['score']=scores

# aspect fuzzy matcher
target_list=['menu','sales','workhub','account','User Interface','search','Note','telephony','appportal','food']
matcher=semanticMatcher()

for content in sent_asp:
    matched_terms = matcher.token_matcher(content['aspect'], target_list, threshold=0.45)
    print('[Text] {};'.format(content['sentence']))
    for term in matched_terms[0]:
        print('- [Feature Mentioned] {:s}; score:{:.4f}'.format(term['label'],term['score']))
    for i,asp in enumerate(content['aspect']):
        print('- [Aspect] {}; [Prediction] {}; [Ontology scores] {}; [Compound score] {}'\
              .format(asp, content['sentiment'][i], content['score'][i], analyzer.lin2compound(content['score'][i]) ))

    print()
```


Key Features Supported with Aspect NLP toolkit
--------
* Aspect detection
* Aspect based sentiment analysis
* Text relevance
* Keyword extraction
* Topic summarization
* sentiment analysis
* Keyword extraction