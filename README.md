Aspect Detection NLP Toolkit
=============

Aspect Detection NLP toolkit (https://pypi.org/project/aspectnlp/) is a Python package that performs various NLP tasks based on aspect detection and aspect based sentiment analysis.

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
from aspectnlp.aspect_detector import aspectDetector
from aspectnlp.aspect_sentiment import AspectSentimentScorer
from aspectnlp.vae_topic_model import VAETopic
from aspectnlp.w2v import fasttext_emb


corpus=[
    "While there's a decent menu, it shouldn't take ten minutes to get your drinks and 45 for a dessert pizza.",
    'Faster than old XBox, less crashing, definitely smoother experience.',
    'The product is really good but performance with things like sharing deals with partners is terrible and takes forever.',
    "Mostly still learning the tool. Things tend to change randomly from time to time... so that can be tricky. But its improving steadily!",
    "Office365 is improving, but there is a lot more we could do to make Office365 work for us and model best practices to our customers for how to use Excel.",
    "but it's not working now. Showing 'error loading accounts'.",
    "Changes in the new version of Chrome are in the right direction, but performance still feels mediocre at best. \
There are various places where it could prevent page-loading delays.",
    ]

asp_detector=aspectDetector('custom_emb.vec.bin')
sent_asp=asp_detector.detect(corpus,disp=True)


# Topic modeling based on aspect words
corpus=[i['aspect'] for i in sent_asp]
topic_model=VAETopic(corpus,n_topics=10)
topic_model.fit()
top_words,topics,topic_ids=topic_model.get_top_words_and_topics(disp=True)

# Aspect based on aspect words
analyzer=AspectSentimentScorer('custom_emb.vec.bin')
for sentId,sample in enumerate(sent_asp):
    content = list(analyzer.generateInputContent(sample))
    sentiment, scores = analyzer.scoreABSA(content)
    sent_asp[sentId]['sentiment']=sentiment
    sent_asp[sentId]['score']=scores
    sent_asp[sentId]['compound']=[analyzer.lin2compound(s) for s in scores]
    
for content in sent_asp:
    print(content)

# aspect fuzzy matcher
word_list=['menu','sales','account','User Interface','search','Note','telephony','portal','food']
fastext= fasttext_emb('custom_emb.vec.bin')
for i,v in enumerate(fastext.w2v(word_list)):
    print(word_list[i],':',v)
```


Key Features Supported with Aspect NLP toolkit
--------
* Aspect detection
* Aspect based sentiment analysis
* Text relevance
* Keyword extraction
* Topic summarization
* sentiment analysis
