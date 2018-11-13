

##An Examination and Application of Text Summarizer (Textsum) on Amazon Customer Reviews
Category: NLP
Team members: Yi He, Can Jin He

## 1 Introduction

Text summarization is a meaningful way to handle large amounts of text data. For this project, we will work on improving the existing text summarizer, Google's Textsum. Additionally, we would like to explore, transfer learning, whether a text summarizer trained on CNN/DailyMail can be used to summarize Amazon reviews. We want to introduce a novel approach by using a universal sentence encoder and word embedding to test whether it improves the model. Once we have a high performing model using the CNN/Daily dataset, we would like to apply it to Amazon reviews to qualitative test the accuracy.

## 2 Approach

**Dataset:** Google's Textsum was developed using the Gigaword dataset, not open sourced and inaccessible by Stanford students. However, recent research on summarizer has moved toward using the CNN/DailyMail dataset. This dataset has the original text and a short set of summarized bullet points that represent meaningful “highlights” of each article.

The CNN contains 92,570 articles, and DailyMail includes 219,503 articles. Moreover, each article includes 3 to 4 bullet points. We use the first two sentences of each article as model input, and the first bullet point as the gold label sentence. We split the dataset to train/dev/test as 80:15:5.

**Preprocessing:** We processed each article using sent tokenize from the NLTK package. We limited the vocabulary size to 200k. We created a vocabulary list which outputs word count frequency. Also, we converted each article to a sequence of a serialized Tensorflow object.

**Training:** Our model was trained for 10 epochs using the following parameters:
- Learning rate = 0.15
- Batch size = 4
- Encoding layer = 4
- Number of hidden units: 256
- Number of softmax samples = 4,096
- Word embedding size = 128

**Evaluation:** ROUGE is a common metric for automated text summarization tasks that computes a number of co-occurrence statistics between a predicted output and true output. [1] We run ROUGE on our generated and gold label for the test set.

## 3 Experiments and Results

**Model description and Training strategy:** We used an open source Tensorflow model, Textsum, as our baseline. Textsum uses an encoder-decoder model with a bidirectional LSTM-RNN encoder and an attentive unidirectional LSTM-RNN decoder with beam search. The encoder-decoder model is trained end-to-end. [1] We followed the approach described by Danqi C. (2017) to train our baseline model. For the first decode timestep, we feed in the last output of the encoder as well as the embedding for the start "(<s>)" token. For subsequent decode timesteps, the decoder uses the last decoder output in addition to the word embedding for the previous word to generate the next word. During the train step, the previous word is the actual previous word from the gold label, but during the decode step, the previous word is the previously-generated word. The decoding process will continue until the generated summary reaches the max decode length set at 30, or until it generates an EOS token. [1]

**Results and Problems:** Similar to the results from Danqi C. (2017), the results we produced from our baseline model is inadequate on the CNN/DailyMail dataset. After training on 80% of the data set, we noticed a large number of <UNK> tokens in the summaries that made it unreadable. After investigating on Github, we found out that we ran into the same issue as other researchers who are trying to use the same model and dataset. CNN/DailyMail contains 312k articles, which is much fewer than Gigaword’s 10 million articles used by Textsum. The discrepancy between the number of articles between the two data sets resulted in such different outcomes.

Two problems we identified in the existing models are a lack of generalization in using longer articles, and an inability to use the source text itself to provide words and relying on using <UNK> tokens instead. [1]

## 4 Future work

We will continuously work on the model to improve its performance on CNN/Daily Mail. We want to take the following steps in our project:
- Because of the limited amount of training data in CNN/DailyMail dataset, we will initialize our word embeddings with ConceptNet Numberbatch to give our model more powerful semantic representations of the source input tokens.
- Since Textsum only used the first two sentences of an article as input sentences, we will use the universal sentence encoder to find the two most similar sentences to the gold label sentence then feed into Textsum.
- Add Amazon reviews to train with CNN\DailyMail and apply the model on Amazon reviews

## 5 Contribution
We have two team members and we worked equally on the project.

