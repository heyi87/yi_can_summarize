# An Examination and Application of Text Summarizer (Textsum) on Amazon Customer Reviews
### Authors:   
###  Yi He (yihe1008@stanford.edu) , Can Jin (cjh0511@standford.edu)

## Requirements:
```
#create virtual environment
virtualenv venv 
source venv/bin/activate
pip install -r requirements.txt

#Install Bazel, needed for building textsum  
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python  
wget "https://github.com/bazelbuild/bazel/releases/download/0.18.1/bazel 0.18.1-installer-linux-x86_64.sh"  
chmod +x bazel-0.18.1-installer-linux-x86_64.sh  
./bazel-0.18.1-installer-linux-x86_64.sh --user
```

## Task
Text summarization is a meaningful way to handle large amounts of text data. For this project, we will work on improving the existing text summarizer, Google's Textsum. Additionally, we would like to explore, transfer learning, whether a text summarizer trained on CNN/DailyMail can be used to summarize Amazon reviews. We want to introduce a novel approach by using a universal sentence encoder and word embedding to test whether it improves the model. Once we have a high performing model using the CNN/Daily dataset, we would like to apply it to Amazon reviews to qualitative test the accuracy.

## Download the Data Set
An example of the data set is located at /example_data/cnn/stories. If you wish to download the entire dataset go to: https://cs.nyu.edu/~kcho/DMQA/ and download the stories.

## Quickstart
To begin with the original TextSum training:

```
bazel build -c opt textsum/...

#create vocab list
python textsum_data_convert.py \  
  --command text_to_vocabulary \  
  --in_directories data/cnn/stories \  
  --out_files data/vocab  
 
#change data into protbuf for tensorflow to train on
python textsum_data_convert.py \  
  --command text_to_binary \  
  --in_directories data/cnn/stories \  
  --out_files data/train.bin, data/valid.bin,data/test.bin \  
  --split 0.8,0.15,0.05  
 
#run training 
bazel-bin/textsum/seq2seq_attention \  
  --mode=train \  
  --article_key=article \  
  --abstract_key=abstract \  
  --data_path=data/train.bin \  
  --vocab_path=data/vocab \  
  --log_root=log \  
  --train_dir=log/train \  
  --truncate_input=True  
  
#validation  
$ bazel-bin/textsum/seq2seq_attention \  
    --mode=eval \  
    --article_key=article \  
    --abstract_key=abstract \  
    --data_path=data/validation-* \  
    --vocab_path=data/vocab \  
    --log_root=textsum/log_root \  
    --eval_dir=textsum/log_root/eval  
  
  
#test  
bazel-bin/textsum/seq2seq_attention \  
  --mode=decode \  
  --article_key=article \  
  --abstract_key=abstract \  
  --data_path=data/test.bin \  
  --vocab_path=data/vocab.bin \  
  --log_root=log \  
  --decode_dir=log/decode \  
  --beam_size=8 \  
  --truncate_input=True

#calculate rouge score from testing
python rougescore.py -d log/decode/decode -r log/decode/ref 
# this will provide the Rouge-1 Rouge-2 and Rouge-l F scores
```

To do transfer learning on Amazon reviews:
```
python amazon_to_stories.py -i example_data/amazon_reviews.json -o example_data/amazon/stories

python textsum_data_convert.py \  
  --command text_to_binary \  
  --in_directories example_data/amazon/stories \  
  --out_files data/amazon_test.bin
  --split 1

bazel-bin/textsum/seq2seq_attention \  
  --mode=decode \  
  --article_key=article \  
  --abstract_key=abstract \  
  --data_path=data/amazon_test.bin \  
  --vocab_path=data/vocab.bin \  
  --log_root=log \  
  --decode_dir=log/amazon_decode \  
  --beam_size=8 \  
  --truncate_input=True

#correlate back the decoded to the amazon reviews from the original json file
python rougescore.py -d log/amazon_decode/decode -r log/amazon_decode/ref  -am example_data/amazon_reviews.json -ao good_amazon_reviews.txt
```

Sentence Encoding
	Find the two most similar sentences in the article to the highlight 
	This will create a stories directory only contain the two most similar sentences of the article to the first highlight 
```
python similar_sentences.py -i example_data/cnn/stories -o example_data/cnn/similar_sentences
```
Once this is created, use it for training for the TextSum and calculate the Rouge scores

ConceptNet NumberBatch:
	ConceptNet NumberBatch require an embedding from http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/17.06/mini.h5 
	We used the mini.h5 which vectorize each English word into 300 numbers. 
```
python ConceptNet_Numberbatch.py -i example_data/cnn/stories -o example_data/ -ne mini.h5
```
Once this is created, use it for training for TextSum and calculate the Rouge scores


