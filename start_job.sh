#!/usr/bin/env bash

mkdir data

#Install Bazel, need for building textsum
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
wget "https://github.com/bazelbuild/bazel/releases/download/0.18.1/bazel-0.18.1-installer-linux-x86_64.sh"
chmod +x bazel-0.18.1-installer-linux-x86_64.sh
./bazel-0.18.1-installer-linux-x86_64.sh --user

bazel build -c opt textsum/...

#create virtual environment
sudo apt install virtualenv
cd ..
virtualenv venv
source venv/bin/activiate
pip install -r yi_can_summarize/requirements.txt
pip install git+https://github.com/tagucci/pythonrouge.git #install rouge-calculation package

#download cnn new articles from S3 account
cd yi_can_summarize
mkdir data
cd data
aws s3 cp s3://yicindy/cnn_stories.tar .
tar -xvf cnn_stories.tar
rm -rf cnn_stories.tar

##divide the cnn stories into pieces for sentence encoding
for f in *;
do
    d=dir_$(printf %03d $((i/20000+1)));
    mkdir -p $d;
    mv "$f" $d;
    let i++;
done

cd ..
python textsum_data_convert.py \
  --command text_to_vocabulary \
  --in_directories data/cnn/stories \
  --out_files data/vocab

python textsum_data_convert.py \
  --command text_to_binary \
  --in_directories data/cnn/stories \
  --out_files data/train.bin, data/valid.bin,data/test.bin \
  --split 0.8,0.15,0.05

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

#check the rouge score
python rouge_calculation.py \
 --reference /log/decode/ref \
 --summary /log/decode/summary
