#!/usr/bin/env bash

mkdir data

#Install Bazel
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
wget "https://github.com/bazelbuild/bazel/releases/download/0.18.1/bazel-0.18.1-installer-linux-x86_64.sh"
chmod +x bazel-0.18.1-installer-linux-x86_64.sh
./bazel-0.18.1-installer-linux-x86_64.sh --user

bazel build -c opt textsum/...

sudo apt install virtualenv
cd ..
virtualenv venv
source venv/bin/activiate
pip install -r yi_can_summarize/requirements.txt

cd yi_can_summarize
mkdir data
cd data
aws s3 cp s3://yicindy/cnn_stories.tar .
tar -xvf cnn_stories.tar
rm -rf cnn_stories.tar

cd ..
python textsum_data_convert.py \
  --command text_to_vocabulary \
  --in_directories data/cnn/stories \
  --out_files data/cnn-vocab.bin

python textsum_data_convert.py \
  --command text_to_binary \
  --in_directories data/cnn/stories \
  --out_files data/cnn-train.bin,data/cnn-validation.bin,data/cnn-test.bin \
  --split 0.8,0.15,0.05

bazel-bin/textsum/seq2seq_attention \
  --mode=train \
  --article_key=article \
  --abstract_key=abstract \
  --data_path=data/cnn-train.bin \
  --vocab_path=data/cnn-vocab.bin \
  --log_root=log_root \
  --train_dir=log_root/train \
  --truncate_input=True

