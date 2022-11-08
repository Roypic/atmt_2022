train_file=train.en
test_file=test.en
valid_file=valid.en
train_out_file=train.bpe.en
test_out_file=test.bpe.en
valid_out_file=valid.bpe.en
num_operations=10000
codes_file=bpe_tokens
vocab_file=dict
subword-nmt learn-bpe -s 32000 < ${train_file} > ${codes_file}
subword-nmt apply-bpe --dropout 0.1 -c ${codes_file} < ${train_file} > ${train_out_file}
subword-nmt apply-bpe  --dropout 0.1 -c ${codes_file} < ${test_file} > ${test_out_file}
subword-nmt apply-bpe  --dropout 0.1 -c ${codes_file} < ${valid_file} > ${valid_out_file}