data_dir=baseline
sed -r 's/(@@ )| (@@ ?$)//g' < ${data_dir}/infopankki_translations.txt  > ${data_dir}/infopankki_translations.bpe.txt
