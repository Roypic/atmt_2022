for ((i=5;i<6;i+=1));
do
python translate_beam.py --dicts ./data/en-fr/prepared --beam-size $i 
bash scripts/postprocess.sh model_translations.txt model_translations.p.txt en
cat model_translations.p.txt | sacrebleu data/en-fr/raw/test.en
done