export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-nf-mha.yml 
python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-f-mha.yml 

python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-nf-smh.yml 
python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-f-smh.yml 

python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-nf-smha.yml
python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-f-smha.yml

python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-nf-mha-35M.yml
python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-f-mha-35M.yml
python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-nf-mha-150M.yml
python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-f-mha-150M.yml

python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-nf-smha-35M.yml
python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-f-smha-35M.yml
python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-nf-smha-150M.yml
python train_search.py configs/FLIP-gb1-sampled/FLIP-gb1-sampled-f-smha-150M.yml
