export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/TAPE-flu/TAPE-flu-nf-mha.yml 
python train_search.py configs/TAPE-flu/TAPE-flu-f-mha.yml 

python train_search.py configs/TAPE-flu/TAPE-flu-nf-smh.yml 
python train_search.py configs/TAPE-flu/TAPE-flu-f-smh.yml 

python train_search.py configs/TAPE-flu/TAPE-flu-nf-smha.yml
python train_search.py configs/TAPE-flu/TAPE-flu-f-smha.yml

python train_search.py configs/TAPE-flu/TAPE-flu-nf-mha-35M.yml
python train_search.py configs/TAPE-flu/TAPE-flu-f-mha-35M.yml
python train_search.py configs/TAPE-flu/TAPE-flu-nf-mha-150M.yml
python train_search.py configs/TAPE-flu/TAPE-flu-f-mha-150M.yml

python train_search.py configs/TAPE-flu/TAPE-flu-nf-smha-35M.yml
python train_search.py configs/TAPE-flu/TAPE-flu-f-smha-35M.yml
python train_search.py configs/TAPE-flu/TAPE-flu-nf-smha-150M.yml
python train_search.py configs/TAPE-flu/TAPE-flu-f-smha-150M.yml
