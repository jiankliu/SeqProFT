export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/GO-MF/GO-MF-nf-mha.yml 
python train_search.py configs/GO-MF/GO-MF-f-mha.yml 

python train_search.py configs/GO-MF/GO-MF-nf-smh.yml 
python train_search.py configs/GO-MF/GO-MF-f-smh.yml 

python train_search.py configs/GO-MF/GO-MF-nf-smha.yml 
python train_search.py configs/GO-MF/GO-MF-f-smha.yml

python train_search.py configs/GO-MF/GO-MF-nf-mha-35M.yml 
python train_search.py configs/GO-MF/GO-MF-f-mha-35M.yml 
python train_search.py configs/GO-MF/GO-MF-nf-mha-150M.yml
python train_search.py configs/GO-MF/GO-MF-f-mha-150M.yml

python train_search.py configs/GO-MF/GO-MF-nf-smha-35M.yml 
python train_search.py configs/GO-MF/GO-MF-f-smha-35M.yml 
python train_search.py configs/GO-MF/GO-MF-nf-smha-150M.yml
python train_search.py configs/GO-MF/GO-MF-f-smha-150M.yml
