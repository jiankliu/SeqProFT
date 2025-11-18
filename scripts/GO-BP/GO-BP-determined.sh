export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/GO-BP/GO-BP-nf-mha.yml 
python train_search.py configs/GO-BP/GO-BP-f-mha.yml 

python train_search.py configs/GO-BP/GO-BP-nf-smh.yml 
python train_search.py configs/GO-BP/GO-BP-f-smh.yml

python train_search.py configs/GO-BP/GO-BP-nf-smha.yml 
python train_search.py configs/GO-BP/GO-BP-f-smha.yml 

python train_search.py configs/GO-BP/GO-BP-nf-mha-35M.yml
python train_search.py configs/GO-BP/GO-BP-f-mha-35M.yml
python train_search.py configs/GO-BP/GO-BP-nf-mha-150M.yml
python train_search.py configs/GO-BP/GO-BP-f-mha-150M.yml

python train_search.py configs/GO-BP/GO-BP-nf-smha-35M.yml
python train_search.py configs/GO-BP/GO-BP-f-smha-35M.yml
python train_search.py configs/GO-BP/GO-BP-nf-smha-150M.yml
python train_search.py configs/GO-BP/GO-BP-f-smha-150M.yml