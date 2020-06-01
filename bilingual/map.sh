for i in 30 50 70	
do
CUDA_VISIBLE_DEVICES=3 python ../../vecmap/map_embeddings.py --unsupervised vector/F10-W5-T$i-L300.2en-csls vector/F10-W5-T$i-L300.2zh-csls vector/mapped-T$i.2en-csls vector/mapped-T$i.2zh-csls -v --csls --cuda &
done
