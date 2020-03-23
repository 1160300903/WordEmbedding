for i in $( seq 1 8 )
do
t=`expr $i * 10`
python ../../vecmap/map_embeddings.py --unsupervised vector/F10-W5-T($t)-L300.1en-csls vector/F10-W5-T($t)-L300.1de-csls vector/mapped-T($t).1en-csls vector/mapped-T($t).1de-csls -v &
done