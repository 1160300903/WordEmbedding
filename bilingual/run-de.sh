top=70
python vec2pairs-csls.py $top F10-W5.1en F10-W5.1de mapped.1en mapped.1de en-de-csls
python pairs2matrixD.py mapped-T$top.en-de-csls F10-W5.1en F10-W5.1de D-en-de-csls
rm -f tempdata/pairs/mapped-T$top.en-de-csls
python matrix2svd.py F10-W5.X-en-de mapped-T$top.D-en-de-csls en-de-csls
rm -f tempdata/matrix/mapped-T$top.D-en-de-csls.npz
python svd2vec.py F10-W5.1en F10-W5.1de F10-W5-T$top-L300.en-de-csls 1en-csls 1de-csls
rm -f tempdata/result/F10-W5-T$top-L300.en-de-*
