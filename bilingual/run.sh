top=120
python vec2pairs-csls.py $top
python pairs2matrixD.py F10-W5-T$top.en-de F10-W5.1en F10-W5.1de D-en-de
rm -f tempdata/pairs/F10-W5-T$top.en-de
python matrix2svd.py F10-W5.X-en-de F10-W5-T$top.D-en-de en-de
rm -f tempdata/matrix/F10-W5-T$top.D-en-de.npz
python svd2vec.py F10-W5.1en F10-W5.1de F10-W5-T$top-L300.en-de 1en-csls 1de-csls
rm -f tempdata/result/F10-W5-T$top-L300.en-de-*
