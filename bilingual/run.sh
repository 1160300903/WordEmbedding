top=210
python vec2pairs-csls.py $top F10-W5.2en F10-W5.2zh mapped.2en mapped.2zh en-zh-csls
python pairs2matrixD.py mapped-T$top.en-zh-csls F10-W5.2en F10-W5.2zh D-en-zh-csls
rm -f tempdata/pairs/mapped-T$top.en-zh-csls
python matrix2svd.py F10-W5.X-en-zh mapped-T$top.D-en-zh-csls en-zh-csls
rm -f tempdata/matrix/mapped-T$top.D-en-zh-csls.npz
python svd2vec.py F10-W5.2en F10-W5.2zh F10-W5-T$top-L300.en-zh-csls 2en-csls 2zh-csls
rm -f tempdata/result/F10-W5-T$top-L300.en-zh-*
