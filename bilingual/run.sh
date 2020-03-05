# corpus2counts.py corpus vocab(OP)
python corpus2counts.py mono.tok.en 1en
python corpus2counts.py mono.tok.de 1de
# corpus2vocab.py src_corpus trg_corpus src_vocab(OP) trg_vocab(OP)
python corpus2vocab.py mono.tok.en mono.tok.de F10-W5.1en F10-W5.1de 1en 1de
# counts2matrixX.py src_count trg_count src_vocab trg_vocab matrixX(OP)
python counts2matrixX.py F10-W5.1en F10-W5.1de F10-W5.1en F10-W5.1de X-en-de.nopmi False
# vec2pairs.py src_vocab trg_vocab src_rdm_vec(P) trg_rdm_vec(P) pairs(OP)
python vec2pairs.py F10-W5.1en F10-W5.1de 1en 1de en-de
# pairs2matrixD.py pairs src_vocab trg_vocab matrixD(OP)
python pairs2matrixD.py F10-W5-L300-T10.en-de F10-W5.1en F10-W5.1de D-en-de
# matrix2svd.py matrixX matrixD svd_output(OP)
python matrix2svd.py F10-W5.X-en-de F10-W5-L300-T10.D-en-de en-de
# svd2vec.py src_vocab trg_vocab svd_path src_vec(OP) trg_vec(OP)
python svd2vec.py F10-W5.1en F10-W5.1de F10-W5-L300.en-de 1en 1de
## O代表输出, P代表文件名会被自动加参数