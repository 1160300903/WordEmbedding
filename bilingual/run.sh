# corpus2counts.py corpus vocab(OP)
python corpus2counts.py mono.tok.en 1en
python corpus2counts.py mono.tok.de 1de
# corpus2vocab.py src_corpus trg_corpus src_vocab(OP) trg_vocab(OP)
python corpus2vocab.py mono.tok.en mono.tok.de 1en 1de
# counts2matrixX.py src_count trg_count src_vocab trg_vocab matrixX(OP)
python counts2matrixX.py F10-W5.1en F10-W5.1de F10.1en F10.1de en-de
# vec2pairs.py src_vocab trg_vocab src_rdm_vec(P) trg_rdm_vec(P) pairs(OP)
python vec2pairs.py F10.1en F10.1de 1en 1de en-de
# pairs2matrixD.py pairs src_vocab trg_vocab matrixD(OP)
python F10-L300-T10.1en-de F10.1en F10.1de en-de
# matrix2svd.py matrixX matrixD src_vocab trg_vocab(OP) src_vec trg_vec(OP)
python matrix2svd F10-W5.en-de F10-L300-T10.en-de F10.1en F10.1de en-de