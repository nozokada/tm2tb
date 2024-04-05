from pathlib import Path
from stops_vector_generator import StopsVectorGenerator
from tm2tb import BitermExtractor
from spacy.lang.ja import stop_words

similiarity_min = 0.3

en_sentence = (
    "Un-retiring a flow step will make it available for addition to smart campaigns.  Services may be retired again later.<p>Are you sure you want to un-retire the <b>{0}</b> flow step service?<div class='mktMessageIndent mktRequired'><span id='mktAgreeCbWrap'></span></div>"
)

jp_sentence = (
    "フローステップの廃止を解除するとスマートキャンペーンに追加できます。サービスは後で再び廃止できます。<p><b>{0}</b> フローステップサービスの廃止を解除してよろしいですか？<div class='mktMessageIndent mktRequired'><span id='mktAgreeCbWrap'></span></div>"
)

if not Path("stops_vectors/768/ja.npy").exists():
    generator = StopsVectorGenerator("sbert_models/LaBSE")
    generator.create_vector(list(stop_words.STOP_WORDS))

biterm_extractor = BitermExtractor((en_sentence, jp_sentence))
biterms = biterm_extractor.extract_terms(similiarity_min)
print(biterms[:10])
