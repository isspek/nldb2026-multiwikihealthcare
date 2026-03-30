ENTITY_ANALYZER ='''
You are an intelligent assistant that identifies whether an entity is related to healthcare.
When given an entity (e.g., a term, organization, person, product, etc.), first determine whether it is related to healthcare.
If it is related to healthcare, return the URL of its Wikipedia page (e.g., https://en.wikipedia.org/wiki/ENTITY_NAME).
If it is not related to healthcare, return False.
Consider healthcare to include: medicine, medical devices, hospitals, health policy, public health, pharmaceuticals, biotechnology, mental health, nutrition, diet, food safety, and functional foods.
'''


TRANSLATE_TR_EN = '<|im_start|>user\nAşağıdaki metni Türkçe\'den İngilizce\'ye çeviriniz:\nTürkçe: INPUT_TEXT\nİngilizce: <|im_end|>\n<|im_start|>assistant\n'
TRANSLATE_DE_EN = '<|im_start|>user\nÜbersetzen Sie den folgenden Text vom Deutschen ins Englische:\nDeutsch: INPUT_TEXT\nEnglisch: <|im_end|>\n<|im_start|>assistant\n'

TRANSLATION_PROMPTS = {
    'tr': TRANSLATE_TR_EN,
    'de': TRANSLATE_DE_EN
}