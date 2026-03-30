system_prompt_en = '''
You are given a factual statement from Wikipedia. Decide whether it would be useful for someone seeking health-related information.
- ***Relevant*** means it provides useful health-related information such as symptoms, causes, risk factors, treatments, prevention, prognosis, lifestyle advice, or other patient-centered context.
- ***Not relevant*** means it is historical, administrative, technical, or other information that is not directly useful for someone with health-related queries.

***Task***: Respond with only “relevant” or “not relevant”.
***Input***: [Single factual statement]
***Response***: relevant or not relevant
'''


system_prompt_tr='''
Wikipedia’dan alınmış bir bilgi cümlesi verilecektir. Bu bilginin, sağlıkla ilgili bilgi arayan biri için yararlı olup olmayacağına karar verin.
- ***Relevant***: Belirtiler, nedenler, risk faktörleri, tedaviler, önleme, tanı, yaşam tarzı önerileri veya hasta odaklı başka bir bağlam gibi sağlıkla ilgili faydalı bilgiler içeriyorsa.
- ***Not relevant***: Tarihsel, idari, teknik (ör. hasta ile ilişkisi olmayan genetik diziler) ya da sağlıkla doğrudan ilgisi olmayan başka bilgiler içeriyorsa.

***Görev***: Yalnızca “relevant” veya “not relevant” şeklinde yanıt verin.
***Girdi***: [Tek bir gerçek / bilgi cümlesi]
***Cevap***: relevant or not relevant
'''

system_prompt_de= '''
Dir wird ein faktenbasiertes Statement von Wikipedia gegeben.Entscheide dich, ob es für jemanden, der nach Gesundheitsbezogene Information sucht, hilfreich ist.
- ***Relevant*** bedeutet, dass es nützliche gesundheitsbezogene Informationen wie Symptome, Ursachen, Risikofaktoren, Behandlungen, Präventionen, Prognosen, Vorschläge zur Anpassung des Lebensstils oder andere Patienten bezogene Informationen sind.
- ***Not relevant*** bedeutet, dass es sich um historische, administrative, technische oder andere Informationen handelt, die nicht direkt nützlich für jemanden mit gesundheitsbezogenen Fragen sind.

***Aufgabe***: Antworte nur mit  “Relevant” oder “Not relevant”.
***Eingabe***: [Einzelner Relevanter Fakt]
***Antwort***:: Relevant / Not relevant
'''

system_prompt_zh = '''
给你一段来自维基百科的事实陈述。请判断它对正在寻求健康相关信息的人是否有用。
- ***Relevant***  指的是该信息提供了有用的健康相关的内容，如症状、病因、风险因素、治疗方法、预防措施、预后情况、生活方式建议或其他以患者为中心的背景信息。
- ***Not Relevant***  指的是该信息属于历史、行政、技术等方面，对有健康相关疑问的人没有直接用处

***任务***： 仅回答“Relevant”或“Not relevant”。
***输入***： [单条事实陈述]
***回答***： Relevant / Not relevant
'''

system_prompt_it = '''
Ricevi una informazione estratta da Wikipedia. Determina se si tratta di un’informazione utile per qualcuno in cerca di informazioni in ambito medico. 
-Rilevante ovvero fornisce informazioni utili circa l’ambito medico come sintomi, cause, fattori di rischio, trattamenti, prevenzione, prognosi, suggerimenti sullo stile di vita o altri elementi relativi al paziente.
-Irrilevante ovvero informazioni storiche, amministrative, tecniche o altro non di utilitá per qualcuno in cerca di informazioni in ambito medico.
***Task***: Rispondi esclusivamente con  “Rilevante” or “Irrilevante”.
***Input***: [Infomrazione]
***Risposta***: Rilevante/Irrilevante
'''


system_prompts = {
    'en': system_prompt_en,
    'tr': system_prompt_tr,
    'de': system_prompt_de,
    'zh': system_prompt_zh,
    'it': system_prompt_it
}

user_prompts = {
    'en': "***Input***:\n{input_text}\n***Response***:\n",
    'tr': "***Girdi***:\n{input_text}\n***Cevap***:\n",
    'de': "***Eingabe***:\n{input_text}\n***Antwort***:\n",
    'zh': "***输入***:\n{input_text}\n***回答***:\n",
    'it': "***Input***:\n{input_text}\n***Risposta***:\n"
}