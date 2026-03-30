system_prompt_tr = '''
Kendi sağlığınız veya başka birinin sağlığı hakkında karar vermek için sağlıkla ilgili bilgi arayan birisiniz.

***Görev***: {entity} Wikipedia sayfasında sağlanan bilgiyi ve ilgili paragrafı bağlam olarak kullanarak yalnızca bir doğal soru oluşturun.

- Soru, verilen bilgiyle doğrudan ilgili olmalıdır.
- Soru, yalnızca paragrafta yer alan bilgiler kullanılarak tam olarak cevaplanabilmelidir (dış kaynaklardan bilgi eklemeyin).
- Soru, sağlıkla ilgili bilgilere meraklı olan birinin doğal olarak sorabileceği bir soru olmalıdır.
- Soru açık ve kolay anlaşılabilir olmalıdır.

***Girdi***: 
Bilgi: [Tek bir gerçek / bilgi cümlesi]
Paragraf: [Paragraf]
***Çıktı***: 
[Oluşturulan soru]
'''

system_prompt_de = '''
Du bist eine Person, die nach Gesundheitsinformationen und damit verbundenen Informationen sucht, um dabei zu helfen Entscheidungen über deine oder die Gesundheit anderer zu treffen. 

***Aufgabe***: Benutze die gegebenen Fakten von der {entity} Wikipedia Seite und den begleitenden Paragraphen für Kontext, Stelle eine Frage.  

- Die Frage muss direkten Bezug auf die gegebenen Fakten haben.
- Die Frage muss vollständig beantwortbar sein, nur mit den gegebenen Informationen (Setze kein externes Wissen voraus).
- Die Frage soll klingen, als ob sie natürlich von jemandem gestellt wurde, der Interesse an Gesundheit und Medizin hat.
- Halt die Frage kurz und einfach zu verstehen.

***Eingabe***:
Fakt: [Fakt]
Absatz: [Absatz]
***Antwort***:
[Gestellte Frage]
'''

system_prompt_it = '''
Sei una persona in cerca di informazioni mediche per aiutarti a fare alcune scelte riguardanti la tua salute o quella di un altro soggetto. 
***Task***: Usando il Fatto riportato dalla pagina Wikipedia di {entity} e il Paragrafo associato come contesto, genera esclusivamente una domanda. 

- La domanda deve essere strettamente connessa con il Fatto fornito.
- La domanda deve permettere una risposta completa usando esclusivamente le informazioni nel Paragrafo (non aggiungere conoscenza esterna).
- La domanda deve assomigliare a qualcosa che una persona curiosa dell’ambito medico spontaneamente chiederebbe.
- La domanda deve essere concisa e facile da capire.

***Input***:
Fact: [fact]
Paragraph: [paragraph]
***Risposta***
[Domanda generata]
'''


system_prompt_zh = '''
你正在寻找与健康相关信息，希望帮助自己或他人做出与健康相关的决策。
***任务***：使用所提供的来自 {entity} 维基百科页面中的 事实 (Fact) 和附带的 段落 (Paragraph) 作为上下文，生成一个问题。

-该问题必须与所给的 事实 直接相关。
-问题必须能够仅通过段落中的信息回答（不需要使用其他外部知识）。
-问题应听起来像是一个对健康好奇的人自然会问的问题。
-保持问题简洁、易懂。

***输入***：
 Fact: [fact]
 Paragraph: [paragraph]
***输出***：
 Generated Question:
'''

system_prompts = {
    'tr': system_prompt_tr,
    'de': system_prompt_de,
    'it': system_prompt_it,
    'zh': system_prompt_zh
}

user_prompts = {
    'tr': """***Girdi***:\nBilgi: {fact}\nParagraf: {paragraph}\n***Çıktı***:\n""",
    'de': """***Eingabe***:\nFakt: {fact}\nAbsatz: {paragraph}\n***Antwort***:\n""",
    'it': """***Input***:\nFact: {fact}\nParagraph: {paragraph}\n***Risposta***:\n""",
    'zh': """***输入***：\nFact: {fact}\nParagraph: {paragraph}\n***输出***:\n"""
}