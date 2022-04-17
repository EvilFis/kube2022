import os
import json
import telebot
import TextPreprocessing as tp

from vosk import Model, KaldiRecognizer
import wave

# Считыаем найстройки с файла
with open('./setting.json', 'r', encoding='utf-8') as f:
    setting = json.load(f) 


dataBase = {} # Переменная для хранения текстов пользователей

model = Model(r"./models/vosk-model-ru-0.22") # Загрузка модели по распознование текстовой речи
bot = telebot.TeleBot(token=setting['token']) # Настройка API бота
textP = tp.TextPreprocessing(setting) # Кастомный класс по сравнению текстов 


def translateSpeech(id):
    '''
    Функция по переводу речи в текст
    
        Parametrs:
            id(int): id пользователя для считывания файла с папки
            
        Return:
            result(str): Текст преобразованного голоса
    
    '''
    wf = wave.open(f"./voice/{id}.wav", "rb")
    rec = KaldiRecognizer(model, 48000)
    
    result = ''
    last_n = False
    
    while True:
        data = wf.readframes(48000)
        if len(data) == 0:
            break
    
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
    
            if res['text'] != '':
                result += f" {res['text']}"
                last_n = False
            elif not last_n:
                result += '\n'
                last_n = True
    
    res = json.loads(rec.FinalResult())
    result += f" {res['text']}"
    return result


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, '''Привет ✌️ 
Вы являетесь счастливым обладателем тестовой версии ПО по проверке регламентов переговоров сотрудников РЖД

Для проверки диалогов укажите диктора перед репликой с помощью специального тега `<s>` или пришлите голосовое сообщение, система автоматически подскажет, что дальше делать.
    

    Пример сообщений с помощью тега:
        
    <s> Машинист поезда №124 следует по маршруту №32. Через пол часа прибуду на место
    
    <s> Машинист поезда №124 продолжайте следование по маршруту. По прибытию в точку назначения следуйте в обратном направлении
    
    _Примечание: Если вы общаетесь с ботом через тег `<s>`, то необходимо в одном сообщении указывать сразу два текста, через разные теги `<s>`_

На данный момент в боте реализованы не все функции. Работа над проектом продолжается!

''', parse_mode="Markdown")


@bot.message_handler(content_types=['text'])
def get_text(message):
    
    mess = (message.text).split('<s>')
    
    if  not len(mess) > 1:
        pass
        #bot.send_message(message.chat.id, 'Шаблон сообщения не был найден!')
    else:
        vectors = textP.vectorize_text(mess[1:3])
        accuracy = textP.cosine_vector(vectors)
        
        text = f'''
**РЕЗУЛЬТАТЫ АНАЛИЗА ТЕКСТА**

Текст переговоров:
    
    _Корреспондент 1:_ {mess[1]}
    
    _Корреспондент 2:_ {mess[2]}
    
    **Схожесть текстов: {accuracy}%**
'''
        
        bot.send_message(message.chat.id, text, parse_mode="Markdown")
        
        
@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    filename = str(message.chat.id)
    file_name_full = f"./voice/{filename}.ogg"
    file_name_full_converted = f"./voice/{filename}.wav"
    
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(file_name_full, 'wb') as new_file:
        new_file.write(downloaded_file)
    
    os.system("ffmpeg -i "+file_name_full+"  "+file_name_full_converted)
    text = translateSpeech(message.chat.id)
    
    os.remove(file_name_full)
    os.remove(file_name_full_converted)
    
    info = dataBase.get(str(message.chat.id))
    
    if info:
         if len(info) == 1:
             
            info.append(text)
            dataBase[str(message.chat.id)] = info
            
            bot.send_message(message.chat.id, 'Обрабатываем полученные данные...')
            
            vectors = textP.vectorize_text(info)
            accuracy = textP.cosine_vector(vectors)
            
            text = f'''
            **РЕЗУЛЬТАТЫ АНАЛИЗА ТЕКСТА**
            
            Текст переговоров:
            
            _Корреспондент 1:_ {info[0]}
            
            _Корреспондент 2:_ {info[1]}
            
            **Схожесть текстов: {accuracy}%**
            '''
            
            bot.send_message(message.chat.id, text, parse_mode="Markdown")
            dataBase.pop(str(message.chat.id))
        
    else:
        dataBase[str(message.chat.id)] = [text]
        bot.send_message(message.chat.id, "Необходимо записать второе сообщение...")
    
    

if __name__ == "__main__":
    bot.infinity_polling()