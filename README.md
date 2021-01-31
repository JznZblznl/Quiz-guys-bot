# Quiz Guys Bot
Команда Quis Guys представляет телеграмм-бота. Проект подготовлен в рамках хакатона Pets Projects Hackaton [Data Ёлка 2020](https://ods.ai/competitions/pet_projects_2020).

## Почему этот бот?
Есть проблема: В сегодняшнем мире, где мы редко встречаемся в реале, в мире удалёнок и виртуального общения, на замену приходят вебинары, чат-боты, коуч-агенства и так далее. Учиться стало веселее, а вот учить - сложнее! Нужно каждый день придумывать новые задания, новые тесты. Больше половины рабочего дня учитель тратит на подготовку и проверку тестов и у него не остается времени на работу с учащимися.

Что мы предлагаем? Мы предлагаем решение - карманного помощника для учителя английского языка - телеграмм-бот [“Ask me”](https://t.me/Quizguysbot). Наш бот ждет от вас текст и на его основе готовит вопросы по тексту. Учитель только выбирает текст, а вопросы к тексту генерируются автоматически. 
Что вы получите? Вы получите два документа, один - для ученика (с текстом и вопросами), второй - для учителя (с текстом, вопросами и ответами на них). Учителю остается лишь разместить вопросы с ответами в тестирующей системе или отправить ученику готовый документ.

## Как это работает?
Как это работает? Бот основан на принципах NLP - Natural Language Processing - обработка текстов на естественном языке (т. е. на языке, на котором говорят и пишут люди).
* Выбираем, к чему задать вопрос: NER (flair) + ключевые слова (pke.unsupervised.SingleRank)
* Генерируем вопросы  предобученной моделью (mrm8488/t5-base-finetuned-question-generation-ap)
* Сохраняем в docx и отправляем пользователю в телеграм-боте.

## Как это запустить?
### Код 
Код представлен в двух вариантах: 
* в виде [Google Colab Notebook](https://github.com/JznZblznl/Quiz-guys-bot/blob/main/QuizGuysBot_v0.ipynb), в том виде, в котором он сейчас функционирует; и 
* в виде [кода на питоне](https://github.com/JznZblznl/Quiz-guys-bot/blob/main/bot.py), с добавленными коментариями

### Необходимые библиотеки
Нужны следующие библиотеки
* [python-telegram-bot](https://pypi.org/project/python-telegram-bot/) библиотека для телеграм-бота
* [transformers](https://pypi.org/project/transformers/) State-of-the-art Natural Language Processing for PyTorch and TensorFlow 2.0
* [sentencepiece](https://pypi.org/project/sentencepiece/) Python wrapper for SentencePiece. This API will offer the encoding, decoding and training of Sentencepiece.
* [flair](https://pypi.org/project/flair/) A very simple framework for state-of-the-art NLP
* [pke](https://github.com/boudinfl/pke) Python keyphrase extraction, тавится с git ```pip install git+https://github.com/boudinfl/pke.git```
* [python-docx](https://pypi.org/project/python-docx-1/) Create and update Microsoft Word .docx files

## Наша комманда
Команда Quis Guys 
* [Liza Nosova](https://t.me/lizavet_nosova)
* [Aleksandr Kolotuskin](https://t.me/tenj1n)
* [Татьяна Д.](https://t.me/toph_b)
* [Mike Woodpecker](https://t.me/voodoo_woodpecker)

Ментор
* [Roman Romadin](https://t.me/RomanRomadin)
