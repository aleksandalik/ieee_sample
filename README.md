Цель соревнования: предсказать, является ли платежная операция мошеннической 
(соревнование https://www.kaggle.com/c/ieee-fraud-detection/overview).

Данные представляют базу из около 500 тысяч транзакций, по каждой из которых имеется информация, представленная в виде 434 колонок 
(информация о карте, адресе платежа, устройстве, продуктах и т.д.)

В ходе работы были отобраны наилучшие признаки (при помощи алгоритма SequentialFeatureSelector пакета mlxtend ), были составлены 
профили покупателей (при помощи кластеризации по входной информации) и благодаря этому сформированы новые признаки.

Задание по состоянию на 15.09.2019 находится в процессе выполнения. Приложенный код - черновой вариант с вырезанными кусками скрипта 
разработки новых признаков.