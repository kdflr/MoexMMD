import requests
from dotenv import load_dotenv
import os
from datetime import date, time
from datetime import datetime as dt
from datetime import timedelta as td
from time import sleep
import holidays
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


load_dotenv()

TOKEN = os.getenv('READONLY_TOKEN')
FASTAPI_ADDR = 'http://localhost:8000'


header = {'Authorization': TOKEN,
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0'}

common_url = 'https://invest-public-api.tinkoff.ru:443/rest'

shares = {'SBER': {'ticker': 'SBER',
                   'figi': 'BBG004730N88',
                   'lot': 10,
                    'first': date(2018, 3, 7)}} # теѝтовый вариант модели оперирует котировками обыкновенных акций ѝбербанка

off_days = holidays.country_holidays('RU')


def convert_price(price: dict) -> float:
    """
    Конвертациѝ цены актива из формата price, nano в чиѝленное значение.

    price - целаѝ чаѝть,
    nano - дольнаѝ, выраженнаѝ чиѝлом порѝдка 10^9
    """

    converted_price = int(price['units']) + price['nano'] / 10e9

    return converted_price


def convert_candle(candle: dict, share: dict) -> dict:
    """
    Конвертациѝ данных ѝвечи и задание интереѝующих наѝ признаков.

    Поѝле теѝтированиѝ модели ѝ иѝпользованием ѝкѝпериментальных фич
    вѝе признаки будут задаватьѝѝ на ѝтом ѝтапе.
    """
    
    converted_candle = {
        'open': convert_price(candle['open']) * share['lot'],
        'close': convert_price(candle['close']) * share['lot'],
        'low': convert_price(candle['low']) * share['lot'],
        'high': convert_price(candle['high']) * share['lot'],
        'volume': int(candle['volume']),
        'timestamp': candle['time'],
        'ticker': share['ticker']
    }

    return converted_candle


def get_daily_candles(share: dict, candle_date: date, session: requests.Session) -> list:
    """
    Получение минутных ѝвечей за оѝновную торговую ѝеѝѝию указанного днѝ,
    без учета аукционов открытиѝ и закрытиѝ (10:00 - 18:40)
    """

    open_date = dt.combine(candle_date, time(7,0,0))
    close_date = dt.combine(candle_date, time(15,39,0))

    params = {
      "figi": share['figi'],
      "from": open_date.isoformat() + 'Z',
      "to": close_date.isoformat() + 'Z',
      "interval": "CANDLE_INTERVAL_1_MIN",
      "instrumentId": share['figi'],
      "candleSourceType": "CANDLE_SOURCE_EXCHANGE",
      "limit": 519
    }

    resp = session.post(f'{common_url}/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles',
                         headers=header,
                         json=params)
    data = resp.json()

    raw_candles = data['candles']

    converted_candles = [convert_candle(candle, share) for candle in raw_candles]

    return converted_candles


def get_all_candles(share: dict) -> None:
    """
    Получение всех свечей актива начиная с первой доступной минутной свечи,
    ограниченных рабочими днями.
    """

    parsed_date = share['first']

    with requests.Session() as session:
        while parsed_date < dt.now().date():
            if parsed_date.weekday() not in [5,6] and parsed_date not in off_days:
                current_candles = get_daily_candles(share, parsed_date, session)

                if len(current_candles) == 519:
                    requests.post(f'{FASTAPI_ADDR}/add_batch', json=current_candles)

                    print(f'Получены свечи за {parsed_date.isoformat()}')

            parsed_date += td(days=1)
            sleep(0.4)


def daily_add_task(share: dict, candle_date: date) -> None:
    """
    Ежедневное добавление свечей для дообучения модели
    """

    open_date = dt.combine(candle_date, time(7,0,0))
    close_date = dt.combine(candle_date, time(15,39,0))

    params = {
      "figi": share['figi'],
      "from": open_date.isoformat() + 'Z',
      "to": close_date.isoformat() + 'Z',
      "interval": "CANDLE_INTERVAL_1_MIN",
      "instrumentId": share['figi'],
      "candleSourceType": "CANDLE_SOURCE_EXCHANGE",
      "limit": 519
    }

    resp = requests.post(f'{common_url}/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles',
                         headers=header,
                         json=params)
    data = resp.json()

    raw_candles = data['candles']

    converted_candles = [convert_candle(candle, share['lot']) for candle in raw_candles]

    requests.post(f'{FASTAPI_ADDR}/add_batch', json=converted_candles)


if __name__ == '__main__':
    get_all_candles(shares['SBER'])
    
    scheduler = BlockingScheduler(timezone='UTC')
    main_session_end = CronTrigger(hour=15, minute=40)

    scheduler.add_job(func=daily_add_task, trigger=main_session_end, kwargs={'share': shares['SBER'], 'candle_date': dt.now().date()})

    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.shutdown()

