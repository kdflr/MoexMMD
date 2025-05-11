import requests

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import pandas as pd
import numpy as np

from datetime import datetime as dt
from datetime import timedelta as td
from datetime import date
from datetime import time

from os import load_dotenv
import os

labels = [1, 0, -1]

class CandlesDataset(Dataset):
    def __init__(self, processed_data: torch.Tensor, input_window_size: int, output_window_size: int = 5):
        total_window_size = input_window_size + output_window_size

        count_windows = processed_data.shape[1] - total_window_size + 1

        self.data_x = torch.empty(processed_data.shape[0], count_windows, input_window_size, 9, device=torch.device('cuda'))
        self.data_y = torch.empty(processed_data.shape[0], count_windows, 3, device=torch.device('cuda'))

        for j in range(processed_data.shape[0]):
            x_batch = torch.empty(count_windows, input_window_size, 9)
            y_batch = torch.empty(count_windows, 3)

            for i in range(count_windows):
                input_window = processed_data[i, i:i+input_window_size]
                x_batch[i] = input_window

                output_window = processed_data[i, i+input_window_size:i+total_window_size]
                window_bodies_sum = output_window[:, 1].sum()
                window_gaps_sum = output_window[1:, 2].sum()

                total_window_sum = window_bodies_sum + window_gaps_sum

                window_direction = torch.sign(total_window_sum)

                onehot = torch.tensor([0, 0, 0], dtype=torch.float32)
                onehot[labels.index(window_direction.item())] = 1

                y_batch[i] = onehot

            self.data_x[j] = x_batch
            self.data_y[j] = y_batch
        
        self.data_x = self.data_x.flatten(0, 1)
        self.data_y = self.data_y.flatten(0, 1)

    def __getitem__(self, indx):
        return (self.data_x[indx], self.data_y[indx])

    def __len__(self):
        return len(self.data_x)
    

class Sluggard(nn.Module):
    def __init__(self):
        super().__init__()
        self.two_patterns = nn.Sequential(
            nn.Conv2d(1, 64, (2, 9)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, (2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, (2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, (2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, (2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(1)
        )
        self.two_clf = nn.Linear(15, 3)

        self.three_patterns = nn.Sequential(
            nn.Conv2d(1, 16, (3, 9)),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, (3, 1)),
            nn.GELU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, (3, 1)),
            nn.GELU(),
            nn.BatchNorm2d(1)
        )
        self.three_clf = nn.Linear(14, 3)

        self.five_patterns = nn.Sequential(
            nn.Conv2d(1, 8, (5, 9)),
            nn.GELU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, (5, 1)),
            nn.GELU(),
            nn.BatchNorm2d(1)
        )
        self.five_clf = nn.Linear(12, 3)

        self.seven_patterns = nn.Sequential(
            nn.Conv2d(1, 8, (7, 9)),
            nn.GELU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, (7, 1)),
            nn.GELU(),
            nn.BatchNorm2d(1)
        )
        self.seven_clf = nn.Linear(8, 3)
        
        self.out = nn.Linear(12, 3)

    def forward(self, x):
        branch_2 = self.two_clf(self.two_patterns(x).squeeze())
        branch_3 = self.three_clf(self.three_patterns(x).squeeze())
        branch_5 = self.five_clf(self.five_patterns(x).squeeze())
        branch_7 = self.seven_clf(self.seven_patterns(x).squeeze())

        return self.out(torch.cat((branch_2, branch_3, branch_5, branch_7), dim=0))


def preprocess(batch_df: list) -> torch.Tensor:
    minibatches_count = batch_df.timestamp.apply(dt.fromisoformat).apply(dt.date).nunique()

    batch_df['body'] = (batch_df.close - batch_df.open).round(1)
    batch_df['gap'] = (batch_df.open - batch_df.close.shift()).round(1)

    batch_df['gap'] = batch_df.gap.fillna(0)

    batch_df['upper_shadow'] = np.minimum(batch_df.high - batch_df.open, batch_df.high - batch_df.close).round(1)
    batch_df['lower_shadow'] = np.minimum(batch_df.open - batch_df.low, batch_df.close - batch_df.low).round(1)

    batch_df['volume'] = batch_df.volume.round(-2)

    batch_df['timestamp'] = batch_df.timestamp.apply(dt.fromisoformat)

    batch_df['weekday'] = batch_df.timestamp.apply(lambda x: x.weekday())

    batch_df['hour'] = batch_df.timestamp.apply(lambda x: x.hour)

    batch_df['hour_sin'] = np.sin(2 * np.pi * (batch_df.hour - 7) / 8)
    batch_df['hour_cos'] = np.cos(2 * np.pi * (batch_df.hour - 7) / 8)

    batch_df['weekday_sin'] = np.sin(2 * np.pi * batch_df.weekday / 4)
    batch_df['weekday_cos'] = np.cos(2 * np.pi * batch_df.weekday / 4)

    batch_df.drop(columns=['weekday', 'hour', 'timestamp', 'open', 'close', 'low', 'high'], inplace=True)

    batch_array = np.array(batch_df)
    batch_split_array = np.array(np.split(batch_array, minibatches_count))

    batch_tensor = torch.tensor(batch_split_array, dtype=torch.float32)

    return batch_tensor

load_dotenv()

TOKEN = os.getenv('READONLY_TOKEN')
FASTAPI_ADDR = os.getenv('FASTAPI_ADDR')

common_url = 'https://invest-public-api.tinkoff.ru:443/rest'

shares = {'SBER': {'ticker': 'SBER',
                   'figi': 'BBG004730N88',
                   'lot': 10,
                    'first': date(2018, 3, 7)}}

header = {'Authorization': TOKEN,
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0'}

share = shares['SBER']

params = {
    'date_from': share['first'].isoformat(),
    'date_to': dt.now().date().isoformat()
}

resp = requests.post(f'{FASTAPI_ADDR}/candles', json=params)

raw_historical_data = resp.json()

print('Получены исторические данные')

raw_historical_data_df = pd.DataFrame(raw_historical_data)

processed_historical_data = preprocess(raw_historical_data_df)

ds = CandlesDataset(processed_historical_data, 20)

print('Сформирован датасет')

train_ds = Subset(ds, range(int(len(ds) * 0.8)))
train_data, val_data = random_split(train_ds, [0.7, 0.3])

test_ds = Subset(ds, range(int(len(ds) * 0.8), int(len(ds))))

d_train = DataLoader(train_data, batch_size=1)
d_val = DataLoader(val_data, batch_size=1)
d_test = DataLoader(test_ds, batch_size=1)

sluggard = Sluggard()

epochs = 30

sluggard.train()

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(sluggard.parameters(), lr=0.01, weight_decay=0.001)

train_losses = []
val_losses = []

for _ in range(epochs):
    current_train_losses = []
    current_val_losses = []

    for x_train, y_train in d_train:
        predict = sluggard(x_train.unsqueeze(0))
        loss = loss_fn(predict.squeeze(), y_train.squeeze())
        current_train_losses.append(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        for x_train, y_train in d_val:
            predict = sluggard(x_train.unsqueeze(0))
            loss = loss_fn(predict.squeeze(), y_train.squeeze())
            current_val_losses.append(loss)

    train_loss = torch.tensor(current_train_losses).mean()
    train_losses.append(train_loss)
    val_loss = torch.tensor(current_val_losses).mean()
    val_losses.append(val_loss)


def get_daily_batch(batch_date: date):
    open_date = dt.combine(batch_date, time(7,0,0))
    close_date = dt.combine(batch_date, time(15,39,0))

    params = {
        'date_from': open_date.isoformat() + 'Z',
        'date_to': close_date.isoformat() + 'Z'
    }

    resp = requests.post(f'{FASTAPI_ADDR}/candles', json=params) 
    daily_candles = resp.json()

    daily_df = pd.DataFrame(daily_candles)

    daily_tensor = preprocess(daily_candles)
    daily_ds = CandlesDataset(daily_tensor)

    daily_refiner = DataLoader(daily_ds, batch_size=1)

    sluggard.train()
    for x_ref, y_ref in daily_refiner:
        predict = sluggard(x_ref.unsqueeze(0))
        loss = loss_fn(predict.squeeze(), y_ref.squeeze())

        opt.zero_grad()
        loss.backward()
        opt.step()


def get_batch_for_prediction():
    current_date = dt.now()
    lagged_date = dt.now() - td(minutes=20)

    open_date = dt.combine(current_date.date(), time(7,0,0))
    close_date = dt.combine(current_date.date(), time(15,39,0))

    if current_date >= open_date and current_date <= close_date:
        params = {
        "figi": share['figi'],
        "from": lagged_date.isoformat() + 'Z',
        "to": current_date.isoformat() + 'Z',
        "interval": "CANDLE_INTERVAL_1_MIN",
        "instrumentId": share['figi'],
        "candleSourceType": "CANDLE_SOURCE_EXCHANGE",
        "limit": 519
        }
        resp = requests.post(f'{common_url}/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles')
        
        pred_batch = resp.json()
        pred_df = pd.DataFrame(pred_batch)
        pred_tensor = preprocess(pred_df)

        sluggard.eval()

        inference = sluggard(pred_tensor.unsqueeze(0))
        readable_labels = ['BUY', 'HOLD', 'SELL']

        print(share['ticker'], readable_labels[inference.argmax().item()])
        


try:
    scheduler = BlockingScheduler()

    main_session_end = CronTrigger(hour=15, minute=41)
    every_minute = CronTrigger(second=0)

    scheduler.add_job(get_daily_batch, main_session_end, kwargs={'batch_date': dt.now().date()})
    scheduler.add_job(get_batch_for_prediction, every_minute)

except KeyboardInterrupt:
    scheduler.shutdown()