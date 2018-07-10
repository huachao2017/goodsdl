import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import django
import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from goodscf.models import Goods, Payment, PaymentGoods
from django.db import connection


# users = Payment.objects.order_by('openid').values_list('openid', flat=True).distinct()
# print(users)
# goods = PaymentGoods.objects.order_by('goods_id').values_list('goods_id', flat=True).distinct()
# print(goods)

with connection.cursor() as cursor:
    cursor.execute('select distinct openid from goodscf_payment order by openid')
    results = cursor.fetchall()
    users = [row[0] for row in results]
x_names = list(range(len(users)))

with connection.cursor() as cursor:
    cursor.execute('select distinct goods_id from goodscf_paymentgoods order by goods_id')
    results = cursor.fetchall()
    goods = [row[0] for row in results]
    print(goods)
y_names = list(range(len(goods)))

data = np.zeros((len(users),len(goods)))
normalize_data = np.zeros((len(users),len(goods)))
with connection.cursor() as cursor:
    cursor.execute('select p.openid,pg.goods_id,count(1) as cnt from goodscf_paymentgoods as pg LEFT JOIN goodscf_payment as p ON pg.payment_id=p.id group by pg.goods_id,p.openid')
    results = cursor.fetchall()
    for row in results:
        u_index = users.index(row[0])
        g_index = goods.index(row[1])
        data[u_index][g_index] = row[2]
        normalize_data[u_index][g_index] = 1
    print(data)


# plt.imshow(data)

# df = pd.DataFrame({'A':np.random.randint(1, 100, 5),'B':np.random.randint(1, 100, 5),'C':np.random.randint(1, 100, 5)})
#
# x_names=['A','B','C']
# y_names=['0','1','2','3','4']
#
# data = df.values

def plot(data, x_names, y_names):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(data, vmin=0, vmax=np.max(data))
    fig.colorbar(cax)
    # x_ticks = np.arange(0,len(x_names),1)
    # y_ticks = np.arange(0,len(y_names),1)
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(y_ticks)

    # ax.set_xticklabels(x_names)
    #
    # ax.set_yticklabels(y_names)

    plt.show()

def normlize_plot(data):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(data, vmin=0, vmax=1)
    fig.colorbar(cax)
    # x_ticks = np.arange(0,len(x_names),1)
    # y_ticks = np.arange(0,len(y_names),1)
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(y_ticks)

    # ax.set_xticklabels(x_names)
    #
    # ax.set_yticklabels(y_names)

    plt.show()


plot(data,x_names,y_names)
normlize_plot(normalize_data)