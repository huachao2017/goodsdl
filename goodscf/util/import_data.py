import os
import pandas as pd
import django
import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from goodscf.models import Goods, Payment, PaymentGoods

goods_data= pd.read_csv('../data/goods_0703.csv')
Goods.objects.all().delete()
for goods in goods_data.values:
    Goods.objects.create(
        goods_code=goods[0],
        goods_name=goods[1],
        price=int(float(goods[10])*100),
        upc=goods[11],
    )

PaymentGoods.objects.all().delete()
Payment.objects.all().delete()

shopid = 587
for filename in os.listdir('../data/payment/'):
    payment_data = pd.read_csv('../data/payment/' + filename)
    count = 0
    for one_payment in payment_data.values:
        if int(one_payment[4]) == shopid:
            count += 1
            payment = Payment.objects.create(
                openid=one_payment[0],
                orderid=one_payment[1],
                order_amount=int(one_payment[2]),
                pay_amount=int(one_payment[3]),
                shop_id=int(one_payment[4]),
                pay_time=datetime.datetime.strptime(one_payment[5], "%Y-%m-%d %H:%M:%S"),
            )
            payment_goods = one_payment[6].split(';')
            for goods_code in payment_goods:
                if goods_code != '':
                    try:
                        goods = Goods.objects.get(goods_code=int(goods_code))
                    except Goods.DoesNotExist:
                        # print(int(goods_code))
                        goods = Goods.objects.create(
                            goods_code=int(goods_code),
                            goods_name='unknown',
                            price=0,
                            upc='unknown',
                        )
                    PaymentGoods.objects.create(
                        payment_id=payment.pk,
                        goods_id=goods.pk,
                    )
    print(filename + ':' + str(count))