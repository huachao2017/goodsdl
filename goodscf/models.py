from django.db import models

# Create your models here.
class Payment(models.Model):
    openid = models.CharField(max_length=50, default='', db_index=True)
    orderid = models.CharField(max_length=20, default='', unique=True)
    order_amount = models.IntegerField()
    pay_amount = models.IntegerField()
    shop_id = models.IntegerField(db_index=True)

    pay_time = models.DateTimeField('pay date')

class Goods(models.Model):
    goods_code = models.IntegerField(unique=True)
    goods_name = models.CharField(max_length=50)
    price = models.IntegerField()
    upc = models.CharField(max_length=20, default='')

class PaymentGoods(models.Model):
    payment = models.ForeignKey(Payment, related_name="goodss", on_delete=models.CASCADE)
    goods = models.ForeignKey(Goods, related_name="payments", on_delete=models.CASCADE )
