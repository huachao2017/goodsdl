import os
import pandas as pd

from surprise import SVD
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import django
import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from goodscf.models import Goods, Payment, PaymentGoods, UserGoods
from django.db import connection

# Load the user_item payment dataset
# Creation of the dataframe. Column names are irrelevant.

def get_rating(count):
    # TODO need add time decay
    if count > 0 and count <= 1:
        ret = 1
    elif count > 1 and count <= 2:
        ret = 2
    elif count > 2 and count <= 5:
        ret = 3
    elif count > 5 and count <= 10:
        ret = 4
    elif count >10:
        ret = 5
    else:
        ret = 0
    return ret

ground_user_goods = {}
with connection.cursor() as cursor:
    cursor.execute('select p.openid,pg.goods_id,count(1) as cnt from goodscf_paymentgoods as pg LEFT JOIN goodscf_payment as p ON pg.payment_id=p.id group by pg.goods_id,p.openid')
    results = cursor.fetchall()
    users = [row[0] for row in results]
    goods = [row[1] for row in results]
    rating = [get_rating(row[2]) for row in results]
    for row in results:
        ground_user_goods[row[0] + '-' + str(row[1])] = get_rating(row[2])

ratings_dict = {'itemID': goods,
                'userID': users,
                'rating': rating}
df = pd.DataFrame(ratings_dict)

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, max(rating)))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# We'll use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


with connection.cursor() as cursor:
    cursor.execute('select distinct openid from goodscf_payment order by openid')
    results = cursor.fetchall()
    all_users = [row[0] for row in results]

with connection.cursor() as cursor:
    cursor.execute('select distinct goods_id from goodscf_paymentgoods order by goods_id')
    results = cursor.fetchall()
    all_goods = [row[0] for row in results]

UserGoods.objects.all().delete()
for i in range(len(all_users)):
    for j in range(len(all_goods)):
        pred = algo.predict(all_users[i], all_goods[j])
        r_ui = 0
        key = all_users[i]+'-'+str(all_goods[j])
        if key in ground_user_goods:
            r_ui = ground_user_goods[key]

        UserGoods.objects.create(
            openid = pred.uid,
            goods_id = pred.iid,
            r_ui = r_ui,
            est = pred.est
        )