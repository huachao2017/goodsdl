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

from goodscf.models import Goods, Payment, PaymentGoods
from django.db import connection

# Load the user_item payment dataset
# Creation of the dataframe. Column names are irrelevant.

with connection.cursor() as cursor:
    cursor.execute('select p.openid,pg.goods_id,count(1) as cnt from goodscf_paymentgoods as pg LEFT JOIN goodscf_payment as p ON pg.payment_id=p.id group by pg.goods_id,p.openid')
    results = cursor.fetchall()
    users = [row[0] for row in results]
    goods = [row[1] for row in results]
    rating = [row[2] for row in results]

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