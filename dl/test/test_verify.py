import os
import tensorflow as tf
import urllib

def main(_):
    picurl = 'http://192.168.1.172/media/images/275/20180503/105652_105652.409527_ai.jpg'
    picurl = urllib.parse.urlencode({'picurl':picurl})
    print(picurl)
    url = 'http://192.168.1.170/api/verifycnt?deviceid=0&paymentID=1&{}&goodscnt=3'.format(picurl)
    print(url)
    response = urllib.request.urlopen(url)

    print(response.read())

if __name__ == '__main__':
    tf.app.run()