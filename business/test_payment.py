import time
import tensorflow as tf
import urllib
from tradition.matcher.thread_pool import ThreadPool
import json
import uuid


def _one_request(thread_name, task_manager, count):
    global success_cnt, error_cnt, task_cnt
    url = 'http://testpay.damaijianshen.cn/order/self-service-pay.json'
    data = {"deviceId":581,
            "shopCode":"AQEUQ11AEaQc",
            "authCode":"281584212033483236",
            "identify":str(uuid.uuid1()),
            "models":[{"shopGoodsId":10,"number":1,"amount":1},{"shopGoodsId":113924799,"number":2,"amount":1},{"shopGoodsId":113924845,"number":3,"amount":1}]
            }
    data = json.dumps(data)
    data = bytes(data, 'utf8')
    print(data)
    headers = {
        # heard部分直接通过chrome部分request header部分
        # 'Accept': 'application/json, text/plain, */*',
        # 'Accept-Language': 'zh-CN,zh;q=0.8',
        # 'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.23 Mobile Safari/537.36'

    }

    # print(count)
    for i in range(count):
        # print(i)
        request = urllib.request.Request(url, headers=headers, data=data)
        result = urllib.request.urlopen(request).read()

        result = str(result, encoding="utf8")
        result = json.loads(result)
        print(thread_name + ':' + result)
        # print(task_manager.task_cnt)

        if result['status'] == 200:
            task_manager.success_cnt += 1
        else:
            task_manager.error_cnt += 1
            print('error:'+ result)
    task_manager.task_cnt += 1

class TaskManager:
    def __init__(self):
        self.thread_pool = ThreadPool(100)
        self.success_cnt = 0
        self.error_cnt = 0
        self.task_cnt = 0

    def do_all(self,task_cnt,sub_task_cnt):
        need_task_cnt = task_cnt
        for i in range(need_task_cnt):
            self.thread_pool.put(_one_request, (self,sub_task_cnt), None)

        while True:
            print("\033[32;0m任务完成(%s:成功%s,错误%s)：目前线程池中有%s个线程，空闲的线程有%s个！\033[0m"
                  % (self.task_cnt,self.success_cnt,self.error_cnt,
                     len(self.thread_pool.generate_list),
                     len(self.thread_pool.free_list)))
            if need_task_cnt == self.task_cnt:
                break
            time.sleep(0.1)

def main(_):
    time0 = time.time()
    task_manager = TaskManager()
    task_manager.do_all(1000,10)
    time1 = time.time()
    print('time:%.2f' % (time1-time0))
    # _one_request('main_thread',task_manager, 2)

if __name__ == '__main__':
    tf.app.run()