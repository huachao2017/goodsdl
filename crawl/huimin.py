import requests
from lxml import html
import json
import math
import csv
import time

USERNAME = "13146266067"
PASSWORD = "266067"

DOMAIN = "https://pcshop.huimin100.cn"
LOGIN_URL = DOMAIN + "/index.php/home/login/index.html"
LOGIN_POST_URL = DOMAIN + "/index.php/home/login/login.html"
MAIN_URL = DOMAIN + "/index.php/home/index/index.html"
# LIST_URL = DOMAIN + "/index.php/home/index/categorylist/cateid/{}/pid/{}.html"
LIST_PAGE_URL = DOMAIN + "/index.php/home/index/prodlist.html"


class HuiMin:
    def __init__(self, file_path):
        self.file_path = file_path
        self.session_requests = None
        self._login()
        self.list_page_infos = {}
        self._get_all_list_page_info()
        self.f_csv = None

    def _login(self):
        # Get login csrf token
        print(LOGIN_URL)
        self.session_requests = requests.session()
        result = self.session_requests.get(LOGIN_URL)
        tree = html.fromstring(result.text)

        # Create logindata
        logindata = {
            "supermarketId": USERNAME,
            "pwd": PASSWORD,
        }

        # Perform login
        print(LOGIN_URL)
        result = self.session_requests.post(LOGIN_POST_URL, data=logindata, headers=dict(referer=LOGIN_URL))
        if result.status_code != 200:
            raise ValueError('login error')

    def _get_all_list_page_info(self):
        print(MAIN_URL)
        result = self.session_requests.get(MAIN_URL, headers=dict(referer=LOGIN_URL))
        tree = html.fromstring(result.content)
        type_list_div = tree.xpath("//div[@class='pc_headIndex_box clearfix']/div[1]/div[1]/div[1]/div")
        for type_div in type_list_div:
            url = type_div.xpath("a/@href")
            url = url[0].strip()
            cat = type_div.xpath("a/p/@data-val")
            cat = int(cat[0].strip())
            type_name = type_div.xpath("a/p/span[2]/text()")
            type_name = type_name[0].strip()
            self.list_page_infos[cat] = {'url': url,
                                          'name': type_name,
                                          'total_num': 0,
                                          'page_size': 50}

        url = LIST_PAGE_URL
        print(url)
        for cat in self.list_page_infos:
            # Create form_data
            form_data = {
                "brand_id": 0,
                "cate_id": cat,
                "sort": 1,
                "p": 1
            }
            result = self.session_requests.post(url, data=form_data,
                                                headers=dict(referer=DOMAIN + self.list_page_infos[cat]['url']))
            content = result.content.decode('utf-8')
            json_content = json.loads(content)
            # print(json_content)
            tree = html.fromstring(json_content)
            page_info = tree.xpath("//div[@id='pager_info']")
            self.list_page_infos[cat]['total_num'] = int(page_info[0].attrib['data-total'])
            self.list_page_infos[cat]['page_size'] = int(page_info[0].attrib['data-ps'])

        print(self.list_page_infos)

    def _search_list_page(self, cat, page_num):
        url = LIST_PAGE_URL
        print(url, cat, page_num)
        # Create form_data
        form_data = {
            "brand_id": 0,
            "cate_id": cat,
            "sort": 1,
            "p": page_num
        }

        result = self.session_requests.post(url, data=form_data,
                                            headers=dict(referer=DOMAIN + self.list_page_infos[cat]['url']))
        content = result.content.decode('utf-8')
        json_content = json.loads(content)
        # print(json_content)
        tree = html.fromstring(json_content)
        goods_div_list = tree.xpath("//div[@class='goods_item shenceView']")
        for goods_div in goods_div_list:
            goods_id = int(goods_div.attrib['data-pid'])
            price = float(goods_div.attrib['data-curprice'])
            name = goods_div.attrib['data-name']
            try:
                unit = goods_div.attrib['data-unit']
            except:
                unit = ''
            try:
                norm = goods_div.attrib['data-norm']
            except:
                norm = ''
            try:
                purchasenum = goods_div.attrib['data-purchasenum']
            except:
                purchasenum = ''
            goods_url = goods_div.xpath('div[1]/a/@href')
            goods_url = goods_url[0].strip()
            goods_img = goods_div.xpath('div[1]/a/img/@src')
            goods_img = goods_img[0].strip()
            upc, goods_type = self._get_goods_upc_and_type(goods_id, goods_url, referer=DOMAIN + self.list_page_infos[cat]['url'])
            goods = {'id': goods_id,
                     'type': goods_type,
                     'name': name,
                     'upc': upc,
                     'norm': norm,
                     'price': price,
                     'unit': unit,
                     'purchasenum': purchasenum,
                     'img': goods_img,
                     }
            self.f_csv.writerow([goods_id, goods_type, name, upc, norm, price, unit, purchasenum, goods_img])
            print(goods)
            # 定期休息
            time.sleep(0.5)

    def _get_goods_upc_and_type(self, goods_id, goods_url, referer=MAIN_URL):
        url = DOMAIN + goods_url
        print(url)
        result = None
        try:
            result = self.session_requests.get(url, headers=dict(referer=referer), timeout=5)
        except:
            self._login()
            for i in range(1, 5):
                print('请求超时，第 % s次重复请求' % i)
                try:
                    result = self.session_requests.get(url, headers=dict(referer=referer), timeout=5)
                    if result.status_code == 200:
                        break
                except Exception as e:
                    print(e)
        if result is None:
            return '',''
        content = result.content.decode('utf-8')
        tree = html.fromstring(content)
        detail_text_list = tree.xpath("//div[@class='goods_detail_box']/p/text()")
        upc = ''
        type = ''
        for detail_text in detail_text_list:
            detail_text = str(detail_text)
            if detail_text.startswith('条形码：'):
                upc = detail_text.replace('条形码：', '')
            if detail_text.startswith('分类：'):
                type = detail_text.replace('分类：', '')

        return upc, type

    def run_search(self):
        csvfile = open(self.file_path, 'w')
        self.f_csv = csv.writer(csvfile)
        self.f_csv.writerow(['id', '分类', '名称', 'upc', '规格', '价格', '单位', '起卖量', '图片'])
        for cat in self.list_page_infos:
            page_num = math.ceil(self.list_page_infos[cat]['total_num'] / self.list_page_infos[cat]['page_size'])

            # 定期重新登录
            self._login()
            for i in range(page_num):
                self._search_list_page(cat, i + 1)

            # 定期休息
            time.sleep(5)

        csvfile.close()

    def test(self):
        # Scrape url
        result = self.session_requests.get(MAIN_URL, headers=dict(referer=LOGIN_URL))
        tree = html.fromstring(result.content)
        one_price = tree.xpath("//div[@class='homeMain-nav clearfix']/div[1]/div[5]/div[2]/div[1]/div[2]/p[4]/i/text()")
        one_price = one_price[0].strip()

        print(one_price)


if __name__ == '__main__':
    huimin = HuiMin('c:/fastbox/1.csv')

    huimin.run_search()
