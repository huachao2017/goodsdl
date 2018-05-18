import requests
from lxml import html
import json
import math
import csv

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
        self.session_requests = requests.session()
        self._login()
        self.list_page_infos = {}
        self._get_all_list_page_info()
        self.f_csv = None

    def _login(self):
        # Get login csrf token
        print(LOGIN_URL)
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
            type = type_div.xpath("a/p/@data-val")
            type = int(type[0].strip())
            type_name = type_div.xpath("a/p/span[2]/text()")
            type_name = type_name[0].strip()
            self.list_page_infos[type] = {'url': url,
                                          'name': type_name,
                                          'total_num': 0,
                                          'page_size': 50}

        url = LIST_PAGE_URL
        print(url)
        for type in self.list_page_infos:
            # Create form_data
            form_data = {
                "brand_id": 0,
                "cate_id": type,
                "sort": 1,
                "p": 1
            }
            result = self.session_requests.post(url, data=form_data,
                                                headers=dict(referer=DOMAIN + self.list_page_infos[type]['url']))
            content = result.content.decode('utf-8')
            json_content = json.loads(content)
            # print(json_content)
            tree = html.fromstring(json_content)
            page_info = tree.xpath("//div[@id='pager_info']")
            self.list_page_infos[type]['total_num'] = int(page_info[0].attrib['data-total'])
            self.list_page_infos[type]['page_size'] = int(page_info[0].attrib['data-ps'])

        print(self.list_page_infos)

    def _search_list_page(self, type, page_num):
        url = LIST_PAGE_URL
        print(url, type, page_num)
        # Create form_data
        form_data = {
            "brand_id": 0,
            "cate_id": type,
            "sort": 1,
            "p": page_num
        }

        result = self.session_requests.post(url, data=form_data,
                                            headers=dict(referer=DOMAIN + self.list_page_infos[type]['url']))
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
            upc, type = self._get_goods_upc_and_type(goods_id, goods_url)
            goods = {'id': goods_id,
                     'type': type,
                     'name': name,
                     'upc': upc,
                     'norm': norm,
                     'price': price,
                     'unit': unit,
                     'purchasenum': purchasenum,
                     'img': goods_img,
                     }
            self.f_csv.writerow([goods_id, type, name, upc, norm, price, unit, purchasenum, goods_img])
            print(goods)

    def _get_goods_upc_and_type(self, goods_id, goods_url):
        url = DOMAIN + goods_url
        print(url)
        result = self.session_requests.get(url, headers=dict(referer=MAIN_URL))
        tree = html.fromstring(result.content)
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
        for type in self.list_page_infos:
            page_num = math.ceil(self.list_page_infos[type]['total_num'] / self.list_page_infos[type]['page_size'])
            for i in range(page_num):
                self._search_list_page(type, i + 1)

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
