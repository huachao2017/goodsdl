from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from collections import OrderedDict, namedtuple

class MyPageNumberPagination(PageNumberPagination):

  def get_paginated_response(self, data):
    return Response(OrderedDict([
      ('count', self.page.paginator.count), # 总数量
      ('next', self.get_next_link()),       # 下一页链接
      ('cur_number', self.page.number),    # 当前页数
      ('previous', self.get_previous_link()), # 前一页链接
      ('results', data)                       # 数据
    ]))
