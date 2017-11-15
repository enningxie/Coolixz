# coding=utf-8
import urllib2


def linkBaidu():
    url = "http://www.baidu.com"
    try:
        response = urllib2.urlopen(url, timeout=3)
    except urllib2.URLError:
        print u"网络错误"
        exit()
    with open('./baidu.txt', 'w') as fp:
        fp.write(response.read())
    print u"获取url信息, response.geturl() \n：%s" % response.geturl()
    print u"获取返回代码, response.getcode() \n：%s" % response.getcode()
    print u"获取返回信息, response.info() \n：%s" % response.info()


if __name__ == '__main__':
    linkBaidu()