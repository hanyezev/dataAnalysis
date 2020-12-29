import os
from queue import Queue
import time
import threading
import re
import requests
from lxml import etree
from urllib import request

def parse_url(url):
    headers = {
        'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
        'Referer':"https://www.doutula.com/"
    }
    res = requests.get(url, headers=headers)
    text = res.text
    html = etree.HTML(text)
    imgs = html.xpath("//div[@class='page-content text-center']//a[@class='col-xs-6 col-sm-3']/img[@class!='gif']")
    for index,img in enumerate(imgs):
        img_url = img.get('data-original')
        alt = img.get('alt')
        alt = re.sub(r"[\?？\.,，。！!/||]","",alt)
        suffix = os.path.splitext(img_url)[1]
        filename = alt + suffix
        if alt == "":
            filename = img_url.split("/")[-1]
        print(str(index) + ": "  + filename)
        request.urlretrieve(img_url, 'images/'+filename)

def main():
    for i in range(1, 101):
        url = f"https://www.doutula.com/photo/list/?page={i}"
        parse_url(url)

if __name__ == "__main__":
    main()
