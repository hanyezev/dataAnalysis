import os
from queue import Queue
import time
import threading
import re
import requests
from lxml import etree
from urllib import request

class Producer(threading.Thread):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
        'Referer': "https://www.doutula.com/"
    }
    def __init__(self, page_queue, img_queue, *args, **kwargs):
        super(Producer, self).__init__(*args, **kwargs)
        self.page_queue = page_queue
        self.img_queue = img_queue

    def run(self):
        while True:
            if self.page_queue.empty():
                break
            url = self.page_queue.get()
            self.parse_url(url)

    def parse_url(self, url):
        res = requests.get(url, headers=self.headers)
        text = res.text
        html = etree.HTML(text)
        imgs = html.xpath("//div[@class='page-content text-center']//a[@class='col-xs-6 col-sm-3']/img[@class!='gif']")
        for index, img in enumerate(imgs):
            img_url = img.get('data-original')
            alt = img.get('alt')
            alt = re.sub(r"[\?？\.,，。！!/|]", "", alt)
            suffix = os.path.splitext(img_url)[1]
            filename = alt + suffix
            if alt == "":
                filename = img_url.split("/")[-1]
            # print(str(index) + ": " + filename)
            self.img_queue.put((img_url, filename))

class Consumer(threading.Thread):
    def __init__(self, page_queue, img_queue, *args, **kwargs):
        super(Consumer, self).__init__(*args, **kwargs)
        self.page_queue = page_queue
        self.img_queue = img_queue
    def run(self):
        while True:
            if self.img_queue.empty() and self.page_queue.empty():
                break
            img_url, filename = self.img_queue.get()
            request.urlretrieve(img_url, 'images/' + filename)
            print(filename + "下载完成!")

def main():
    page_queue = Queue(100)
    img_queue = Queue(1000)
    for i in range(1, 10):
        url = f"https://www.doutula.com/photo/list/?page={i}"
        page_queue.put(url)

    for x in range(5):
        t = Producer(page_queue, img_queue)
        t.start()

    for x in range(5):
        t = Consumer(page_queue, img_queue)
        t.start()

if __name__ == "__main__":
    main()
