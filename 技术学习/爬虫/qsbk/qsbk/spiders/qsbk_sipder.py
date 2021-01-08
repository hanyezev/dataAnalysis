import scrapy
from qsbk.items import QsbkItem


class QsbkSipderSpider(scrapy.Spider):
    name = 'qsbk_spider'
    allowed_domains = ['qiushibaike.com']
    start_urls = ['https://www.qiushibaike.com/text/page/1/']
    base_domain = "https://www.qiushibaike.com"

    def parse(self, response):
        duanzidivs = response.xpath("//div[@class='col1 old-style-col1']/div")
        for duanzidiv in duanzidivs:
            author = duanzidiv.xpath(".//h2/text()").get().strip()
            content = duanzidiv.xpath(
                ".//div[@class='content']/span/text()").getall()
            content = "".join(content).strip()
            # duanzi = {"author": author, "content":content}
            item = QsbkItem(author=author, content=content)
            yield item
        next_url = response.xpath("//ul[@class='pagination']/li[last()]/a/@href").get()
        print(next_url)
        if not next_url:
            return
        else:
            yield scrapy.Request(self.base_domain+next_url, callback=self.parse)
