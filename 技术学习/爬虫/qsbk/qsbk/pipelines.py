# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json


# class QsbkPipeline:
#     def __init__(self):
#         self.fp = open("duanzi.json", "w",encoding="UTF-8")
#
#     def open_spider(self, spider):
#         print("爬虫开始了...")
#
#     def process_item(self, item, spider):
#         item_json = json.dumps(dict(item), ensure_ascii=False)
#         self.fp.write(item_json+"\n")
#         return item
#
#     def close_spider(self,spider):
#         self.fp.close()
#         print("爬虫结束了...")


# from scrapy.exporters import JsonItemExporter, JsonLinesItemExporter
# class QsbkPipeline:
#     def __init__(self):
#         self.fp = open("duanzi.json", "wb")
#         self.exporter = JsonItemExporter(self.fp, ensure_ascii=False, encoding="UTF-8")
#         self.exporter.start_exporting()
#
#     def open_spider(self, spider):
#         print("爬虫开始了...")
#
#     def process_item(self, item, spider):
#         self.exporter.export_item(item)
#         return item
#
#     def close_spider(self, spider):
#         self.exporter.finish_exporting()
#         self.fp.close()
#         print("爬虫结束了...")


from scrapy.exporters import JsonLinesItemExporter
class QsbkPipeline:
    def __init__(self):
        self.fp = open("duanzi.json", "wb")
        self.exporter = JsonLinesItemExporter(self.fp, ensure_ascii=False, encoding="UTF-8")

    def open_spider(self, spider):
        print("爬虫开始了...")

    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item

    def close_spider(self, spider):
        self.fp.close()
        print("爬虫结束了...")
