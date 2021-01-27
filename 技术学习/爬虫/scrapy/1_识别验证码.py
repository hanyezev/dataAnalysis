from urllib import request
from base64 import b64encode
import requests

cap_url = "http://icode.renren.com/getcode.do?t=web_login&rnd=0.7187561132518416"

request.urlretrieve(cap_url,"captcha.png")

recognize_url = "http://aiocr.market.alicloudapi.com/ocr/v1"

formdata = {}
with open("captcha.png", 'rb') as fp:
    data = fp.read()
    pic = b64encode(data)
    formdata["pic"] = pic

appcode = "8fd364dfba6d4ee0a8631359fa3da7cd"
headers = {
    'Content-Type': 'application/json; charset=UTF-8',
    'Authorization': 'APPCODE ' + appcode
}

res = requests.post(recognize_url, data=formdata, headers=headers)
print(res.json())

