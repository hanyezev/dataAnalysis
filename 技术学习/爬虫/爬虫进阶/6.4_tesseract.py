import pytesseract
from PIL import Image
from urllib import request
import time

# img = Image.open(r'C:\Users\16526\Desktop\test.png')
# code = pytesseract.image_to_string(img, lang='chi_sim+eng')
# print(code)

def main():
    url = "https://ca.csu.edu.cn/authserver/getCaptcha.htl?1609658067959"
    while True:
        request.urlretrieve(url, 'captcha.png')
        img = Image.open(r'captcha.png')
        code = pytesseract.image_to_string(img, lang="eng")
        print(code)
        time.sleep(2)


if __name__ == '__main__':
    main()






