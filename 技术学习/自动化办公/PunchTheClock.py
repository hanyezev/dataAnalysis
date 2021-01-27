from selenium import webdriver
import time
import smtplib
from email.mime.text import MIMEText

try:
    MAIL_USER = "18169240419@163.com"  # 用于发送通知的邮箱
    MAIL_PWD = "1137362935.zc"
    username = "194712252"
    password = "1137362935"
    accept = "1652620697@qq.com"
    delay = 5   # 延迟设置

    def mail(mail_text, mail_to):
        # set the mail context
        msg = MIMEText(mail_text)
        # set the mail info
        msg['Subject'] = "打卡"
        msg['From'] = MAIL_USER
        msg['To'] = mail_to
        # send the mail
        send = smtplib.SMTP_SSL("smtp.163.com", 465)
        send.login(MAIL_USER, MAIL_PWD)
        send.send_message(msg)
        # quit QQ EMail
        send.quit()

    # 这部分用来设置运行时不显示浏览器窗口
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver_path = r"F:\ChromeDriver\chromedriver.exe"
    driver = webdriver.Chrome(options=options, executable_path=driver_path)
    driver.get('http://ca.its.csu.edu.cn/Home/Login/215')
    driver.find_element_by_id("userName").send_keys(username)
    driver.find_element_by_id("passWord").send_keys(password)
    time.sleep(delay)
    login = driver.find_element_by_id('login-btn')
    login.click()
    time.sleep(delay)
    everyday = driver.find_element_by_xpath("//div[@class='content-wrap'][1]/ul/li/span")
    everyday.click()
    time.sleep(delay)
    area = driver.find_element_by_name("area")
    area.click()
    time.sleep(delay)
    submit = driver.find_element_by_xpath("//div[@class='footers']/a")
    submit.click()
    time.sleep(delay*2)
    msg = driver.find_element_by_xpath("//div[@class='wapat-title']").text
    print(msg)
    if msg == "每天只能填报一次，你已提交过":
        mail("已打卡过一次", accept)
    else:
        mail("今日打卡成功", accept)
except Exception as e:
    msg = "打卡程序错误!!!" + str(e)
    mail(msg, accept)
finally:
    driver.quit()
