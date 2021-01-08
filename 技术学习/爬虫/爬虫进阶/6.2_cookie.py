from selenium import webdriver


driver_path = r"F:\ChromeDriver\chromedriver.exe"
driver = webdriver.Chrome(executable_path=driver_path)
driver.get('http://www.baidu.com/')

for cookie in driver.get_cookies():
    print(cookie)

driver.delete_cookie("BD_UPN")
driver.delete_all_cookies()
print("***********")

print(driver.get_cookie("BD_UPN"))