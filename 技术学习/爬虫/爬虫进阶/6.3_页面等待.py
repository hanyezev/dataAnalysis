from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# driver_path = r"F:\ChromeDriver\chromedriver.exe"
# driver = webdriver.Chrome(executable_path=driver_path)
# driver.get('http://www.baidu.com/')

# WebDriverWait(driver, 10).until(
#     EC.presence_of_all_elements_located((By.ID, "affasfasf"))
# )

# 切换页面
# driver.execute_script("window.open('https://www.douban.com/')")
# print(f'driver当前页面:{driver.current_url}')
# print(driver.window_handles)
# driver.switch_to.window(driver.window_handles[1])
# print(f'driver当前页面:{driver.current_url}')

# 设置代理ip
# driver_path = r"F:\ChromeDriver\chromedriver.exe"
# options = webdriver.ChromeOptions()
# options.add_argument("--proxy-server=http://175.42.68.154:9999")
# driver = webdriver.Chrome(executable_path=driver_path, options=options)
# driver.get('http://httpbin.org/ip')





