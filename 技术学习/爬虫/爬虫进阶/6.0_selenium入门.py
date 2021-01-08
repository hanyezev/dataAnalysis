from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time

driver_path = r"F:\ChromeDriver\chromedriver.exe"
driver = webdriver.Chrome(executable_path=driver_path)
driver.get('http://www.baidu.com/')

# time.sleep(5)
# # driver.quit()
# input = driver.find_element_by_id('kw')
# input.send_keys('python')
# submit = driver.find_element_by_id('su')
# submit.click()

# find_element_by_name
# find_element_by_xpath
# find_element_by_link_text
# find_element_by_partial_link_text
# find_element_by_tag_name
# find_element_by_class_name
# find_element_by_css_selector

# selectBtn = Select(driver.find_element(By.ID,'kw'))
# # input.send_keys('python')
# selectBtn.select_by_visible_text("选项文本")
# selectBtn.deselect_all()