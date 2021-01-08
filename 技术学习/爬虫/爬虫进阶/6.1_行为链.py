from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time

driver_path = r"F:\ChromeDriver\chromedriver.exe"
driver = webdriver.Chrome(executable_path=driver_path)
driver.get('http://www.baidu.com/')

inputTag = driver.find_element_by_id('kw')
submitBtn = driver.find_element_by_id('su')

actions = ActionChains(driver)
actions.move_to_element(inputTag)
actions.send_keys('python')
actions.move_to_element(submitBtn)
actions.click(submitBtn)
actions.perform()