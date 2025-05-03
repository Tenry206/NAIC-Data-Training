from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import requests
import base64
import time
import os
import re

def scrollToBottom():
    lastHeight = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(3)
        try:
            driver.find_element(By.CSS_SELECTOR, ".YstHxe input").click()
            time.sleep(3)
        except:
            pass
        newHeight = driver.execute_script("return document.body.scrollHeight")
        if newHeight == lastHeight:
            break
        lastHeight = newHeight

# --- CONFIGURABLE SECTION ---
query = "ketayap cake"
dirName = "Kuih-Ketayap"
imgCount = 40

# --- WINDOWS CHROME/CHROMEDRIVER PATHS ---
chrome_path = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"  # Update if needed
chromedriver_path = "C:/Users/User/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"

chromeOptions = Options()
chromeOptions.binary_location = chrome_path
service = Service(executable_path=chromedriver_path)

driver = webdriver.Chrome(service=service, options=chromeOptions)
driver.maximize_window()

# --- GO TO GOOGLE IMAGES AND SEARCH ---
driver.get('https://images.google.com/')
box = driver.find_element(By.NAME, "q")
box.send_keys(query)
box.send_keys(Keys.ENTER)

scrollToBottom()

# --- CREATE FOLDER IF NOT EXIST ---
if not os.path.exists(dirName):
    os.makedirs(dirName)

# --- GET HIGHEST IMAGE NUMBER TO CONTINUE APPENDING ---
existing_images = [f for f in os.listdir(dirName) if f.endswith(".jpeg")]
existing_indices = [int(re.findall(r'\d+', fname)[0]) for fname in existing_images if re.findall(r'\d+', fname)]
start_index = max(existing_indices, default=0) + 1

# --- SCRAPE AND DOWNLOAD IMAGES ---
imgArr = driver.find_elements(By.CLASS_NAME, 'YQ4gaf')
print(f"Found {len(imgArr)} image elements!")

i = 0
imgIndex = start_index

while imgIndex < start_index + imgCount:
    if i >= len(imgArr):
        print("No more images found.")
        break

    # --- FILTER IMAGE BASED ON 'ketayap' IN ALT TEXT ---
    alt_text = imgArr[i].get_attribute("alt") or imgArr[i].get_attribute("aria-label") or ""
    if "ketayap" not in alt_text.lower():
        i += 1
        continue

    imgSrc = imgArr[i].get_attribute("src")
    if imgSrc:
        try:
            if imgSrc.startswith("https://encrypted-tbn0.gstatic.com/"):
                imgData = requests.get(imgSrc).content
            else:
                srcSeg = re.split(":|;|,", imgSrc)
                ext = srcSeg[1].split('/')[1]
                if ext != "jpeg":
                    i += 1
                    continue
                imgData = base64.b64decode(srcSeg[3])
            imgName = f"{dirName}/img-{imgIndex}.jpeg"
            with open(imgName, "wb") as f:
                f.write(imgData)
            print(f"{imgName} successfully written!")
            imgIndex += 1
        except Exception as e:
            print(f"Error writing image {imgIndex}: {e}")
    i += 1

driver.close()
