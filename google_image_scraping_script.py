import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import requests
import os
from PIL import Image
import io
import hashlib
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from datetime import datetime
import timeout_decorator

# All in same directory
#DRIVER_PATH = 'chromedriver.exe'

def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        #for i in range(3):
        #wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        #scroll_pause_time = 2
        #time.sleep(scroll_pause_time) 
        #wd.execute_script("window.scrollBy(0, 1000);")  # Scroll down by 1000 pixels
        #WebDriverWait(wd, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'img[jsname="Q4LuWd"]')))
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait for new images to load or for a 'Load More' button to appear
        WebDriverWait(wd, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'img[jsname="Q4LuWd"], [jsaction="Pmjnye"]'))
        ) 
        
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    
    image_urls = set()
    image_count = 0
    results_start = 0
    error_clicks = 0
    while (image_count < max_links_to_fetch) & (error_clicks < 30): # error clicks to stop when there are no more results to show by Google Images. You can tune the number
        scroll_to_end(wd)

        print('Starting search for Images')

        # get all image thumbnail results
        #thumbnail_results = wd.find_elements(By.CSS_SELECTOR, 'img[jsname="Q4LuWd"].rg_i Q4LuWd')
        thumbnail_results = wd.find_elements(By.CSS_SELECTOR, 'img[jsname="Q4LuWd"]')
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        for img in thumbnail_results[results_start:max_links_to_fetch]:
            # try to click every thumbnail such that we can get the real image behind it
            print("Total Errors till now:", error_clicks)
            try:
                print('Trying to Click the Image')
                img.click()
                WebDriverWait(wd, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'img[jsname="kn3ccd"]'))
                )
                #time.sleep(sleep_between_interactions)
                print('Image Click Successful!')
            except Exception:
                error_clicks = error_clicks + 1
                print('ERROR: Unable to Click the Image')
                if(results_start < number_results):
                	continue
                else:
                    break
            #else: 	
                #results_start = results_start + 1

            # extract image urls    
            print('Extracting of Image URLs')
            actual_images = wd.find_elements(By.CSS_SELECTOR, 'img[jsname="kn3ccd"]')
            #for actual_image in actual_images:
                #if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    #image_urls.add(actual_image.get_attribute('src'))
            for index, img in enumerate(actual_images):
                image_urls.add(img.get_attribute('src'))
                print('Image URLS:', img.get_attribute('src'))

            image_count = len(image_urls)

            print('Current Total Image Count:', image_count)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
            else:
                try:
                    load_more_button = wd.find_element(By.CSS_SELECTOR, '[jsaction="Pmjnye"]')
                    if load_more_button:
                        wd.execute_script("arguments[0].click();", load_more_button)
                except Exception as e:
                    print("Load more button not found or not clickable", e)
                    break
            	        
        results_start = len(thumbnail_results)

    return image_urls

@timeout_decorator.timeout(10) # if taking more than 1 minute then timeout
def persist_image(folder_path:str,file_name:str,url:str, target_size=(640, 480)):
    try:
        try:
            image_content = requests.get(url).content

        except Exception as e:
            print(f"ERROR - Could not download {url} - {e}")

        try:
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file).convert('RGB')
            #image = image.resize(target_size)
            folder_path = os.path.join(folder_path,file_name)
            if os.path.exists(folder_path):
                file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
            else:
                try:
                    os.makedirs(folder_path, exist_ok=True)
                    file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
                except Exception as e:
                    print(f"ERROR - Could not create directory {folder_path} - {e}")
            with open(file_path, 'wb') as f:
                image.save(f, "JPEG", quality=85)
            print(f"SUCCESS - saved {url} - as {file_path}")
        except Exception as e:
            print(f"ERROR - Could not save {url} - {e}")
    except TimeoutError:
        print('Timeout!')

if __name__ == '__main__':
    #wd = webdriver.Chrome(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36")
    wd = webdriver.Chrome(options=options)
    queries = ["toyota camry 2010 exterior"]  #change your set of queries here
    for query in queries:
        wd.get('https://google.com')
        search_box = wd.find_element(By.CSS_SELECTOR, 'textarea[jsname="yZiJbe"].gLFyf')
        #search_box = wd.find_element_by_css_selector('input.gLFyf')
        search_box.send_keys(query)
        links = fetch_image_urls(query,15,wd) # 500 denotes no. of images you want to download
        images_path = 'dataset/'
        
        for i in links:
            persist_image(images_path,query,i)
    wd.quit()
