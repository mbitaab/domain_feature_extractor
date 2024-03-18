'''

sudo docker run -d -p 4445:4444 -e SE_NODE_MAX_SESSIONS=12 -e SE_NODE_OVERRIDE_MAX_SESSIONS=true --shm-size="2g" seleniarm/standalone-chromium:latest

sample cmd: 
python crawler_mp.py --num_procs 5 --input_file /home/ubuntu/new_drive/backupp/SAC/data/all_data_2023_08_23.json --cache_mode --output_file /home/ubuntu/new_drive/BP/saved_features/features_2023_08_23.pkl

'''


import numpy as np
import subprocess
import json
import pickle
import random
import whois
import re
import pandas as pd
import argparse
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from multiprocessing import Pool
import os
from urllib.parse import urlsplit
import IP2Location
from bs4 import BeautifulSoup
from tqdm import tqdm
import socket
import datetime
import pandas as pd
import numpy as np
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from config import *
import multiprocessing as mp

def check_url(url):
    u = urlsplit(url)
    # Check if the scheme of the URL is HTTP or HTTPS
    if u.scheme in ['http', 'https']:
        return u.netloc
    else:
        return u.path
    
def convert_to_feature(icann_data, countries, html_string, url, data_file="IP2LOCATION-LITE-DB1.BIN"):
    ip_location = IP2Location.IP2Location(data_file)
    u = check_url(url)
    X = []

    # range
    try:
        created_date = datetime.datetime.strptime(icann_data[url]['creation_date'].split(' ')[0], '%Y-%m-%d')
        exp_date = datetime.datetime.strptime(icann_data[url]['expiration_date'].split(' ')[0], '%Y-%m-%d')
        total_age = (exp_date - created_date).days / 365.0
    except:
        total_age = -1

    # set country
    country_feature = [0 for i in range(255)]
    if 'country' in icann_data[url] and icann_data[url]['country'] in countries:
        country = countries.index(
            icann_data[url]['country']
        )
        country_feature[country] = 1
    else:
        country = -1

    # whois guard is used or not
    names = icann_data[url]['registrar'] if 'registrar' in icann_data[url] else None
    if type(names) is list:
        names = names[0]

    whois_guard_keywords = ['WhoisGuard'.lower(), 'REDACTED FOR PRIVACY'.lower(), 'Private Whois'.lower(), 'DOMAIN PRIVACY'.lower()]

    for kw in whois_guard_keywords:
        if names != None and (kw in names.lower() or 'priva' in names.lower()):
            guard = 1
            break
        elif names == None:
            guard = -1
        else:
            guard = 0

    # get features from a single page
    # parse DOM tree
    parsed_tree = BeautifulSoup(html_string, 'html.parser')
    script_tags = parsed_tree.find_all('script', src=True)
    
    link_tags = parsed_tree.find_all('a', href=True)
    num_external_links = 0

    if link_tags != None:
        for link_tag in link_tags:
            link = link_tag['href'] # if 'href' in link_tag else ''
            # check if it's external link
            if link.find('http') == 0 and u not in link:
                num_external_links += 1
    
    # Shopping sites only filter: filter sites with payment
    # if 'payment' not in html_string.lower() and 'cart' not in html_string.lower():
    #     continue
    
    # check social media links
    total_social_medias = [-1, -1, -1]
    social_media_regex = [
        r'instagram\.com\/[a-zA-Z0-9_\-]+',
        r'facebook\.com\/[a-zA-Z0-9_\-]+',
        r'twitter\.com\/[a-zA-Z0-9_\-]+',
    ]
    
    for i, r in enumerate(social_media_regex):
        found = re.findall(r, html_string)
        if len(found) > 0: # and 'login' not in found[0]:
            if found[0].split('/')[-1] in url:
                total_social_medias[i] = 1
            else:
                total_social_medias[i] = 0

    # Host country
    host_country_feature = [0 for i in range(255)]
    try:
        host_ip = socket.gethostbyname(u)
        response = ip_location.get_all(host_ip)
        host_country = response.country_short.decode()
        host_domain_same = 1 if countries.index(host_country) == country else 0
        host_country = countries.index(host_country)
        host_country_feature[host_country] = 1
    except:
        host_domain_same = -1
        host_country = -1

    has_digit = False
    for i in url:
        if i.isdigit():
            has_digit = True
            break

    is_cheap = 0
    cheap_registrars = ['Namecheap', 'GoDaddy', 'Porkbun', 'NameSilo', 'Danesco', 'Hostinger']
    
    for cr in cheap_registrars:
        if 'registrar' in i and i['registrar'].lower() is not None and cr.lower() in i['registrar'].lower():
            is_cheap = 1
            break
    
    top_cheap_domains = ['club', 'buzz', 'xyz', 'ua', 'icu', 'space', 'agency', 'monster', 'pw', 'click', 'website', 'site', 'club', 'online', 'link', 'shop', 'feedback', 'uno', 'press', 'best', 'fun', 'host', 'store', 'tech', 'top', 'it']
    uses_cheap_domain = 0
    for tld in top_cheap_domains:
        if tld in url:
            uses_cheap_domain = 1
            
    # domain in text
    if parsed_tree.find('body') is not None:
        domain_in_text = parsed_tree.find('body').text.count(u)
    else:
        domain_in_text = -1
        
    # append feature vector to dataset
    domain_name = '.'.join(u.split('.')[:-1])
    
    X.append([
        guard, 
        total_social_medias[0], 
        total_social_medias[1], 
        total_social_medias[2], 
        num_external_links, 
        host_domain_same,
        len(script_tags),
        1 if '-' in url else 0,
        domain_name.count('.'),
        1 if has_digit else 0,
        is_cheap,
        uses_cheap_domain,
        domain_in_text,
        0,
        1 if u.split('.')[-1] not in ['com', 'net', 'org', 'uk', 'gov', 'au'] else 0,
        total_age,
    ])

    X[-1].extend(country_feature)
    X[-1].extend(host_country_feature)

    # count missing features
    missings = 0

    for i in X[-1]:
        if i == -1:
            missings += 1
    if country == -1:
        missings += 1
    if host_country == -1:
        missings += 1

    X[-1].append(missings)

    return X

def get_source(driver, url, output_file):

    try:
        if not url.startswith('http'):
            url = 'https://' + url
        print(url)
        driver.get(url)
        time.sleep(5)
        # Get page source or content
        content = driver.page_source

        # save the content
        fout = open(output_file, 'w')
        fout.write(content)
        fout.close()
        
    except Exception as e:
        print('Remote Error:', e)
     


class Crawler:
    def __init__(self, pid, cache_mode, source_path):
        self.pid = pid
        self.cache_mode = cache_mode
        self.source_path = source_path

        # load whois data
        self.whois_data = {}

        # load contries
        self.countries = list(json.load(open('./assets/country.json', 'r', encoding='utf-8')).keys())

        self.X, self.Y, self.collected_urls = [], [], []
        self.ip_failed_urls = []

    def close_all(self):
        # self.info_getter.close_driver()
        pass

    def crawl(self, data):
        print(f"crawl {data}")
        url = data
        if '"' in url:
            url = url.replace('"', '')

        unified_url_whois = {check_url(url): value for url, value in self.whois_data.items()}

        # get whois data
        try:
            if url in unified_url_whois :
                icann_data = unified_url_whois[url]
            else:
                icann_data = whois.whois(url)
                self.whois_data[url] = icann_data

            if icann_data == {} or 'creation_date' not in icann_data or icann_data['creation_date'] == None:
                self.ip_failed_urls.append((url, 'whois'))
                return
        except Exception as e:
            print(f"get whois error : {e}")
            self.ip_failed_urls.append((url, 'whois'))
            # print(full_stack())
            return
  

        # get page source (from cache or DL)
        page_content = None
        fpath = os.path.join(self.source_path, url.replace('/', '').replace('?', '').replace('!', '').replace('@', '').replace(':', '') + '.html')
        print(f"start ...")
        if not os.path.exists(fpath):
            try:
                d = DesiredCapabilities.CHROME
                d['goog:loggingPrefs'] = { 'performance':'ALL', 'browser':'ALL' }
                options = uc.ChromeOptions()
                options.add_argument("--incognito")
                options.add_argument("--enable-javascript")
                options.add_argument("--ignore-certificate-errors")
                prefs = {
                    "translate_whitelists": {"fr":"en", "es":"en"},
                    "translate":{"enabled":"true"}
                }
                options.add_experimental_option("prefs", prefs)
                print(f"sel add : {selenium_address}")
                driver = webdriver.Remote(selenium_address, d, options=options)
                driver.set_page_load_timeout(10)
                print(f'************************************ {fpath} ***********')
                get_source(driver, url, fpath)
                # Close the browser
                driver.quit()

                page_content = open(fpath, 'r', errors='ignore').read()
            except Exception as e:
                print(f"error in get page source {e}")
                self.ip_failed_urls.append((url, 'content'))
                pass
        else:
            page_content = open(fpath, 'r', errors='ignore').read()

        # create features
        if page_content != None and len(page_content) > 3000:
            sample_features = convert_to_feature({url: icann_data}, self.countries, page_content, url, './assets/IP2LOCATION-LITE-DB1.BIN')

            self.X.append(sample_features)
            self.collected_urls.append(url)
        else:
            self.ip_failed_urls.append((url, 'too little content'))


def crawl_list(urls):
    # urls[0] has:: u, ind, args.cache_mode, args.source_path
    print("start thread ...")
    cache_mode = urls[0][2]
    source_path = urls[0][3]
    crawler = Crawler(os.getpid(), cache_mode, source_path)

    # get process num
    if urls[0][1] == 0:
        iterator = urls # tqdm(urls, total=len(urls))
    else:
        iterator = urls

    for url in iterator:
        crawler.crawl(url[0])

    collected_urls = crawler.collected_urls
    X = crawler.X
    failed = crawler.ip_failed_urls
    crawler.close_all()
    return [collected_urls, X, failed]

def main(args):
    # create processes to crawl
    all_urls = []

    with open(args.input_file, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            line = line.strip().replace('"', '')
            if line == '':
                continue

            u = check_url(line)
            # unify urls
            u = u.replace('https://', '').replace('http://', '')

            if '/' in u:
                u = u.split('/')[0]
            all_urls.append(u)

    print('Total number of domains = %d' %len(all_urls))

    if args.random_choose != -1 and len(all_urls) > args.random_choose:
        all_urls = random.sample(all_urls, args.random_choose)
        with open(args.input_file.replace('all_data_', 'some_data_'), 'w') as fout:
            for url in all_urls:
                fout.write(url.strip() + '\n')

    all_urls_added = []
    visited_domains = set()
    for ind, u in enumerate(list(set(all_urls))):
        domain = u
        if domain not in visited_domains:
            visited_domains.add(domain)
        else:
            continue

        all_urls_added.append([
            u, ind, args.cache_mode, args.source_path
        ])

    print('Total number of domains = %d' %len(all_urls_added))

    # create batches according to num_procs
    if len(all_urls_added) // args.num_procs == 0 :
        n = 1
    else:
        n = len(all_urls_added) // args.num_procs
    
    urls = [all_urls_added[i * n:(i + 1) * n] for i in range((len(all_urls_added) + n - 1) // n )] 
    print(f"urls : {len(urls)} , args.num_procs : {args.num_procs} , n :{n}")
    pool = Pool(args.num_procs)
    results = pool.map(crawl_list, urls)
    #pool.close()
    #pool.join()

    print('Merging the results ...')

    all_urls = []
    all_X = []
    all_failed = []
    results = [i for i in results]
    for i in results:
        for failed in i[2]:
            all_failed.append(failed)
        for url, x in zip(i[0], i[1]):
            print(url, x)
            all_urls.append(url)
            all_X.append(x)

    print('Results len = %d' %len(all_X))
    pickle.dump((all_X, all_urls), open(args.output_file, 'wb'))
    df = pd.DataFrame(all_failed, columns=["Failed URLs", "Reason"])

    # Write to a CSV file
    df.to_csv(args.output_file.replace('.pkl', '-failed.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--num_procs', type=int, help='number of processes', required=True)
    parser.add_argument('--input_file', type=str, help='input txt/json of urls', required=True)
    parser.add_argument('--source_path', type=str, help='input txt/json of urls', required=True)
    parser.add_argument('--output_file', type=str, help='output file', required=True)
    parser.add_argument('--cache_mode', action='store_true', help='use SA mode', required=False, default=False)
    parser.add_argument('--random_choose', help='random select urls', required=False, type=int, default=-1)

    args = parser.parse_args()
    main(args)
