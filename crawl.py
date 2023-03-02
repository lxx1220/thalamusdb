from requests import get
from bs4 import BeautifulSoup
from time import sleep
from random import randint  # avoid throttling by not sending too many requests one after the other
from warnings import warn
import numpy as np
import pandas as pd

# Get the first page of the east bay housing prices
# Get rid of those lame-o's that post a housing option without a pic using their filter
response = get('https://sfbay.craigslist.org/search/eby/fua?hasPic=1&availabilityMode=0')
html_soup = BeautifulSoup(response.text, 'html.parser')

# find the total number of posts to find the limit of the pagination
results_num = html_soup.find('div', class_='search-legend')
# pulled the total count of posts as the upper bound of the pages array
results_total = int(results_num.find('span', class_='totalcount').text)

# each page has 119 posts so each new page is defined as follows: s=120, s=240, s=360, and so on. So we need to step in size 120 in the np.arange function
pages = np.arange(0, results_total + 1, 120)
total_iterations = len(pages)

iterations = 0

post_timing = []
post_hoods = []
post_title_texts = []
post_links = []
post_prices = []
post_imgs = []

for page in pages:

    # get request
    response = get("https://sfbay.craigslist.org/search/eby/fua?"
                   + "s="  # the parameter for defining the page number
                   + str(page)  # the page number in the pages array from earlier
                   + "&hasPic=1"
                   + "&availabilityMode=0")

    sleep(randint(1, 5))

    # throw warning for status codes that are not 200
    if response.status_code != 200:
        warn('Request: {}; Status code: {}'.format(response.request.url, response.status_code))

    # define the html text
    page_html = BeautifulSoup(response.text, 'html.parser')

    # define the posts
    posts = page_html.find_all('li', class_='result-row')
    # print(type(posts))  # To double check that I got a ResultSet
    # print(len(posts))  # To double check I got 120 (elements/page)


    # extract data item-wise
    for post in posts:
        # print(post)

        if post.find('span', class_='result-hood') is not None:

            # posting date
            # grab the datetime element 0 for date and 1 for time
            post_datetime = post.find('time', class_='result-date')['datetime']
            post_timing.append(post_datetime)

            # neighborhoods
            post_hood = post.find('span', class_='result-hood').text[2:-1].replace(' / ', ',').lower()
            post_hoods.append(post_hood)

            # title text
            post_title = post.find('a', class_='result-title hdrlnk')
            post_title_text = post_title.text.replace("\t", " ")
            post_title_texts.append(post_title_text)

            # post link
            post_link = post_title['href']
            post_links.append(post_link)

            # removes the \n whitespace from each side, removes the currency symbol and commas, and turns it into an int
            post_price = int(post.a.text.strip().replace("$", "").replace(",", ""))
            post_prices.append(post_price)

            # Add image_urls, removes first two characters '3:'
            img_ids = [img_id[2:] for img_id in post.a['data-ids'].split(',')]
            img_urls = [f'https://images.craigslist.org/{img_id}_300x300.jpg' for img_id in img_ids]
            img_urls_str = ','.join(img_urls)
            post_imgs.append(img_urls_str)

    iterations += 1
    print(f"Page {iterations}/{total_iterations} scraped successfully!")

print("\n")

print("Scrape complete!")

furniture = pd.DataFrame({'time': post_timing,
                        'neighborhood': post_hoods,
                        'title': post_title_texts,
                        'url': post_links,
                        'price': post_prices,
                        'imgs': post_imgs
                        })
print(furniture.info())
furniture.head(10)

furniture.to_csv('craigslist/furniture.tsv', sep="\t")

furniture = pd.read_csv('craigslist/furniture.tsv', sep='\t', index_col=0)

import urllib.request

for rid, img_urls_str in enumerate(furniture['imgs']):
    # if rid < 2689:
    #     continue

    if rid % 50 == 0:
        sleep(randint(1, 5))

    img_urls = img_urls_str.split(',')
    for img_id, img_url in enumerate(img_urls):
        try:
            urllib.request.urlretrieve(img_url, f'craigslist/furniture_imgs/{rid}_{img_id}.jpg')
        except urllib.error.HTTPError as err:
            print(err)
            print(img_url, rid, img_id)




