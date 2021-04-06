import shutil
import re
import urllib3
import time
import pandas as pd
from bs4 import BeautifulSoup
from model import load_model, prepare_input

our_model = None

try:
  our_model = load_model("saved_model/current")
except:
  pass

def make_backup():
  new_backup_name = "backups/%i" % int(time.time())
  try:
    shutil.copy("user.csv", new_backup_name)
    print("Made a backup to %s" % new_backup_name)
  except:
    pass

def load_data():
  try:
    df = pd.read_csv("user.csv")
  except Exception as e:
    print("%s - creating fresh" % e)
    df = pd.DataFrame({})
  return df

def save_data(data_frame):
  data_frame.to_csv("user.csv", index=False)

def scrape(pages):
  url = "https://news.ycombinator.com/news?p="
  http = urllib3.PoolManager()

  scraped_articles = []

  for i in range(pages):
    try:
      r = http.request('GET', f"{url}{i}")
    except Exception as E:
      print(f"Could not open {i} because {E}")
      continue

    if r.status != 200:
      print("Non-200 code")
      continue

    soup = BeautifulSoup(r.data, features="html.parser")
    print(f"Loaded page {i}")

    item_list_table = soup.find("table", attrs={"class":"itemlist"})

    if item_list_table == None:
      print("Could not find any data on the page")
      continue

    item_list_items = item_list_table.find_all("tr")
    print(f"Len: {len(item_list_items)}")

    for i in range(0, len(item_list_items) - 2, 3):
      first = item_list_items[i]
      second = item_list_items[i + 1]

      # Extract the link, title and age
      result = first.find("a", attrs={"class":"storylink"})
      age = second.find("span", attrs={"class":"age"}).get_text().split()[0]

      # Extract the score (upvotes)
      score = 0
      score_element = second.find("span", attrs={"class":"score"})
      if score_element != None:
        score = score_element.get_text().split()[0]

      # Extract the comments
      comments = 0

      # The "comments" tag is unlabelled so we need to search for it
      elements_that_might_be_comments = second.find_all("a")
      for element in elements_that_might_be_comments:

        match = re.search('(\d+)\s+comment', element.get_text())

        if match != None:
          comments = match.group(1)
          break

      scraped_articles.append((result.get('href'), result.get_text(), score, comments, age))

  return scraped_articles

def get_interest():
  while True:
    interest = input("Interest (0-3): ")

    # Sanitize our input to a value between 0 and 3
    try:
      interest = int(interest)
      if interest < 0 or interest > 3:
        print("Bad input. Try again")
        continue
      return interest
    except:
      print("Bad input. Try again")
      continue

def score_article(data_frame, article):
  url = article[0]

  if not (url.startswith("https://") or url.startswith("http://")):
    print("Not a web-link. Skipping article")
    return

  title = article[1]
  score = article[2]
  comments = article[3]
  age = article[4]

  if our_model != None:
    prediction = our_model.predict([prepare_input(url, [title, score, comments, age, 0])[0]])
    print(f"Our prediction for {title}: {prediction[0][0] > 0} {prediction[0][0]}")

  previous = df.get(url)
  if not isinstance(previous, pd.Series):
    print("Title: %s" % title)
    print("Url: %s" % url)

    interest = get_interest()

    data_frame[url] = [title, score, comments, age, interest]

  else:
    # If we have seen it before then skip
    return

if __name__ == "__main__":

  make_backup()

  print("Loading existing data")
  df = load_data()

  print("---------------------------------")
  print("Existing Classifications")

  for article in df:
    print(article)
    data = df[article]
    for dp in data:
      print(dp)

  print("---------------------------------")

  print("Starting scraper")
  scraped = scrape(5)

  print("Scraped: ", scraped)

  for article in scraped:
    score_article(df, article)
    save_data(df)
