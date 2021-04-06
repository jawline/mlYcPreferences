import shutil
import urllib3
import time
import pandas as pd
from bs4 import BeautifulSoup

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
      r = http.request('GET', "%s%i" % (url, i))
    except Exception as E:
      print("Could not open %i because %s" % (i, E))
      continue

    if r.status != 200:
      print("Non-200 code")
      continue

    soup = BeautifulSoup(r.data, features="html.parser")
    print("Loaded page %i" % i)

    results = soup.find_all('a', attrs={"class":"storylink"})

    for result in results:
      scraped_articles.append((result.get('href'), result.get_text()))

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
  previous = df.get(url)
  if not isinstance(previous, pd.Series):
    print("Title: %s" % title)
    print("Url: %s" % url)

    interest = get_interest()

    data_frame[url] = [title, interest]
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

  for article in scraped:
    score_article(df, article)
    save_data(df)
