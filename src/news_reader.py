import shutil
import re
import urllib3
import time
import pandas as pd
from bs4 import BeautifulSoup
from model import build_bert_model, load_model, predict

# Reload our current prediction model if it exists, if we haven't trained one yet then do nothing
our_model = None
bert_model = build_bert_model()
new_predictions = 0
correct_new_predictions = 0

try:
  our_model = load_model("saved_model/current")
except:
  pass

# Make a backup of our user data (.csv) before operating on it, just in case.
def make_backup():
  new_backup_name = "backups/%i" % int(time.time())
  try:
    shutil.copy("user.csv", new_backup_name)
    print("Made a backup to %s" % new_backup_name)
  except:
    pass

# Load the current user data
def load_data():
  try:
    df = pd.read_csv("user.csv")
  except Exception as e:
    print("%s - creating fresh" % e)
    df = pd.DataFrame({})
  return df

# Save the current user data
def save_data(data_frame):
  data_frame.to_csv("user.csv", index=False)
  print("Saved preferences")

# Find an article in the data frame (TODO: This could be a hash lookup for speed)
def find_article(data_frame, url):
  for article in data_frame.iterrows():
    if article[1]["url"] == url:
      return True
  return False

# Scrape the first {pages} pages from yc and extract the features we want
def scrape(pages):
  url = "https://news.ycombinator.com/news?p="
  http = urllib3.PoolManager()

  scraped_articles = []

  print(f"Loading {pages} pages")

  # THN pages have trivially predictable URLs
  for i in range(pages):

    # Download the frontpage + i'th page
    try:
      r = http.request('GET', f"{url}{i}")
    except Exception as E:
      print(f"Could not open {i} because {E}")
      continue

    # If non-200 either something is wrong upstream or we have been blocked
    if r.status != 200:
      print("Non-200 code")
      continue

    # Parse the HTML of the page we've downloaded
    soup = BeautifulSoup(r.data, features="html.parser")

    # The table tagged "itemlist" contains the data we want
    item_list_table = soup.find("table", attrs={"class":"itemlist"})

    if item_list_table == None:
      print("Could not find any data on the page")
      continue

    # Each item is split across two table rows, the even indexed rows contain the title and the odd indexed rows contain the comments, score, and age.
    item_list_items = item_list_table.find_all("tr")

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

      # Finally attach this piece of scraped data to our results
      scraped_articles.append((result.get('href'), result.get_text(), score, comments, age))

  return scraped_articles

# Read a value between 0 and 3 from the user. Block until the user enters a valid input.
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

# This method does a prediction using the current
def do_prediction(features):

  global new_predictions
  global correct_new_predictions

  # Run the model on the new article and make a prediction
  prediction = predict(bert_model, our_model, features)[0]

  new_predictions += 1

  # Interpret the prediction as a binary (true / false) and interpret the score we
  # just entered as a binary (> 1 in 0-3) then compare the results.
  interest_as_binary = features[-1] > 1
  interest_prediction_as_binary = prediction > 0

  # If our prediction as a binary (> 0 predicts we like the article) matches the input that we just entered
  # (> 1 = predicted we like the article, <= 1 = predicted we disliked the article) then we mark it as correct
  if interest_as_binary == interest_prediction_as_binary:
    correct_new_predictions += 1

  # Print some statistics about the prediction accuracy over this session
  print(interest_as_binary, prediction > 0, interest_as_binary == interest_prediction_as_binary)
  print(f"Our prediction for {features['title']}: {prediction > 0} {prediction}")
  print(f"Of the new predictions made this session {correct_new_predictions} of {new_predictions} were correct {(float(correct_new_predictions) / float(new_predictions)) * 100.0}%")

# This method generates the data from the scrape and then asks a user to score it before finally adding the scored data to the data frame.
# If there is an existing model it will also print out expected prediction about it.
def score_article(data_frame, article):

  url = article[0]

  if not (url.startswith("https://") or url.startswith("http://")):
    print("Not a web-link. Skipping article")
    return

  title = article[1]
  score = article[2]
  comments = article[3]
  age = article[4]

  if not find_article(data_frame, url):
    print("Title: %s" % title)
    print("Url: %s" % url)

    interest = get_interest()
    data_frame.loc[df.shape[0]] = {"url": url, "title": title, "score": score, "comments": comments, "age": age, "user_interest": interest, "user_ranking_time": int(time.time())}

    if our_model != None:
      do_prediction(data_frame.loc[df.shape[0] - 1])
  else:
    print(f"Skipping {title} because you have seen this article before")
    # If we have seen it before then skip
    return

if __name__ == "__main__":

  make_backup()

  print("Loading existing data")
  df = load_data()

  print("Starting scraper")
  scraped = scrape(10)

  # For each article that we scraped ask the user to score it and then save the user data
  for article in scraped:
    score_article(df, article)
    save_data(df)
