## ycombinator hacker news preferences tool

A tool that guesses if you are interested in an article on THN based on previous inputs.

### Getting Setup

First install python3, pip, and get a virtual environment setup. Then install all of the packages in requirements.txt.

### Resetting Data

The project comes with a data file full of my own preferences. To reset it and start building up your own dataset remove "user.csv".

### Usage

To launch the project execute: `./collect_my_opinions`. The project integrates
the classifier and news reader into a single tool that will scrape the current
frontpage of news.ycombinator.com, ask you what you think about each article,
and then use the classifier to tell you what the decision the model would
have made.

After the news reader is finished or you exit (Ctrl-C) a retraining step
will be run to recompute preferences based on any new data that may have been
added. The next time you run the `collect_my_opinions` script the model will
be based on the data from the previous run.
