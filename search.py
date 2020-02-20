#!/usr/bin/python

# WRITTEN MOSTLY BY GOOGLE, BUT WITH A FEW TWEAKS FROM RYAN SYNK
# THIS HAS A HARD-CODED API KEY IN IT. If you have access to the youtube
# api, simply set an environment variable YOUTUBE_API to the key given
# by youtube, and this should run fine.

# This sample executes a search request for the specified search term.
# Sample usage:
#   python search.py --q=surfing --max-results=10
# NOTE: To use the sample, you must provide a developer key obtained
#       in the Google APIs Console. Search for "REPLACE_ME" in this code
#       to find the correct place to provide that key..

import argparse
import os
import csv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of
#   https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
DEVELOPER_KEY = os.environ['YOUTUBE_API']
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# Gets URLS, and writes them to a csv
def youtube_search(options):
  youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)

  pageTok = None
  urls = []

  while len(urls) < options.n:

    # Call the search.list method to retrieve results matching the specified
    # query term.
    search_response = youtube.search().list(
      q=options.q,
      part='id,snippet',
      pageToken=pageTok,
      maxResults=options.max_results
    ).execute()

    # Add each result to the appropriate list, and then display the lists of
    # matching videos, channels, and playlists.
    for search_result in search_response.get('items', []):
      if search_result['id']['kind'] == 'youtube#video':
        url = 'https://www.youtube.com/watch?v=' + search_result['id']['videoId']
        urls.append(url)

    # Updates page token to get next page of results
    pageTok = search_response.get('nextPageToken')

  with open(options.f, "w") as csvfile:
      write = csv.writer(csvfile)
      for u in urls:
          write.writerow([u])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--q', help='Search term', default='Google')
  parser.add_argument('--max-results', help='Max results', default=50)
  parser.add_argument('--f', help='File name', default='search.csv')
  parser.add_argument('--n', help='Number of search terms', default=500)
  args = parser.parse_args()

  youtube_search(args)
