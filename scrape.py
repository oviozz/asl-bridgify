
import requests
import re

def scrape_signing_savvy(word):
    base_url = 'https://www.signingsavvy.com'
    search_url = f'{base_url}/search/{word}'

    response = requests.get(search_url)
    print(f'Requesting URL: {search_url}')

    if response.status_code == 200:
        html_content = response.text

        # Regular expression pattern to extract URLs ending with .mp4
        pattern = r'href="([^"]+\.mp4)"'

        # Using re.search to find the first match
        match = re.search(pattern, html_content)

        if match:
            # Extracting the URL from the match
            mp4_link = match.group(1)

            print(f'Found .mp4 link: {mp4_link}')
            return mp4_link
        else:
            print('No .mp4 link found')
            return "No .mp4 link found"
    else:
        print(f'Failed to retrieve the webpage. Status code: {response.status_code}')
        return "Failed to retrieve the webpage"
