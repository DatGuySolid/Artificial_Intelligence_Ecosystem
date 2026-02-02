import requests
from bs4 import BeautifulSoup

def scrape_wiki_page():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find('div', id='mw-content-text')
            paragraphs = content.find_all('p')
            extracted_text = '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            with open('Selected_Document.txt', 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print("Success! Text saved to Selected_Document.txt")
            return extracted_text
        else:
            print(f"Failed with status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    scrape_wiki_page()