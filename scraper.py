import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_core.documents import Document

def scrape_website(start_url: str, max_pages: int = 10) -> list[Document]:
    visited = set()
    to_visit = [start_url]
    documents = []
    base_domain = urlparse(start_url).netloc

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = "\n".join(line for line in text.splitlines() if line.strip())

        if text:
            documents.append(Document(page_content=text, metadata={"source": url}))

        for a_tag in soup.find_all("a", href=True):
            href = urljoin(url, a_tag["href"])
            parsed = urlparse(href)
            if parsed.netloc == base_domain and href not in visited:
                to_visit.append(href)

    return documents