import asyncio
import re

import requests
from playwright.async_api import async_playwright
import platform


# Ensure ProactorEventLoop on Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def scrape_website(website):
    if not website.startswith(('http://', 'https://')):
        website = 'http://' + website

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # Use Chromium browser in headless mode
        page = await browser.new_page()
        await page.goto(website)
        print(f"Page Loaded: {website}")
        html = await page.content()  # Get page content as string
        await browser.close()
        return html

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
visited_links = set()
def scrape_url(url):
    content = []
    try:
        response = requests.get(url, timeout=30)

        # Handle warnings from requests about SSL verification
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for elem in soup.find_all(['script', 'style']):
            elem.extract()

        # Extract text from all paragraphs (or other tags as needed)
        raw_text = soup.get_text()
        clean_text = re.sub(r'\s+', ' ', raw_text).strip()

        content.append(clean_text)

        # Extract all links on the current page
        links = set(a.get("href") for a in soup.find_all("a", href=True) if a.get("href") is not None)

        # Normalize links and recursively scrape
        for link in links:
            if link.startswith("/"):
                link = url + link

            date_pattern = r"\d{4}-\d{2}-\d{2}"

            if link.startswith(url) and link not in visited_links:
                if re.search(date_pattern, link):
                    continue

                # Split the URL into components to check depth
                path = link[len(url):]
                path_components = path.split("/")

                # Allow only up to 2 path components
                if len([comp for comp in path_components if comp]) <= 1:
                    visited_links.add(link)

                    scrape_url(link)


        return content

    except Exception as e:
        print(f"Error scraping {url}: {e}")

async def extract_anchor_tags(base_url):
    """

    :param base_url: It is the url that is to be scrapped from
    :return:  String of all the links scrapped from the page.
    """
    html_content = await scrape_website(base_url)
    soup = BeautifulSoup(html_content, "html.parser")
    anchor_tags = soup.find_all("a", href=True)
    links = []
    for tag in anchor_tags:
        counter = 0
        href = tag['href']
        full_url = urljoin(base_url, href)  # Convert relative URL to absolute
        if urlparse(full_url).netloc == urlparse(base_url).netloc:
            links.append(full_url)
        links.append(full_url)
        counter += 1

    return "\n".join(set(links))

# Store all the anchor_tags
all_links_of_nepalnow = {}

from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

async def extract_all_links_and_content(base_url):
    if not base_url.startswith(('http://', 'https://')):
        base_url = 'https://' + base_url

    html_content = await scrape_website(base_url)
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all anchor tags
    anchor_tags = soup.find_all("a", href=True)
    links = []

    # Parse the domain of the base URL
    base_domain = urlparse(base_url).netloc

    for tag in anchor_tags:
        href = tag['href']
        full_url = urljoin(base_url, href)  # Convert relative URL to absolute
        if urlparse(full_url).netloc == base_domain:  # Check if domain matches
            print(f"Valid link (same domain): {full_url}")
            links.append(full_url)  # Add to links list
        else:
            print(f"Discarded link (different domain): {full_url}")

    # Dictionary to store URL, raw HTML, and cleaned content
    link_content_mapping = {}

    for link in links:
        try:
            print(f"Scraping URL: {link}")
            page_html = await scrape_website(link)
            body_content = extract_body_content(page_html)  # Use your existing extract_body_content logic
            cleaned_content = clean_body_content(body_content)  # Use your existing clean_body_content logic

            # Store scraped content
            link_content_mapping[link] = {
                "html": page_html,
                "body": cleaned_content,
            }
        except Exception as e:
            print(f"Failed to scrape {link}: {e}")

    return link_content_mapping



def split_dom_content(dom_content, max_length=6000):
    return [
        dom_content[i : i + max_length] for i in range(0, len(dom_content), max_length)
    ]

def extract_body_content(html_content, exclude_selectors=None):
    """
    Extracts the body content of a webpage, excluding repetitive elements.

    Args:
    - html_content (str): The raw HTML content of the webpage.
    - exclude_selectors (list): A list of CSS selectors for elements to exclude.

    Returns:
    - str: The body content with specified elements excluded.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body

    # If no body tag is found, return an empty string
    if not body_content:
        return ""

    # Exclude repetitive elements based on selectors
    if exclude_selectors:
        for selector in exclude_selectors:
            for element in body_content.select(selector):
                element.decompose()  # Remove the element from the soup

    return str(body_content)


def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )
    return cleaned_content

def deduplicate_output(parsed_result):
    """
    Removes redundant or overlapping parts from the parsed result.

    Args:
        parsed_result (str): The concatenated parsed output from all chunks.

    Returns:
        str: The deduplicated result.
    """
    lines = parsed_result.split("\n")
    unique_lines = []
    seen = set()


    for line in lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    return "\n".join(unique_lines)
