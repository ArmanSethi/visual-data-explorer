"""Web scraper module for collecting heterogeneous data from the internet."""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse
import os
import time
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """Scrapes images, tables, and text from web pages."""
    
    def __init__(self, base_url: str, output_dir: str = "data/raw"):
        self.base_url = base_url
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/tables", exist_ok=True)
        os.makedirs(f"{output_dir}/text", exist_ok=True)
    
    def scrape_images(self, url: str, limit: int = 50) -> List[str]:
        """Scrape images from a URL."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            images = []
            for img in soup.find_all('img', limit=limit):
                img_url = img.get('src') or img.get('data-src')
                if img_url:
                    img_url = urljoin(url, img_url)
                    images.append(img_url)
            
            logger.info(f"Found {len(images)} images at {url}")
            return images
        except Exception as e:
            logger.error(f"Error scraping images from {url}: {e}")
            return []
    
    def scrape_tables(self, url: str) -> List[pd.DataFrame]:
        """Scrape tables from a URL."""
        try:
            tables = pd.read_html(url)
            logger.info(f"Found {len(tables)} tables at {url}")
            return tables
        except Exception as e:
            logger.error(f"Error scraping tables from {url}: {e}")
            return []
    
    def scrape_text(self, url: str) -> Dict[str, str]:
        """Scrape text content from a URL."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text_data = {
                'title': soup.title.string if soup.title else '',
                'headings': [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])],
                'paragraphs': [p.get_text(strip=True) for p in soup.find_all('p')],
                'full_text': soup.get_text(separator=' ', strip=True)
            }
            
            logger.info(f"Scraped text from {url}")
            return text_data
        except Exception as e:
            logger.error(f"Error scraping text from {url}: {e}")
            return {}
    
    def download_image(self, img_url: str, filename: str) -> bool:
        """Download an image to disk."""
        try:
            response = self.session.get(img_url, timeout=10)
            response.raise_for_status()
            
            filepath = os.path.join(self.output_dir, 'images', filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded image: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error downloading image {img_url}: {e}")
            return False
    
    def save_table(self, df: pd.DataFrame, filename: str):
        """Save a table to CSV."""
        filepath = os.path.join(self.output_dir, 'tables', filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved table: {filename}")
    
    def save_text(self, text_data: Dict, filename: str):
        """Save text data to file."""
        filepath = os.path.join(self.output_dir, 'text', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {text_data.get('title', '')}\n\n")
            f.write(f"Headings:\n{chr(10).join(text_data.get('headings', []))}\n\n")
            f.write(f"Content:\n{text_data.get('full_text', '')}")
        logger.info(f"Saved text: {filename}")

if __name__ == "__main__":
    # Example usage
    scraper = WebScraper("https://example.com")
    print("Web scraper module loaded successfully")
