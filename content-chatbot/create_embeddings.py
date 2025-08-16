import argparse
import os
import pickle
import requests
import xmltodict
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=' ')
    return ' '.join(text.split())


def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)


if __name__ == '__main__':
    load_dotenv(find_dotenv(), override=False)
    parser = argparse.ArgumentParser(description='Embedding website content')
    parser.add_argument('-m', '--mode', type=str, choices=['sitemap', 'zendesk'], default='sitemap',
                        help='Mode for data extraction: sitemap or zendesk')
    parser.add_argument('-s', '--sitemap', type=str, required=False,
                        help='URL to your sitemap.xml', default='https://www.paepper.com/sitemap.xml')
    parser.add_argument('-f', '--filter', type=str, required=False,
                        help='Text which needs to be included in all URLs which should be considered',
                        default='https://www.paepper.com/blog/posts')
    parser.add_argument('-z', '--zendesk', type=str, required=False,
                        help='URL to your zendesk api')
    parser.add_argument('-o', '--store', type=str, required=False, default='faiss_store.pkl',
                        help='Output path for the FAISS store pickle file')
    args = parser.parse_args()

    if args.mode == 'sitemap':
        r = requests.get(args.sitemap)
        xml = r.text
        raw = xmltodict.parse(xml)

        pages = []
        for info in raw['urlset']['url']:
            # info example: {'loc': 'https://www.paepper.com/...', 'lastmod': '2021-12-28'}
            url = info['loc']
            if args.filter in url:
                pages.append({'text': extract_text_from(url), 'source': url})
    else:  # args.mode == 'zendesk'
        def fetch_all_articles(api_url):
            pages_accum = []
            url = api_url
            while url:
                resp = requests.get(url)
                data = resp.json()
                articles = data.get('articles', [])
                for article in articles:
                    body = article.get('body') or ''
                    source = article.get('html_url') or article.get('url') or ''
                    if body and source:
                        pages_accum.append({"text": clean_html(body), "source": source})
                url = data.get('next_page')
            return pages_accum

        pages = fetch_all_articles(args.zendesk)

    # Robust splitter to avoid oversize chunks for embeddings
    # Use recursive splitter to guarantee breaking very long paragraphs without separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    batch_size = 40
    store = None
    for i in range(0, len(docs), batch_size):
        batch_texts = docs[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        if store is None:
            store = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_metas)
        else:
            store.add_texts(batch_texts, metadatas=batch_metas)
        print(f"Embedded {min(i + batch_size, len(docs))}/{len(docs)}")
    with open(args.store, "wb") as f:
        pickle.dump(store, f)
