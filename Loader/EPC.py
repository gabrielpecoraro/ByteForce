import pdfplumber
import PyPDF2
import collections
from collections import Counter


def is_bold(word):
    """
    Check if the word is bold
    """
    return "fontname" in word and "Bold" in word["fontname"]


def partition_in_chapter_and_article(path):
    """
    Partition the PDF file into chapters and articles within each chapter
    """
    if type(path) != str:
        raise TypeError("path must be a string")

    chapters = []
    current_chapter = []
    chapter_titles = ["Chapter", "Chapitre", "Kapitel"]
    article_titles = ["Article", "Artikel", "Article"]

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            lines = page.extract_words()
            for line in lines:
                line_text = line["text"]
                if any(title in line_text for title in chapter_titles) and is_bold(
                    line
                ):
                    if current_chapter:
                        chapters.append(current_chapter)
                        current_chapter = []
                current_chapter.append(line_text)
        if current_chapter:
            chapters.append(current_chapter)

    chapters_with_articles = []
    for chapter in chapters:
        articles = []
        current_article = []
        for line in chapter:
            if any(title in line for title in article_titles) and is_bold(line):
                if current_article:
                    articles.append("\n".join(current_article))
                    current_article = []
            current_article.append(line)
        if current_article:
            articles.append("\n".join(current_article))
        chapters_with_articles.append(articles)

    return chapters_with_articles


def chunk_text(text, chunk_size=512):
    """
    Chunk the text into smaller pieces
    """
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    return chunks


# Example usage
chapters_with_articles = partition_in_chapter_and_article(
    "../Dataset/1-EPC_17th_edition_2020_en.pdf"
)
for i, articles in enumerate(chapters_with_articles):
    print(f"Chapter {i + 1}:")
    for j, article in enumerate(articles):
        print(
            f"  Article {j + 1}:\n{article[:500]}...\n"
        )  # Print first 500 characters of each article
        text_chunks = chunk_text(article)
        print(f"  Generated {len(text_chunks)} chunks for Article {j + 1}")
