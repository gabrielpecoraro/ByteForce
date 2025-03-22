import pickle

def is_bold(word):
    """
    Check if the word is bold
    """
    return "fontname" in word and "Bold" in word["fontname"]

def partition_in_chapter_and_article_from_pkl(pkl_path):
    """
    Partition the data from a .pkl file into chapters and articles within each chapter
    """
    if type(pkl_path) != str:
        raise TypeError("pkl_path must be a string")

    chapters = []
    current_chapter = []
    chapter_titles = ["Chapter", "Chapitre", "Kapitel"]
    article_titles = ["Article", "Artikel", "Article"]

    # Load the data from the .pkl file
    try:
        with open(pkl_path, "rb") as f:
            documents = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{pkl_path}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading '{pkl_path}': {e}")

    # Process each document (assuming each document has `page_content` containing text)
    for doc in documents:
        lines = doc.page_content.splitlines()  # Split the page content into lines
        for line_text in lines:
            # Check if the line matches chapter titles
            if any(title in line_text for title in chapter_titles):
                if current_chapter:
                    chapters.append(current_chapter)
                    current_chapter = []
            current_chapter.append(line_text)
        if current_chapter:  # Append the last chapter
            chapters.append(current_chapter)

    # Partition chapters into articles
    chapters_with_articles = []
    for chapter in chapters:
        articles = []
        current_article = []
        for line in chapter:
            # Check if the line matches article titles
            if any(title in line for title in article_titles):
                if current_article:
                    articles.append("\n".join(current_article))
                    current_article = []
            current_article.append(line)
        if current_article:  # Append the last article
            articles.append("\n".join(current_article))
        chapters_with_articles.append(articles)

    return chapters_with_articles


def chunk_text(text, chunk_size=512):
    """
    Chunk the text into smaller pieces
    """
    words = text.split()
    chunks = [
        " ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    return chunks