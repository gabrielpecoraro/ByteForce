import re
import pickle

def split_pct(text, output_file=None):
    heading_pattern = re.compile(r'^(?:Article|Rule)\s+(\d+)(.*)$', re.IGNORECASE)
    chunks = []
    current_heading = None
    current_chunk = []
    
    for line in text.splitlines():
        match = heading_pattern.match(line)
        if match:
            if current_heading and current_chunk:
                chunks.append({
                    "heading": current_heading,
                    "content": "\n".join(current_chunk).strip()
                })
                current_chunk = []
            article_number = match.group(1)
            article_title = match.group(2).strip()
            current_heading = f"Article {article_number}: {article_title}"
            current_chunk.append(line)
        else:
            current_chunk.append(line)
                
    if current_heading and current_chunk:
        chunks.append({
            "heading": current_heading,
            "content": "\n".join(current_chunk).strip()
        })
    
    if output_file:
        with open(output_file, "wb") as f:
            pickle.dump(chunks, f)
    
    return chunks
