import os
import json
import re
import fitz
import pandas as pd
from glob import glob
from collections import Counter
import datetime
import torch
from sentence_transformers import SentenceTransformer, util

# ==============================================================================
# PART 1: HEURISTIC OUTLINE EXTRACTOR
# ==============================================================================

class OutlineExtractor:
    """
    Extracts a structured outline from a PDF using a sophisticated heuristic engine.
    This is an internal component for the SemanticAnalyser.
    """
    def __init__(self):
        pass

    def _get_line_properties(self, doc):
        lines_data = []
        all_font_sizes = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        for span in line['spans']:
                            all_font_sizes.append(round(span['size']))

        if not all_font_sizes:
            return pd.DataFrame()

        body_size = Counter(all_font_sizes).most_common(1)[0][0]

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        line_text = " ".join([span['text'] for span in line['spans']]).strip()
                        if not line_text or not line['spans']:
                            continue

                        span = line['spans'][0]
                        font_size = round(span['size'])
                        is_bold = 'bold' in span['font'].lower() or span.get('flags', 0) & 2**4

                        score = 0
                        if font_size > body_size:
                            score += (font_size - body_size)
                        if is_bold:
                            score += body_size * 0.5
                        if bool(re.match(r'^\s*(\d+(\.\d+)*|[A-Za-z][\.\)])\s+', line_text)):
                            score += body_size * 0.7
                        if len(line_text) < 100:
                            score += 1

                        lines_data.append({
                            'text': line_text, 'font_size': font_size,
                            'page': page_num + 1, 'score': score, 'y0': line['bbox'][1]
                        })
        return pd.DataFrame(lines_data)

    def _build_hierarchy(self, lines_df):
        if lines_df.empty:
            return "Untitled Document", []

        min_score_threshold = lines_df['score'].quantile(0.90)
        headings_df = lines_df[lines_df['score'] > min_score_threshold].copy()

        if headings_df.empty:
            return "Untitled Document (No Headings Found)", []

        headings_df.sort_values(by=['page', 'y0'], inplace=True)
        doc_title = headings_df.iloc[0]['text']
        headings_df = headings_df.iloc[1:]

        heading_font_sizes = sorted(headings_df['font_size'].unique(), reverse=True)
        size_to_level = {size: f"H{i+1}" for i, size in enumerate(heading_font_sizes[:3])}

        outline = []
        for _, h in headings_df.iterrows():
            level = size_to_level.get(h['font_size'])
            if level:
                outline.append({"level": level, "text": h['text'], "page": h['page']})
        return doc_title, outline

    def extract_outline(self, doc):
        lines_df = self._get_line_properties(doc)
        if lines_df.empty:
            return {"title": "Empty Document", "outline": []}
        title, outline = self._build_hierarchy(lines_df)
        return {"title": title, "outline": outline}


# ==============================================================================
# PART 2: SEMANTIC ANALYSER
# ==============================================================================

class SemanticAnalyser:
    def __init__(self):
        model_path = 'models/all-MiniLM-L6-v2'
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model directory not found at '{model_path}'. "
                "Please run the 'download_model_locally.py' script first to download the model files."
            )

        print(f"Initializing Semantic Analyser from local path: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.parser = OutlineExtractor()
        print("SentenceTransformer model loaded successfully from local files.")

    def _get_full_text(self, doc):
        return "".join(page.get_text() for page in doc)

    def _perform_structure_aware_chunking(self, doc_path):
        doc = fitz.open(doc_path)
        full_text = self._get_full_text(doc)
        outline_data = self.parser.extract_outline(doc)

        chunks = []
        headings = outline_data.get('outline',)

        if not headings:
            return [{'text': para, 'page': 0, 'title': 'N/A'} for para in full_text.split('\n\n') if para.strip()]

        for i, heading in enumerate(headings):
            start_pos = full_text.find(heading['text'])
            if start_pos == -1:
                continue
            end_pos = len(full_text)
            if i + 1 < len(headings):
                next_heading_text = headings[i + 1]['text']
                end_pos_candidate = full_text.find(next_heading_text, start_pos)
                if end_pos_candidate != -1:
                    end_pos = end_pos_candidate
            chunk_text = full_text[start_pos:end_pos].strip()
            chunks.append({'text': chunk_text, 'page': heading['page'], 'title': heading['text']})
        return chunks

    def _get_refined_text(self, text, query_embedding, top_k=5):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if not sentences:
            return text[:500]
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, sentence_embeddings)
        top_indices = torch.topk(similarities, k=min(top_k, len(sentences))).indices.squeeze().tolist()
        if not isinstance(top_indices, list):
            top_indices = [top_indices]
        top_sentences = [sentences[i] for i in sorted(top_indices)]
        return " ".join(top_sentences)

    def analyze_documents(self, pdf_paths, persona, job_to_be_done):
        query = f"As a {persona}, I need to {job_to_be_done}."
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        all_chunks = []

        for doc_path in pdf_paths:
            doc_chunks = self._perform_structure_aware_chunking(doc_path)
            for chunk in doc_chunks:
                chunk['doc_path'] = os.path.basename(doc_path)
                all_chunks.append(chunk)

        if not all_chunks:
            return {}

        chunk_texts = [chunk['text'] for chunk in all_chunks]
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, chunk_embeddings)

        # Flatten the similarities into a 1D tensor
        similarity_scores = similarities.view(-1)

        # Ensure length match
        assert len(all_chunks) == len(similarity_scores), "Mismatch between chunks and similarity scores"

        # Assign similarity scores to each chunk
        for chunk, score in zip(all_chunks, similarity_scores):
            chunk['similarity'] = score.item()


        ranked_chunks = sorted(all_chunks, key=lambda x: x['similarity'], reverse=True)

        extracted_sections = []
        sub_section_analysis = []

        for i, chunk in enumerate(ranked_chunks[:10]):
            extracted_sections.append({
                "document": chunk['doc_path'],
                "section_title": chunk['title'],
                "importance_rank": i + 1,
                "page_number": chunk['page'],
            })
            if i < 3:
                refined_text = self._get_refined_text(chunk['text'], query_embedding)
                sub_section_analysis.append({
                    "document": chunk['doc_path'],
                    "page_number": chunk['page'],
                    "refined_text": refined_text
                })

        return {
            "Metadata": {
                "input_documents": [os.path.basename(p) for p in pdf_paths],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.datetime.now().isoformat()
            },
            "extracted_section": extracted_sections,
            "sub-section_analysis": sub_section_analysis
        }


# ==============================================================================
# PART 3: MAIN EXECUTION
# ==============================================================================

def read_text_file(filepath, default_text=""):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return default_text

def run_challenge_1b():
    input_dir = 'input'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    persona = read_text_file(os.path.join(input_dir, 'persona.txt'), "PhD Researcher")
    job_to_be_done = read_text_file(os.path.join(input_dir, 'job_to_be_done.txt'), "Prepare a literature review")
    pdf_files = glob(os.path.join(input_dir, '*.pdf'))

    if not pdf_files:
        print("Error: No PDF files found in the input directory.")
        return

    analyser = SemanticAnalyser()
    analysis_result = analyser.analyze_documents(pdf_files, persona, job_to_be_done)

    output_path = os.path.join(output_dir, 'challenge1b_output.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=4, ensure_ascii=False)

    print(f"\nAnalysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_challenge_1b()
