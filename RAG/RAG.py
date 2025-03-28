from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from docx import Document
import random
from pathlib import Path
import os
from RAG.llm_ollama import OllamaLLM
from requests.exceptions import ConnectionError
import streamlit as st
import re


def extract_qa_from_docx(path):
    doc = Document(path)
    qa_pairs = []
    question = ""
    in_question = False
    answer = ""

    for para in doc.paragraphs:
        text = para.text.strip()

        if text.lower().startswith("question"):
            in_question = True
            question = ""
            answer = ""
        elif text.lower().startswith("answer"):
            in_question = False
        elif text.lower().startswith("the correct answer is"):
            answer = text
            qa_pairs.append((question.strip(), answer.strip()))
        elif in_question:
            question += " " + text

    return qa_pairs


@st.cache_data(show_spinner=False)
def format_few_shots(qa_pairs, max_examples=10):
    """
    Formats few-shot examples to demonstrate how a highly experienced legal assistant should answer questions.
    Each example should illustrate that answers are strictly based on the provided legal context.
    """
    prompt = (
        "Below are examples of how a highly experienced legal assistant answers legal questions using only the provided context. "
        "If the context is insufficient, the assistant clearly states that fact.\n\n"
    )
    for q, a in qa_pairs[:max_examples]:
        prompt += f"Question: {q}\nAnswer: {a}\n----\n"
    return prompt


def get_random_test_questions(qa_pairs, n=5, seed=19):
    random.seed(seed)
    return [q for q, _ in random.sample(qa_pairs, min(n, len(qa_pairs)))]


class RAG:
    def __init__(
        self,
        embedding_model_name,
        faiss_index_dir,
        llm_model_name,
        few_shot_docx_path="Questions Sup OEB.docx",
        max_few_shot_examples=2,
    ):
        # Initialize embedding model with new package
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Check if FAISS index exists
        if os.path.exists(os.path.join(faiss_index_dir, "index.faiss")):
            self.vectorstore = FAISS.load_local(
                faiss_index_dir, self.embedder, allow_dangerous_deserialization=True
            )
        else:
            print(f"Warning: FAISS index not found at {faiss_index_dir}")
            print("Please run the indexing process first using main.py")
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_dir}")

        self.client = OllamaLLM(model_name=llm_model_name)

        self.few_shot_prompt = ""
        if few_shot_docx_path and Path(few_shot_docx_path).exists():
            qa_pairs = extract_qa_from_docx(few_shot_docx_path)
            self.few_shot_prompt = format_few_shots(
                qa_pairs, max_examples=max_few_shot_examples
            )
        self.test_questions = [
            "What are the requirements for patentability under EPC Article 52?"
        ]

        self.prompt_template = PromptTemplate(
            template=(
                "You are an expert European Patent Attorney answering questions about patent law. "
                "Provide a clear and detailed answer using ONLY the information from the context below. "
                "If specific articles or guidelines are mentioned in the context, cite them.\n\n"
                "If the context contains the information:\n"
                "- Provide a direct, factual answer\n"
                "- Quote relevant passages when appropriate\n"
                "- Explain technical terms\n"
                "- Structure your response with clear sections\n\n"
                "If the context lacks specific information:\n"
                "- Say 'The provided context does not contain sufficient information about [topic]'\n"
                "- Do NOT refer to external sources or suggest looking elsewhere\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:\n"
            ),
            input_variables=["context", "question"],
        )

        # Add question type templates
        self.question_templates = {
            "open": PromptTemplate(
                template=(
                    "Based on the following legal context, generate an open-ended question "
                    "that tests understanding of patent law concepts.\n\n"
                    "Context:\n{context}\n\n"
                    "Requirements:\n"
                    "1. Question should be specific to the provided context\n"
                    "2. Focus on testing understanding of legal principles\n"
                    "3. Question should require detailed explanation\n"
                    "Generate only the question."
                ),
                input_variables=["context"],
            ),
            "mcq": PromptTemplate(
                template=(
                    "Based on the following legal context, generate a multiple-choice question "
                    "about patent law. Include 4 options (A, B, C, D) with one correct answer.\n\n"
                    "Context:\n{context}\n\n"
                    "Requirements:\n"
                    "1. Question should be specific to the context\n"
                    "2. All options should be plausible\n"
                    "3. Include the correct answer at the end\n"
                    "Format:\n"
                    "Question: [Your question]\n"
                    "A) [Option A]\n"
                    "B) [Option B]\n"
                    "C) [Option C]\n"
                    "D) [Option D]\n"
                    "Correct: [Letter of correct answer]"
                ),
                input_variables=["context"],
            ),
        }

    def generate_exam_question(self, custom_prompt=None, num_contexts=3):
        """
        Generates a legal exam question using the LLM based on randomly selected contexts
        from the FAISS index.
        """
        try:
            # Randomly sample documents from the vector store
            all_docs = self.vectorstore.docstore._dict.values()
            selected_docs = random.sample(
                list(all_docs), min(num_contexts, len(all_docs))
            )
            context = "\n\n".join(
                [
                    f"Metadata: {doc.metadata}\nContent: {doc.page_content}"
                    for doc in selected_docs
                ]
            )

            # Create a question generation prompt
            question_prompt = PromptTemplate(
                template=(
                    "You are a legal patent examiner creating exam questions. "
                    "Based on the following legal context, generate a challenging question "
                    "that tests understanding of patent law concepts.\n\n"
                    "Context:\n{context}\n\n"
                    "Requirements:\n"
                    "1. Question should be specific to the provided context\n"
                    "2. Focus on testing understanding of legal principles\n"
                    "3. Avoid yes/no questions\n"
                    "4. Make it suitable for patent law examination\n\n"
                    "Generate only the question, without any additional commentary."
                ),
                input_variables=["context"],
            )

            # Format the prompt with the context
            formatted_prompt = question_prompt.format(context=context)

            # Generate the question
            generated_question = self.client.generate(
                formatted_prompt,
                temperature=0.7,  # Higher temperature for more creative questions
                max_tokens=200,
            )

            # Store the context for later use in evaluation
            if not hasattr(self, "question_contexts"):
                self.question_contexts = {}
            self.question_contexts[generated_question] = context

            return generated_question

        except Exception as e:
            return f"Error generating question: {str(e)}"

    def evaluate_answer(self, question, student_answer):
        """
        Evaluates a student's answer using the context that was used to generate the question.
        """
        if (
            not hasattr(self, "question_contexts")
            or question not in self.question_contexts
        ):
            return "Error: Cannot find original context for this question."

        evaluation_prompt = PromptTemplate(
            template=(
                "You are an expert patent law examiner evaluating exam answers.\n\n"
                "Original Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Student's Answer:\n{answer}\n\n"
                "Evaluate the answer based on:\n"
                "1. Accuracy according to the legal context\n"
                "2. Completeness of the response\n"
                "3. Understanding of legal principles\n\n"
                "Provide a score (0-10) and brief explanation."
            ),
            input_variables=["context", "question", "answer"],
        )

        formatted_prompt = evaluation_prompt.format(
            context=self.question_contexts[question],
            question=question,
            answer=student_answer,
        )

        return self.client.generate(formatted_prompt, temperature=0.2, max_tokens=300)

    def query(self, question, top_k=20):
        try:
            # Extract article numbers if mentioned in the question
            article_number = None
            article_keywords = set(
                ["definition", "define", "explain", "what is", "meaning"]
            )
            is_definition_query = any(
                keyword in question.lower() for keyword in article_keywords
            )

            if "article" in question.lower():
                article_match = re.search(r"article\s+(\d+)", question.lower())
                if article_match:
                    article_number = article_match.group(1)

            # Enhanced search strategy
            search_queries = []

            # Add article-specific queries
            if article_number:
                search_queries.extend(
                    [
                        f"article {article_number} definition",
                        f"article {article_number} requirements",
                        f"article {article_number} explanation",
                        f"meaning of article {article_number}",
                    ]
                )

            # Add the original question
            search_queries.append(question)

            # Collect documents from all queries
            all_docs = []
            for query in search_queries:
                docs = self.vectorstore.similarity_search(query, k=5)
                all_docs.extend(docs)

            # Remove duplicates while preserving order
            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)

            # Categorize documents with enhanced metadata handling
            primary_docs = []
            definition_docs = []
            guideline_docs = []
            supporting_docs = []

            for doc in unique_docs:
                content = doc.page_content.strip()
                meta = (
                    doc.metadata
                    if isinstance(doc.metadata, dict)
                    else {"source": "EPC"}
                )

                # Enhanced categorization logic
                if article_number and f"article {article_number}" in content.lower():
                    if is_definition_query and any(
                        term in content.lower() for term in article_keywords
                    ):
                        definition_docs.append((meta, content))
                    else:
                        primary_docs.append((meta, content))
                elif "guideline" in str(meta).lower():
                    guideline_docs.append((meta, content))
                elif any(
                    term in content.lower()
                    for term in ["article", "rule", "regulation"]
                ):
                    supporting_docs.append((meta, content))

            # Build structured context
            context_parts = []

            # Add definitions first if requested
            if definition_docs:
                context_parts.append("## Article Definition:")
                for meta, content in definition_docs[:2]:
                    context_parts.append(f"{content}")

            # Add primary content
            if primary_docs:
                context_parts.append("\n## Legal Provisions:")
                for meta, content in primary_docs[:3]:
                    source = meta.get("source", "EPC")
                    context_parts.append(f"From {source}:\n{content}")

            # Add relevant guidelines
            if guideline_docs:
                context_parts.append("\n## Implementation Guidelines:")
                for meta, content in guideline_docs[:2]:
                    context_parts.append(content)

            # Add supporting context
            if supporting_docs:
                context_parts.append("\n## Related Provisions:")
                for meta, content in supporting_docs[:2]:
                    context_parts.append(content)

            # Combine context with markdown formatting
            context = "\n\n".join(context_parts)

            # Enhanced prompt template with markdown
            enhanced_prompt = """You are an expert European Patent Attorney providing comprehensive answers about patent law.
            Format your response using Markdown and following this structure:

            # Answer Summary
            Brief overview of the key points

            ## Legal Framework
            Cite and explain the relevant legal provisions

            ## Detailed Explanation
            Break down the concepts and their implications

            ## Practical Application
            Provide practical context and examples if available

            Context:
            {context}

            Question: {question}

            Answer:"""

            # Generate response
            formatted_prompt = PromptTemplate(
                template=enhanced_prompt, input_variables=["context", "question"]
            ).format(context=context, question=question)

            response = self.client.generate(
                formatted_prompt, temperature=0.2, max_tokens=1500
            )

            return response.strip()

        except ConnectionError:
            return "Error: Cannot connect to Ollama. Please ensure Ollama is running with 'ollama serve' command."
        except Exception as e:
            return f"Error: {str(e)}"
