"""
RAGMill Core Engine

Provides directory crawling, semantic splitting, and pipeline execution.
"""

import logging
import os
import re
from typing import List, Dict, Any, Generator

from ragmill.parsers import extract_pdf_text, extract_docx_text

logger = logging.getLogger(__name__)

PLAIN_TEXT_EXTENSIONS = (".txt", ".md", ".log", ".rst")
PDF_EXTENSIONS = (".pdf",)
DOCX_EXTENSIONS = (".docx",)


class RAGEngine:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initializes the RAG engine.

        :param chunk_size: Maximum character size per chunk.
        :param overlap: Character overlap between consecutive chunks for context.
        """
        self.chunk_size = chunk_size
        self.overlap = max(0, overlap)

        if self.overlap >= self.chunk_size:
            raise ValueError(
                "Overlap threshold cannot be greater than or equal to total chunk size."
            )

    def stream_directory(self, directory_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Walks a directory and yields extracted file metadata for supported formats.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(
                f"Target path tracking validation failed for: '{directory_path}'"
            )

        for root, _, files in os.walk(directory_path):
            for file in files:
                extension = os.path.splitext(file)[1].lower()
                if extension not in PLAIN_TEXT_EXTENSIONS + PDF_EXTENSIONS + DOCX_EXTENSIONS:
                    continue

                full_path = os.path.join(root, file)
                try:
                    content = self._extract_content(full_path, extension)

                    yield {
                        "source_path": os.path.abspath(full_path),
                        "filename": file,
                        "raw_content": content.strip(),
                        "modified_at": os.path.getmtime(full_path),
                    }
                except Exception as e:
                    logger.warning("Unable to parse file %s: %s", full_path, e, exc_info=True)

    def _extract_content(self, full_path: str, extension: str) -> str:
        if extension in PLAIN_TEXT_EXTENSIONS:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if extension in PDF_EXTENSIONS:
            return extract_pdf_text(full_path)
        if extension in DOCX_EXTENSIONS:
            return extract_docx_text(full_path)
        raise ValueError(f"Unsupported extension: {extension}")

    def semantic_chunking(self, text: str) -> List[str]:
        """
        Recursively splits text payloads based on logical paragraph, structural,
        and grammatical sentence boundaries to protect semantic context integrity.
        """
        if not text:
            return []

        # Split along logical structural breaks (paragraphs, list blocks, markdown breaks)
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        current_buffer = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Handle edge cases where single paragraphs wildly exceed standard target size limits
            if len(paragraph) > self.chunk_size:
                # If buffer already holds content, clear it to start clean sentence processing
                if current_buffer:
                    chunks.append(current_buffer.strip())
                    current_buffer = ""

                # Split down into sentence tokens
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    if len(current_buffer) + len(sentence) + 1 <= self.chunk_size:
                        current_buffer = f"{current_buffer} {sentence}".strip()
                    else:
                        if current_buffer:
                            chunks.append(current_buffer)

                        # Handle long sentence edge case: verify slice safety before copying historical context
                        overlap_prefix = (
                            current_buffer[-self.overlap :]
                            if len(current_buffer) >= self.overlap
                            else current_buffer
                        )
                        current_buffer = (
                            f"{overlap_prefix} {sentence}".strip() if self.overlap > 0 else sentence
                        )
            else:
                # Standard appending logic for typical sized semantic paragraphs
                spacing = "\n\n" if current_buffer else ""
                if len(current_buffer) + len(spacing) + len(paragraph) <= self.chunk_size:
                    current_buffer = f"{current_buffer}{spacing}{paragraph}"
                else:
                    if current_buffer:
                        chunks.append(current_buffer.strip())

                    # Establish overlap baseline from preceding content block
                    overlap_prefix = (
                        current_buffer[-self.overlap :]
                        if len(current_buffer) >= self.overlap
                        else current_buffer
                    )
                    current_buffer = (
                        f"{overlap_prefix}\n\n{paragraph}".strip()
                        if self.overlap > 0
                        else paragraph
                    )

        if current_buffer:
            chunks.append(current_buffer.strip())

        # Hard fallback: enforce chunk_size by slicing any oversized chunk
        # into windows of chunk_size with overlap.
        result: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                result.append(chunk)
            else:
                # Slice into overlapping windows of chunk_size
                step = max(1, self.chunk_size - self.overlap)
                for start in range(0, len(chunk), step):
                    window = chunk[start : start + self.chunk_size]
                    if window:
                        result.append(window)

        return result

    def execute_pipeline(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Compiles the full ingestion and data structure cycle across a target directory.

        :return: Array containing distinct dictionary models containing contextual metadata maps.
        """
        pipeline_payloads: List[Dict[str, Any]] = []

        for file_manifest in self.stream_directory(directory_path):
            text_chunks = self.semantic_chunking(file_manifest["raw_content"])

            for index, chunk in enumerate(text_chunks):
                pipeline_payloads.append(
                    {
                        "metadata": {
                            "source_file": file_manifest["source_path"],
                            "filename": file_manifest["filename"],
                            "chunk_index": index,
                            "character_length": len(chunk),
                            "modified_at": file_manifest["modified_at"],
                        },
                        "content": chunk,
                    }
                )

        return pipeline_payloads
