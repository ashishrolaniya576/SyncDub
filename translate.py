import json
import copy
import logging
import time
from typing import List, Dict, Optional, Any, Union
from itertools import chain
from tqdm import tqdm
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# Add this at the top of your translate.py file
# Load environment variables from .env file
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Language code constants - simplified for now
ISO_LANGUAGE_CODES = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "ar": "arabic",
}

def fix_language_code(language_code: Optional[str]) -> str:
    """Convert language code to format compatible with translator."""
    if not language_code:
        return "auto"
    
    # Clean up language code (remove region specifiers)
    language_code = language_code.lower().split('-')[0]
    
    # Return the cleaned code if it's in our list, otherwise default to auto
    return language_code if language_code in ISO_LANGUAGE_CODES else "auto"

def translate_iterative(segments: List[Dict[str, Any]], 
                        target_lang: str, 
                        source_lang: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Translate text segments individually to the specified language.

    Args:
        segments: List of dictionaries with 'text' key containing the text to translate
        target_lang: Target language code
        source_lang: Source language code (defaults to auto-detect)

    Returns:
        List of segments with translated text
    """
    segments_copy = copy.deepcopy(segments)
    source = fix_language_code(source_lang)
    target = fix_language_code(target_lang)
    
    logger.info(f"Translating {len(segments)} segments from {source} to {target} (iterative)")
    translator = GoogleTranslator(source=source, target=target)

    for i, segment in enumerate(tqdm(segments_copy, desc="Translating")):
        text = segment["text"].strip()
        try:
            translated_text = translator.translate(text)
            segments_copy[i]["text"] = translated_text
        except Exception as error:
            logger.error(f"Error translating segment {i}: {error}")
            # Keep original text if translation fails
            segments_copy[i]["text"] = text

    return segments_copy

def verify_translation(original_segments: List[Dict[str, Any]],
                      segments_copy: List[Dict[str, Any]],
                      translated_lines: List[str],
                      target_lang: str,
                      source_lang: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Verify translation integrity and assign translated text to segments.
    Falls back to iterative translation if segment counts don't match.
    """
    if len(original_segments) == len(translated_lines):
        for i in range(len(segments_copy)):
            segments_copy[i]["text"] = translated_lines[i].replace("\t", " ").replace("\n", " ").strip()
        return segments_copy
    else:
        logger.error(
            f"Translation failed: segment count mismatch. Original: {len(original_segments)}, "
            f"Translated: {len(translated_lines)}. Switching to iterative translation."
        )
        return translate_iterative(original_segments, target_lang, source_lang)

def translate_batch(segments: List[Dict[str, Any]], 
                   target_lang: str, 
                   chunk_size: int = 4000,
                   source_lang: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Translate a batch of text segments in chunks to respect API limits.

    Args:
        segments: List of dictionaries with 'text' key
        target_lang: Target language code
        chunk_size: Maximum character count per chunk (default: 4000)
        source_lang: Source language code (defaults to auto-detect)

    Returns:
        List of segments with translated text
    """
    segments_copy = copy.deepcopy(segments)
    source = fix_language_code(source_lang)
    target = fix_language_code(target_lang)
    
    logger.info(f"Translating {len(segments)} segments from {source} to {target} (batch)")

    # Extract text from segments
    text_lines = [segment["text"].strip() for segment in segments]
    
    # Create chunks respecting character limit
    text_chunks = []
    current_chunk = ""
    chunk_segments = []
    segment_tracking = []
    
    for line in text_lines:
        line = " " if not line else line
        if (len(current_chunk) + len(line) + 7) <= chunk_size:  # 7 for separator
            if current_chunk:
                current_chunk += " ||||| "
            current_chunk += line
            chunk_segments.append(line)
        else:
            text_chunks.append(current_chunk)
            segment_tracking.append(chunk_segments)
            current_chunk = line
            chunk_segments = [line]
    
    if current_chunk:
        text_chunks.append(current_chunk)
        segment_tracking.append(chunk_segments)

    # Translate chunks
    translator = GoogleTranslator(source=source, target=target)
    translated_segments = []
    progress_bar = tqdm(total=len(segments), desc="Translating")
    
    try:
        for chunk_text, chunk_segments in zip(text_chunks, segment_tracking):
            translated_chunk = translator.translate(chunk_text.strip())
            split_translations = translated_chunk.split("|||||")
            
            # Verify chunk integrity
            if len(split_translations) == len(chunk_segments):
                progress_bar.update(len(split_translations))
                translated_segments.extend([t.strip() for t in split_translations])
            else:
                logger.warning(
                    f"Chunk translation mismatch. Expected {len(chunk_segments)}, "
                    f"got {len(split_translations)}. Translating segment by segment."
                )
                for segment in chunk_segments:
                    translated_text = translator.translate(segment.strip())
                    translated_segments.append(translated_text.strip())
                    progress_bar.update(1)
        
        progress_bar.close()
        
        # Verify and return
        return verify_translation(segments, segments_copy, translated_segments, target_lang, source_lang)
    
    except Exception as error:
        progress_bar.close()
        logger.error(f"Batch translation failed: {error}")
        return translate_iterative(segments, target_lang, source_lang)

def translate_with_groq(segments: List[Dict[str, Any]],
                       target_lang: str,
                       model_name: str = "llama-3.3-70b-versatile",
                       source_lang: Optional[str] = None,
                       batch_size: int = 10) -> List[Dict[str, Any]]:
    """
    Translate text segments using Groq API.
    
    Args:
        segments: List of dictionaries with 'text' key
        target_lang: Target language code
        model_name: Groq model to use (default: "llama-3.3-70b-versatile")
        source_lang: Source language code (optional)
        batch_size: Number of segments to process in each API call
        
    Returns:
        List of segments with translated text
    """
    segments_copy = copy.deepcopy(segments)
    
    # Get language names instead of codes for clarity in prompting
    target_language = ISO_LANGUAGE_CODES.get(fix_language_code(target_lang), "the target language")
    source_language = "auto-detected language"
    if source_lang:
        source_language = ISO_LANGUAGE_CODES.get(fix_language_code(source_lang), "the source language")
    
    logger.info(f"Translating {len(segments)} segments from {source_language} to {target_language} using Groq")
    
    # Set up Groq LLM
    llm = ChatGroq(model_name=model_name, temperature=0.2)
    
    # Process segments in batches
    translated_segments = []
    total_batches = (len(segments) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Translating batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(segments))
        batch = segments[start_idx:end_idx]
        
        # Extract text from segments
        batch_texts = [segment["text"].strip() for segment in batch]
        
        # Create numbered text array for the prompt
        numbered_texts = [f"{i+1}. {text}" for i, text in enumerate(batch_texts)]
        batch_content = "\n".join(numbered_texts)
        
        # Create a prompt template for translation
        template = """
        You are a professional translator. Translate the following text segments from {source_language} to {target_language}.
        
        IMPORTANT INSTRUCTIONS:
        1. Preserve the meaning, tone, and style of the original text
        2. Only respond with JSON in the exact format shown below
        3. Each numbered segment should be translated separately
        4. Maintain the original numbering in your response
        
        Text to translate:
        {text_segments}
        
        The response should be ONLY a JSON array with this exact structure:
        [
          "translated segment 1",
          "translated segment 2",
          ...
        ]
        """
        
        prompt = ChatPromptTemplate.from_messages([("system", template)])
        
        try:
            # Create a chain and execute
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(
                source_language=source_language,
                target_language=target_language,
                text_segments=batch_content
            )
            
            # Parse the response
            # First try to find JSON in the response using regex
            import re
            json_match = re.search(r'\[.*\]', response.strip(), re.DOTALL)
            
            if json_match:
                try:
                    translated_texts = json.loads(json_match.group(0))
                except:
                    # If regex json extraction fails, try direct parsing
                    translated_texts = json.loads(response.strip())
            else:
                # If no JSON array found, try to parse directly
                translated_texts = json.loads(response.strip())
            
            # Verify correct count
            if len(translated_texts) != len(batch):
                logger.warning(
                    f"Translation count mismatch. Expected {len(batch)}, "
                    f"got {len(translated_texts)}. Falling back to Google Translate for this batch."
                )
                # Fall back to Google for this batch
                fallback_translations = translate_iterative(batch, target_lang, source_lang)
                translated_texts = [segment["text"] for segment in fallback_translations]
            
            # Add translations to the result
            translated_segments.extend(translated_texts)
            
            # Avoid hitting rate limits
            time.sleep(0.5)
            
        except Exception as error:
            logger.error(f"Groq translation error for batch {batch_idx+1}/{total_batches}: {error}")
            logger.warning("Falling back to Google Translate for this batch")
            
            # Fall back to Google for this batch
            fallback_translations = translate_iterative(batch, target_lang, source_lang)
            batch_translations = [segment["text"] for segment in fallback_translations]
            translated_segments.extend(batch_translations)
    
    # Verify and update segments
    return verify_translation(segments, segments_copy, translated_segments, target_lang, source_lang)

def translate_text(segments: List[Dict[str, Any]],
                  target_lang: str,
                  translation_method: str = "groq",
                  chunk_size: int = 4000,
                  source_lang: Optional[str] = None,
                  groq_model: str = "llama-3.3-70b-versatile",
                  groq_batch_size: int = 10) -> List[Dict[str, Any]]:
    """
    Main translation function that handles different translation methods.
    
    Args:
        segments: List of dictionaries with 'text' key
        target_lang: Target language code
        translation_method: "batch", "iterative", or "groq" (default: "batch")
        chunk_size: Maximum character count per chunk for batch translation
        source_lang: Source language code (defaults to auto-detect)
        groq_model: Model name for Groq translation
        groq_batch_size: Batch size for Groq translation
        
    Returns:
        List of segments with translated text
    """
    if not segments:
        logger.warning("No segments to translate")
        return segments
    
    if translation_method == "batch":
        return translate_batch(segments, target_lang, chunk_size, source_lang)
    elif translation_method == "iterative":
        return translate_iterative(segments, target_lang, source_lang)
    elif translation_method == "groq":
        return translate_with_groq(
            segments, 
            target_lang, 
            model_name=groq_model,
            source_lang=source_lang,
            batch_size=groq_batch_size
        )
    else:
        logger.error(f"Unknown translation method: {translation_method}")
        return translate_batch(segments, target_lang, chunk_size, source_lang)
    
def generate_srt_subtitles(segments, output_file="output.srt"):
    """
    Generate an SRT subtitle file from translated segments.
    
    Args:
        segments: List of dictionaries with 'start', 'end', and 'text' keys
        output_file: Path to the output SRT file
        
    Returns:
        Path to the created SRT file
    """
    logger.info(f"Generating SRT subtitle file: {output_file}")
    
    # Format time as HH:MM:SS,mmm
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            # Extract timing information
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            # Skip empty segments
            if not text:
                continue
                
            # Write subtitle entry
            f.write(f"{i}\n")
            f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            f.write(f"{text}\n\n")
    
    logger.info(f"SRT subtitle file created successfully: {output_file}")
    return output_file
