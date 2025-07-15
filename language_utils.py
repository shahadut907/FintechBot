"""
Language detection and translation utilities for the RAG service.

This module provides utilities for detecting the language of user queries
and translating text between languages to ensure consistent handling
regardless of input language.
"""

import logging
from typing import Optional, Dict, Any, Tuple, List
import re
import os
from pathlib import Path

# Language detection
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Translation utilities
from deep_translator import GoogleTranslator

# Set seed for reproducible language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# Common language mappings between different standards
LANGUAGE_CODE_MAPPING = {
    # ISO 639-1 to full name
    'en': 'english',
    'bn': 'bengali',
    'ar': 'arabic',
    'fr': 'french',
    'de': 'german',
    'es': 'spanish',
    'hi': 'hindi',
    'ja': 'japanese',
    'ko': 'korean',
    'ru': 'russian',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
}

# Common phrases in different languages for better detection
LANGUAGE_PHRASES = {
    'bn': [
        'kemon acho', 'kemon achen', 'ki khobor', 'bhalo achi', 'dhonnobad',
        'ami', 'tumi', 'apni', 'tomra', 'amra', 'ki', 'keno', 'kothay',
        'bangla', 'বাংলা', 'কেমন আছেন', 'কেমন আছো', 'কি খবর', 'ভালো আছি', 'আমি', 'তুমি'
    ],
    'en': [
        'hi', 'hey', 'hello', 'good morning', 'good afternoon', 'good evening', 
        'how are you', 'what\'s up', 'sup', 'yo', 'thanks', 'thank you',
        'yes', 'no', 'maybe', 'please', 'welcome', 'bye', 'goodbye'
    ]
}

# Last detected language by session to maintain consistency
SESSION_LANGUAGES = {}

# Default to English for new sessions
DEFAULT_LANGUAGE = 'en'

def detect_language(text: str, session_id: str = None) -> Tuple[str, float]:
    """
    Detect the language of a given text with improved pattern recognition.
    
    Args:
        text: The text to analyze
        session_id: Optional session ID to track language preferences
        
    Returns:
        A tuple containing (language_code, confidence_score)
    """
    if not text or len(text.strip()) < 3:
        # For very short texts, check common greetings first
        text_lower = text.lower().strip()
        
        # Check for common English greetings
        if text_lower in ['hi', 'hey', 'hello', 'yo', 'sup']:
            if session_id:
                SESSION_LANGUAGES[session_id] = 'en'
            return 'en', 0.9
            
        # Use the last detected language if available
        if session_id and session_id in SESSION_LANGUAGES:
            return SESSION_LANGUAGES[session_id], 0.7
        return DEFAULT_LANGUAGE, 0.8  # Default to English
    
    # Check for language-specific patterns first (more reliable for short texts)
    text_lower = text.lower().strip()
    
    # Check for English common phrases
    for phrase in LANGUAGE_PHRASES.get('en', []):
        if phrase == text_lower or text_lower.startswith(phrase + ' ') or text_lower.endswith(' ' + phrase):
            if session_id:
                SESSION_LANGUAGES[session_id] = 'en'
            return 'en', 0.9
    
    # Check for Bangla phrases
    for phrase in LANGUAGE_PHRASES.get('bn', []):
        if phrase in text_lower or text_lower in phrase:
            if session_id:
                SESSION_LANGUAGES[session_id] = 'bn'
            return 'bn', 0.9
    
    # Look for language change commands
    if any(marker in text_lower for marker in ['in bangla', 'switch to bangla', 'translate to bangla', 'continue with bangla', 'speak bangla']):
        if session_id:
            SESSION_LANGUAGES[session_id] = 'bn'
        return 'bn', 0.95
    
    if any(marker in text_lower for marker in ['in english', 'switch to english', 'translate to english', 'continue with english', 'speak english']):
        if session_id:
            SESSION_LANGUAGES[session_id] = 'en'
        return 'en', 0.95
    
    # Retain the current session language if command indicates continuation
    if session_id and session_id in SESSION_LANGUAGES:
        if any(marker in text_lower for marker in ['continue', 'keep going', 'go on']):
            return SESSION_LANGUAGES[session_id], 0.8
    
    try:
        # Use standard language detection for longer texts
        lang_code = detect(text)
        
        # Safety check - override for commonly misdetected short phrases
        if len(text) < 10 and lang_code not in ['en', 'bn']:
            # For very short messages, default to English unless explicitly detected as Bangla
            logger.warning(f"Short message '{text}' detected as {lang_code}, defaulting to English")
            lang_code = 'en'
        
        # Store in session if provided
        if session_id:
            SESSION_LANGUAGES[session_id] = lang_code
        
        # Always return the ISO 639-1 code
        return lang_code, 0.9  # langdetect doesn't provide confidence scores
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}")
        
        # Fall back to session language if available
        if session_id and session_id in SESSION_LANGUAGES:
            return SESSION_LANGUAGES[session_id], 0.6
        
        return DEFAULT_LANGUAGE, 0.5  # Default to English with low confidence
    except Exception as e:
        logger.error(f"Unexpected error in language detection: {e}")
        return DEFAULT_LANGUAGE, 0.5  # Default to English with low confidence

def translate_text(text: str, target_lang: str = 'en', source_lang: Optional[str] = None) -> str:
    """
    Translate text from source language to target language.
    
    Args:
        text: The text to translate
        target_lang: Target language code (ISO 639-1)
        source_lang: Source language code (if None, auto-detected)
        
    Returns:
        Translated text
    """
    if not text or len(text.strip()) < 3:
        return text
    
    if target_lang == source_lang:
        return text
        
    try:
        # Initialize translator
        translator = GoogleTranslator(
            source=source_lang or 'auto', 
            target=target_lang
        )
        
        # Special handling for Bangla (bn) as target
        if target_lang == 'bn':
            common_responses = {
                'hello': 'হ্যালো',
                'hi': 'হাই',
                'how are you': 'কেমন আছেন',
                'how are you doing': 'কেমন আছো',
                "i'm fine": 'আমি ভালো আছি',
                'thank you': 'ধন্যবাদ',
                'welcome': 'স্বাগতম',
                'yes': 'হ্যাঁ',
                'no': 'না',
                "i don't have that information": 'আমার কাছে এই তথ্য নেই',
                "i don't know": 'আমি জানি না',
                "can i help you with something else": 'আমি কি আপনাকে অন্য কিছু সাহায্য করতে পারি?'
            }
            
            text_lower = text.lower().strip()
            for eng, ban in common_responses.items():
                if eng in text_lower:
                    return ban
        
        # Translate text
        translated = translator.translate(text)
        return translated
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  # Return original text on error
        
def translate_content_blocks(search_results: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
    """
    Translate content blocks retrieved from vector store.
    Includes batch processing and error handling to ensure reliable translations.
    
    Args:
        search_results: List of document chunks from the vector store
        target_lang: Target language code (ISO 639-1)
        
    Returns:
        Translated search results
    """
    if not search_results or target_lang == 'en':
        return search_results
    
    # Group content for batch translation to improve performance
    contents_to_translate = []
    for result in search_results:
        if 'content' in result and result['content'] and isinstance(result['content'], str):
            contents_to_translate.append(result['content'])
    
    # Skip if nothing to translate
    if not contents_to_translate:
        return search_results
    
    try:
        # Initialize translator
        translator = GoogleTranslator(
            source='en',  # Always translate from English
            target=target_lang
        )
        
        # Translate content in smaller batches to avoid errors
        translated_contents = []
        batch_size = 5  # Process in small batches
        
        for i in range(0, len(contents_to_translate), batch_size):
            batch = contents_to_translate[i:i+batch_size]
            
            try:
                # Try batch translation first
                batch_translations = translator.translate_batch(batch)
                translated_contents.extend(batch_translations)
            except Exception:
                # Fall back to individual translation on batch failure
                for text in batch:
                    try:
                        translated = translator.translate(text)
                        translated_contents.append(translated)
                    except Exception:
                        # Keep original text if translation fails
                        translated_contents.append(text)
                        logger.warning(f"Translation failed for text block, keeping original")
        
        # Update results with translations
        translated_results = []
        translation_index = 0
        
        for result in search_results:
            translated_result = result.copy()
            
            if 'content' in result and result['content'] and isinstance(result['content'], str):
                if translation_index < len(translated_contents):
                    translated_result['content'] = translated_contents[translation_index]
                    translation_index += 1
            
            translated_results.append(translated_result)
        
        logger.info(f"Successfully translated {translation_index} content blocks to {target_lang}")
        return translated_results
        
    except Exception as e:
        logger.error(f"Error in batch translation: {e}")
        # Return original results if translation fails
        return search_results

def normalize_language_name(language: str) -> str:
    """
    Normalize language name or code to a standard form.
    
    Args:
        language: Language name or ISO code
        
    Returns:
        Normalized language name or code
    """
    language = language.lower().strip()
    
    # If it's a language code, try to map to full name
    if language in LANGUAGE_CODE_MAPPING:
        return LANGUAGE_CODE_MAPPING[language]
    
    # Otherwise return as is
    return language 

def get_session_language(session_id: str) -> Optional[str]:
    """Get the currently active language for a session"""
    return SESSION_LANGUAGES.get(session_id)

def set_session_language(session_id: str, language: str) -> None:
    """Set the active language for a session"""
    SESSION_LANGUAGES[session_id] = language

def reset_session_language(session_id: str) -> None:
    """Reset the language for a session to default"""
    if session_id in SESSION_LANGUAGES:
        del SESSION_LANGUAGES[session_id] 