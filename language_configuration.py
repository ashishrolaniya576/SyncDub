from .logging_setup import logger

# Simplified language list focusing on the main languages
LANGUAGES = {
    "Automatic detection": "Automatic detection",
    "English (en)": "en",
    "Hindi (hi)": "hi",
    "French (fr)": "fr",
    "Italian (it)": "it",
    "German (de)": "de",
    "Korean (ko)": "ko",
    "Chinese - Simplified (zh-CN)": "zh",
    "Chinese - Traditional (zh-TW)": "zh-TW",
}

BASE_L_LIST = LANGUAGES.keys()
LANGUAGES_LIST = [list(BASE_L_LIST)[0]] + sorted(list(BASE_L_LIST)[1:])
INVERTED_LANGUAGES = {value: key for key, value in LANGUAGES.items()}

def fix_code_language(translate_to, syntax="google"):
    if syntax == "google":
        # google-translator, gTTS
        replace_lang_code = {"zh": "zh-CN", "he": "iw", "zh-cn": "zh-CN"}
    elif syntax == "coqui":
        # coqui-xtts
        replace_lang_code = {"zh": "zh-cn", "zh-CN": "zh-cn", "zh-TW": "zh-cn"}

    new_code_lang = replace_lang_code.get(translate_to, translate_to)
    logger.debug(f"Fix code {translate_to} -> {new_code_lang}")
    return new_code_lang

# Simplified list of BARK voices for the main languages
BARK_VOICES_LIST = {
    # German BARK voices
    "de_speaker_0-Male BARK": "v2/de_speaker_0",
    "de_speaker_3-Female BARK": "v2/de_speaker_3",
    
    # English BARK voices
    "en_speaker_0-Male BARK": "v2/en_speaker_0",
    "en_speaker_9-Female BARK": "v2/en_speaker_9",
    
    # French BARK voices
    "fr_speaker_0-Male BARK": "v2/fr_speaker_0",
    "fr_speaker_1-Female BARK": "v2/fr_speaker_1",
    
    # Hindi BARK voices
    "hi_speaker_0-Female BARK": "v2/hi_speaker_0",
    "hi_speaker_2-Male BARK": "v2/hi_speaker_2",
    
    # Italian BARK voices
    "it_speaker_0-Male BARK": "v2/it_speaker_0",
    "it_speaker_2-Female BARK": "v2/it_speaker_2",
    
    # Korean BARK voices
    "ko_speaker_0-Female BARK": "v2/ko_speaker_0",
    "ko_speaker_1-Male BARK": "v2/ko_speaker_1",
    
    # Chinese BARK voices
    "zh_speaker_0-Male BARK": "v2/zh_speaker_0",
    "zh_speaker_4-Female BARK": "v2/zh_speaker_4",
}

# Simplified list of VITS voices for the main languages
VITS_VOICES_LIST = {
    "en-facebook-mms VITS": "facebook/mms-tts-eng",
    "fr-facebook-mms VITS": "facebook/mms-tts-fra",
    "de-facebook-mms VITS": "facebook/mms-tts-deu",
    "it-facebook-mms VITS": "facebook/mms-tts-ita",
    "ko-facebook-mms VITS": "facebook/mms-tts-kor",
    "hi-facebook-mms VITS": "facebook/mms-tts-hin",
    "zh_Hakka-facebook-mms VITS": "facebook/mms-tts-hak",
    "zh_MinNan-facebook-mms VITS": "facebook/mms-tts-nan",
}


# Three-letter language codes for the focus languages
LANGUAGE_CODE_IN_THREE_LETTERS = {
    "Automatic detection": "aut",
    "en": "eng",
    "hi": "hin",
    "fr": "fre",
    "it": "ita",
    "de": "ger",
    "ko": "kor",
    "zh": "chi",
    "zh-TW": "chi",
}
