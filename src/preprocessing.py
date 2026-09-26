import re
import unidecode
import polars as pl

# Common legal suffixes to normalize or trim
LEGAL_SUFFIXES = {
    r'\bprivate limited\b': 'pvt ltd',
    r'\bpvt limited\b': 'pvt ltd',
    r'\bprvt ltd\b': 'pvt ltd',
    r'\bprivate ltd\b': 'pvt ltd',
    r'\bcorporation\b': 'corp',
    r'\bincorporated\b': 'inc',
    r'\blimited\b': 'ltd',
    r'\bcompany\b': 'co',
    r'\blimited liability company\b': 'llc',
    r'\blimited liability partnership\b': 'llp',
}

# Common address abbreviations
ADDRESS_ABBREVS = {
    r'\bstreet\b': 'st',
    r'\broad\b': 'rd',
    r'\bavenue\b': 'ave',
    r'\bboulevard\b': 'blvd',
    r'\bdrive\b': 'dr',
    r'\bplace\b': 'pl',
    r'\bterrace\b': 'ter',
    r'\bbuilding\b': 'bldg',
    r'\bfloor\b': 'fl',
    r'\bsuite\b': 'ste',
    r'\bapartment\b': 'apt',
}

def clean_text(text: str) -> str:
    """Normalize text: accents, lowercase, legal suffixes, address abbrevs, punctuation."""
    if not text or text == "null" or text == "None":
        return ""
    
    # Accent normalization / ASCII transliteration
    text = unidecode.unidecode(text)
    text = text.lower()
    
    # Replace separators with spaces
    text = re.sub(r'[\/\\,._\-()&:]', ' ', text)
    
    # Legal suffixes normalization
    for pattern, repl in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, repl, text)
        
    # Address abbreviations normalization
    for pattern, repl in ADDRESS_ABBREVS.items():
        text = re.sub(pattern, repl, text)
        
    # Remove remaining non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Clean whitespace
    return ' '.join(text.split())

def extract_numbers(text: str) -> str:
    """Extract all digit sequences from text separated by space."""
    if not text:
        return ""
    nums = re.findall(r'\d+', text)
    return ' '.join(nums)

def get_sorted_tokens(text: str) -> str:
    """Return space-separated alphabetically sorted unique tokens."""
    if not text:
        return ""
    tokens = sorted(set(text.split()))
    return ' '.join(tokens)

def preprocess_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply text preprocessing to polars DataFrame containing business_name and business_address.
    Adds clean_name, clean_address, sorted_name, address_nums columns.
    """
    # Use python map or map_elements for string cleaning
    clean_name_col = df['business_name'].fill_null('').map_elements(clean_text, return_dtype=pl.String)
    clean_addr_col = df['business_address'].fill_null('').map_elements(clean_text, return_dtype=pl.String)
    
    sorted_name_col = clean_name_col.map_elements(get_sorted_tokens, return_dtype=pl.String)
    addr_nums_col = clean_addr_col.map_elements(extract_numbers, return_dtype=pl.String)
    
    return df.with_columns([
        clean_name_col.alias('clean_name'),
        clean_addr_col.alias('clean_address'),
        sorted_name_col.alias('sorted_name'),
        addr_nums_col.alias('address_nums')
    ])
