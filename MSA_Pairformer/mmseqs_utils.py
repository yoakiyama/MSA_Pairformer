# Utility functions for querying the MMseqs2 server and parsing/pairing MSAs
# (c) 2025, Yo Akiyama
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional
from string import ascii_uppercase
from typing import List

class ColabFoldPairedMSA:
    """
    Get paired MSAs from ColabFold with extended filtering and genomic distance support
    """
    def __init__(
        self,
        host_url: str = "https://api.colabfold.com",
        cache_dir: Optional[str] = None
    ):
        self.host_url = host_url
        self.job_id = None
        self.parsed_entries = None  # List of parsed entries with metadata

        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = Path.home() / ".colabfold_cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Cache directory: {self.cache_dir}")

        # Initialize UniProt converter
        self._init_uniprot_converter()

    def _init_uniprot_converter(self):
        f"""
        Initialize UniProt ID to number conversion tables
        UniProt IDs have specific formats:
        - Standard format: [OPQ][0-9][A-Z,0-9]{3}[0-9] (e.g., P12345, Q8N726) 6 characters
        - Secondary format: [A-N,R-Z][0-9][A-Z][A-Z,0-9]{2}[0-9] (e.g., A2BC19) 6 characters (Follows the same format as the first 6 characters of the Extended format)
        - Extended format: [A-N,R-Z][0-9][A-Z][A-Z,0-9]{2}[0-9][A-Z][A-Z,0-9]{2}[0-9] (e.g., A0A023GPI8) 10 characters (Last 4 chars follow the same format as the first 4 characters of the Secondary format)
        - UPI format: UPI[0-9A-F]{10} (e.g., UPI0000000001) 13 characters, last 10 are hexadecimal

        IDs will be processed in reverse order (from last to first character). For Extended format, last 4 characters will be processed the same way as the last 4 characters of the Secondary format.
        Then the rest of the Extended format ID (6 characters) will be processed as the 6 characters of the Secondary format
        UPI IDs will be processed as hexadecimal numbers (0-F)
        """
        # Initialize format detector
        # Stores index for converter table
        # Secondary and Extended format: 0, Standard format: 1
        self.format_detector = {a: 0 for a in ascii_uppercase}
        for a in ["O", "P", "Q"]:
            self.format_detector[a] = 1

        # ma[format_type][position][character] = value
        # Format type: 0 = secondary and extended format, 1 = standard format
        # Position: 0-5 for each character position in the reversed ID (6 characters)
        # List[List[Dict], List[Dict]]
        self.ma = [[{} for k in range(6)], [{} for k in range(6)]]

        # Fill encoding tables
        # Positions 0 and 4 are digits 1-9 (last character and 2nd character)
        for n in range(10):
            for i in [0, 1]:
                for j in [0, 4]:
                    self.ma[i][j][str(n)] = n
        # Positions 1 and 2 are letters A-Z and 0-9 and are assigned numbers 0-35 (4th and 5th characters and 8th and 9th of Extended)
        for n, ascii_char in enumerate(list(ascii_uppercase) + list(range(10))):
            for i in [0, 1]:
                for j in [1, 2]:
                    self.ma[i][j][str(ascii_char)] = n
            # For Standard format, position 3 is also A-Z and 0-9 (3rd character of Standard)
            self.ma[1][3][str(ascii_char)] = n
        # For Secondary and Extended format, position 3 is A-Z (3rd of Secondary/Extended and 7th of Extended)
        # For all formats, position 5 is A-Z (first position for all formats)
        for n, ascii_char in enumerate(ascii_uppercase):
            self.ma[0][3][str(ascii_char)] = n
            for i in [0, 1]:
                self.ma[i][5][str(ascii_char)] = n

        # Separate encoding table for UPI IDs
        self.upi_encoding = {}
        hex_chars = list(range(10)) + ['A', 'B', 'C', 'D', 'E', 'F']
        for n, char in enumerate(hex_chars):
            self.upi_encoding[str(char)] = n
    
    def _extract_uniprot_id(self, header: str) -> str:
        """
        Extract UniProt ID from header.
        """
        pos = header.find("UniRef")
        if pos == -1: # UniRef not in header
            return ""

        start = header.find('_', pos)
        if start == -1: # No underscore after UniRef in header
            return ""
        start += 1

        end = start
        while end < len(header) and header[end] not in ' _\t': # Find end of header string or before a space, underscore, or tab
            end += 1

        uid = header[start:end] # Take everything between underscore and end

        # Validate - including UPI IDs
        if len(uid) >= 3 and uid[:3] == "UPI":
            return uid

        # Regular UniProt ID validation
        if len(uid) not in [6, 10]: # If not a UPI ID, must be a 6 or 10 character UniProt ID
            return ""
        if not uid[0].isalpha(): # First character must be a letter
            return ""
        return uid

    def _uniprot_to_number(self, uniprot_ids: List[str]) -> List[int]:
        """
        Convert UniProt IDs to numbers for distance calculation using uniprot converter tables
        """
        numbers = []
        # Iterate over each UniProt ID and convert to a number
        for uni in uniprot_ids:
            if not uni or not uni[0].isalpha(): # First character must be a letter, otherwise skip and just assign 0 (maybe use NaNs here instead)
                numbers.append(0)
                continue

            # Handle UPI IDs
            if uni.startswith("UPI") and len(uni) == 13:
                hex_part = uni[3:]  # Remove "UPI" prefix
                num = 0
                tot = 1

                # Process hexadecimal characters in reverse order
                for u in reversed(hex_part):
                    if str(u) in self.upi_encoding:
                        num += self.upi_encoding[str(u)] * tot
                        tot *= 16  # Base 16 for hexadecimal
                    else:
                        # Invalid hex character, assign 0
                        num = 0
                        break
                # Add large offset to distinguish UPI IDs from standard ones
                numbers.append(num + 10**12)
                continue

            # Handle standard and secondary/extended UniProt IDs
            id_format = self.format_detector.get(uni[0], 0)
            tot, num = 1, 0
            # Last 4 characters of Extended format
            if len(uni) == 10:
                for n, u in enumerate(reversed(uni[-4:])):
                    if str(u) in self.ma[id_format][n]:
                        num += self.ma[id_format][n][str(u)] * tot
                        tot *= len(self.ma[id_format][n].keys())
            # First 6 characters of standard/secondary/extended format
            for n, u in enumerate(reversed(uni[:6])):
                if n < len(self.ma[id_format]) and str(u) in self.ma[id_format][n]:
                    num += self.ma[id_format][n][str(u)] * tot
                    tot *= len(self.ma[id_format][n].keys())
            # Add large offsets
            if len(uni) == 10:
                num += 10**13
            elif len(uni) == 6 and id_format == 1:
                num += 10**9
            numbers.append(num)
        return numbers