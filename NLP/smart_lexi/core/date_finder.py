import re


class DateFinder(object):
    patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'(\d{1,2}-\d{1,2}-\d{2})',
        r'(\d{1,2}-\d{1,2}-\d{4})',
        r'(\d{1,2} (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]* \d{4})',
        r'(\d{1,2} (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]* \d{2})',
        r'(\d{1,2} (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{4})',
        r'(\d{1,2} (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{2})',
        r'(\d{1,2}-(?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{4})',
        r'(\d{1,2}-(?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{2})',
        r'(\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2})',
        r'(\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4})',
        r'(\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2})',
        r'(\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4})',
        r'(\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{2})',
        r'(\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})',

        r'(\d{1,2}st (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]* \d{4})',
        r'(\d{1,2}st (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]* \d{2})',
        r'(\d{1,2}st (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{4})',
        r'(\d{1,2}st (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{2})',
        r'(\d{1,2}st-(?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{4})',
        r'(\d{1,2}st-(?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{2})',

        r'(\d{1,2}nd (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]* \d{4})',
        r'(\d{1,2}nd (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]* \d{2})',
        r'(\d{1,2}nd (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{4})',
        r'(\d{1,2}nd (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{2})',
        r'(\d{1,2}nd-(?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{4})',
        r'(\d{1,2}nd-(?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{2})',

        r'(\d{1,2}rd (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]* \d{4})',
        r'(\d{1,2}rd (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]* \d{2})',
        r'(\d{1,2}rd (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{4})',
        r'(\d{1,2}rd (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{2})',
        r'(\d{1,2}rd-(?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{4})',
        r'(\d{1,2}rd-(?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{2})',

        r'(\d{1,2}th (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]* \d{4})',
        r'(\d{1,2}th (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]* \d{2})',
        r'(\d{1,2}th (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{4})',
        r'(\d{1,2}th (?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{2})',
        r'(\d{1,2}th-(?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{4})',
        r'(\d{1,2}th-(?:January|February|March|April|May|Jun|July|August|September|October|November|December)[a-z]*-\d{2})',

        r'(\d{1,2}st (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})',
        r'(\d{1,2}st (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{2})',
        r'(\d{1,2}st (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4})',
        r'(\d{1,2}st (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2})',
        r'(\d{1,2}st-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4})',
        r'(\d{1,2}st-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2})',

        r'(\d{1,2}nd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})',
        r'(\d{1,2}nd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{2})',
        r'(\d{1,2}nd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4})',
        r'(\d{1,2}nd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2})',
        r'(\d{1,2}nd-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4})',
        r'(\d{1,2}nd-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2})',

        r'(\d{1,2}rd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})',
        r'(\d{1,2}rd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{2})',
        r'(\d{1,2}rd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4})',
        r'(\d{1,2}rd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2})',
        r'(\d{1,2}rd-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4})',
        r'(\d{1,2}rd-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2})',

        r'(\d{1,2}rd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})',
        r'(\d{1,2}rd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{2})',
        r'(\d{1,2}rd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4})',
        r'(\d{1,2}rd (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2})',
        r'(\d{1,2}rd-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4})',
        r'(\d{1,2}rd-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2})',

    ]

    @staticmethod
    def extract(text):
        print(text)
        dates = []
        for pattern in DateFinder.patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                dates.append(date_str)
                text = text.replace(date_str, '')
        return dates
