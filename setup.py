# Run this file locally (streamlit run setup.py) to download the NLTK data (it's a lot). 
# When prompted, type 'd' to download.
# Then, type 'all' to download all packages.
# And finally, 'q' to quit.
# You can then run the demo.py file to open the app.

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

