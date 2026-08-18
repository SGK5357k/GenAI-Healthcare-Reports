import PyPDF2


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF file.
    """

    try:

        reader = PyPDF2.PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                pages.append(page_text)

        return "\n".join(pages)

    except Exception as e:

        return f"PDF extraction failed: {str(e)}"