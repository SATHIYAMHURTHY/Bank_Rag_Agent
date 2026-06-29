"""
sources.py
----------
Registry of bank policy/scheme pages to scrape.
Each entry's metadata gets attached to every chunk extracted from that page.

js_render: True  = needs Playwright (JS-rendered page, aiohttp returns near-zero content)
js_render: False = works with plain aiohttp request (static HTML)

Banks covered (11 total):
    Public sector  : SBI, Bank of Baroda, Punjab National Bank,
                     Canara Bank, Union Bank of India,
                     Indian Overseas Bank, Indian Bank
    Private sector : HDFC, ICICI, Axis Bank, City Union Bank
"""

SOURCES = [

    # ── HDFC Bank (static HTML, aiohttp works) ───────────────────────────────
    {
        "url": "https://www.hdfcbank.com/personal/borrow/popular-loans/educational-loan/educational-loan-for-indian-education",
        "bank": "HDFC",
        "scheme_type": "education_loan",
        "scheme_name": "Education Loan for Indian Education",
        "doc_type": "product_page",
        "js_render": False,
    },
    {
        "url": "https://www.hdfcbank.com/personal/borrow/popular-loans/educational-loan/education-loan-for-foreign-education",
        "bank": "HDFC",
        "scheme_type": "education_loan",
        "scheme_name": "Education Loan for Foreign Education",
        "doc_type": "product_page",
        "js_render": False,
    },

    # ── ICICI Bank (static HTML, aiohttp works) ───────────────────────────────
    {
        "url": "https://www.icicibank.com/personal-banking/loans/education-loan/benefits-and-features",
        "bank": "ICICI",
        "scheme_type": "education_loan",
        "scheme_name": "Education Loan",
        "doc_type": "product_page",
        "js_render": False,
    },
    {
        "url": "https://www.icicibank.com/personal-banking/loans/education-loan/education-loan-faqs",
        "bank": "ICICI",
        "scheme_type": "education_loan",
        "scheme_name": "Education Loan FAQs",
        "doc_type": "faq_page",
        "js_render": False,
    },

    # ── SBI (static HTML, aiohttp works) ─────────────────────────────────────
    {
        "url": "https://sbi.co.in/web/personal-banking/loans/education-loans/student-loan-scheme",
        "bank": "SBI",
        "scheme_type": "education_loan",
        "scheme_name": "Student Loan Scheme",
        "doc_type": "product_page",
        "js_render": False,
    },
    {
        "url": "https://sbi.co.in/web/personal-banking/loans/education-loans/global-ed-vantage-scheme",
        "bank": "SBI",
        "scheme_type": "education_loan",
        "scheme_name": "Global Ed-Vantage Scheme",
        "doc_type": "product_page",
        "js_render": False,
    },
    # ── Bank of Baroda (JS-rendered, needs Playwright) ────────────────────────
    {
        "url": "https://www.bankofbaroda.in/personal-banking/loans/education-loan",
        "bank": "BOB",
        "scheme_type": "education_loan",
        "scheme_name": "Education Loan Overview",
        "doc_type": "product_page",
        "js_render": True,
    },
    {
        "url": "https://www.bankofbaroda.in/personal-banking/loans/education-loan/baroda-scholar",
        "bank": "BOB",
        "scheme_type": "education_loan",
        "scheme_name": "Baroda Scholar Foreign Education",
        "doc_type": "product_page",
        "js_render": True,
    },
    {
        "url": "https://www.bankofbaroda.in/personal-banking/loans/education-loan/baroda-gyan",
        "bank": "BOB",
        "scheme_type": "education_loan",
        "scheme_name": "Baroda Gyan Indian Education",
        "doc_type": "product_page",
        "js_render": True,
    },


    # ── Union Bank of India (static HTML, aiohttp works) ─────────────────────
    {
        "url": "https://www.unionbankofindia.co.in/english/Education-Loan.aspx",
        "bank": "UnionBank",
        "scheme_type": "education_loan",
        "scheme_name": "Education Loan",
        "doc_type": "product_page",
        "js_render": False,
    },

    # ── Axis Bank (JS-rendered, needs Playwright) ─────────────────────────────
    {
        "url": "https://www.axisbank.com/retail/loans/education-loan",
        "bank": "Axis",
        "scheme_type": "education_loan",
        "scheme_name": "Education Loan",
        "doc_type": "product_page",
        "js_render": True,
    },

    # ── Kotak Mahindra Bank (JS-rendered, needs Playwright) ───────────────────
    {
        "url": "https://www.kotak.com/en/personal-banking/loans/education-loan.html",
        "bank": "Kotak",
        "scheme_type": "education_loan",
        "scheme_name": "Education Loan",
        "doc_type": "product_page",
        "js_render": True,
    },

    # ── City Union Bank (static HTML, aiohttp works) ──────────────────────────
    {
        "url": "https://www.cityunionbank.com/cub-education-loan-vidya-vani",
        "bank": "CUB",
        "scheme_type": "education_loan",
        "scheme_name": "CUB Vidya Vani Education Loan",
        "doc_type": "product_page",
        "js_render": False,
    },
    {
        "url": "https://www.cityunionbank.com/student-education-loan",
        "bank": "CUB",
        "scheme_type": "education_loan",
        "scheme_name": "CUB Student Education Loan",
        "doc_type": "product_page",
        "js_render": False,
    },
]