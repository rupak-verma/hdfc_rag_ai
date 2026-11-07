"""Builds rag/evaluation/test_questions.json (DRAFT written by Claude; humans must review - see 'review_status').
Run: python -m rag.evaluation.make_test_questions
answerable: question + the document that answers it (+ any document with an identical answer is accepted automatically)
absent    : questions whose answer is NOT in the knowledge base; the right behaviour is to abstain
            kinds: fake_product, off_topic (easy), near_topic (related documents exist, none answers it - hard), personal_data
Split: half 'dev' (used to set the abstention thresholds) and half 'test' (used only to report results)."""
import json, random, re
import pandas as pd

ANSWERABLE = [  # (document_id, question, style)   style: para = few shared words, kw = shares key words
 ("DOC-00559","What paperwork do I take to the bank to get a student loan?","para"),
 ("DOC-00520","What is it called when a company lets current shareholders buy extra shares at a discount?","para"),
 ("DOC-00541","How can I generate a PIN for my new ATM card?","para"),
 ("DOC-00210","Tell me about government gold bonds that pay interest on top of gold price returns","para"),
 ("DOC-00667","Can I put more than the usual 12% of my pay into my provident fund?","kw"),
 ("DOC-00716","Why does the RBI buy or sell dollars when the rupee swings sharply?","para"),
 ("DOC-00958","Can I divide one fixed deposit into several smaller deposits?","para"),
 ("DOC-00426","How many days of cash outflow must banks cover with liquid assets?","kw"),
 ("DOC-00392","What does a bank's balance sheet show?","kw"),
 ("DOC-00280","Which loan pays for salaries and inventory of a small business?","para"),
 ("DOC-00465","Which mutual fund invests mainly in stocks?","para"),
 ("DOC-00539","Is Aadhaar compulsory for opening a bank account?","para"),
 ("DOC-00458","Who regulates the stock market in India?","para"),
 ("DOC-00866","What do you call a borrower who can pay but deliberately does not?","para"),
 ("DOC-00961","How do government subsidies reach my bank account directly?","para"),
 ("DOC-00814","How many days after buying do shares reach my demat account?","para"),
 ("DOC-00069","Why is a banker's cheque safer than a personal cheque?","kw"),
 ("DOC-00277","Can a bank sell a defaulter's secured property without going to court?","para"),
 ("DOC-00139","How do apps like Paytm keep my money?","para"),
 ("DOC-00250","What document confirms that my loan has been approved?","para"),
 ("DOC-00540","I never received a debit card after opening my savings account. What should I do?","para"),
 ("DOC-00098","Is tax cut from the interest I earn on my savings account?","para"),
 ("DOC-00712","What is a letter of credit that pays at once when shipping documents are shown?","para"),
 ("DOC-00763","Where are IPOs and new bond issues sold for the first time?","para"),
 ("DOC-00661","Who is licensed by IRDAI to sell policies of only one insurer?","para"),
 ("DOC-00704","What is an account used to trade currency pairs through a broker?","para"),
 ("DOC-01859","Can I choose the date of my monthly recurring deposit instalment?","kw"),
 ("DOC-01221","Where can I use my EasyShop International Debit Card?","kw"),
 ("DOC-02470","What are the foreclosure charges on a Loan Against Rental Receivable?","kw"),
 ("DOC-02441","Can I get a loan for any brand of tractor?","para"),
 ("DOC-02155","Does a KYC Acknowledgement ever expire?","kw"),
 ("DOC-02812","What does the loss of checked baggage benefit cover?","kw"),
 ("DOC-01699","Will a Defence ID be accepted as photo proof for a defence salary account?","kw"),
 ("DOC-01440","Is there a fee for paying bills at an ATM through BillPay?","para"),
 ("DOC-02179","Can I still go through my broker to apply for an IPO?","kw"),
 ("DOC-02410","Can I repay my student loan using post-dated cheques or a standing instruction?","para"),
 ("DOC-02648","How do I follow the daily NAV of my policy fund?","kw"),
 ("DOC-02702","When does my Accidental Protection Plan Hospital Cash policy begin?","kw"),
 ("DOC-01240","What should I do about a void transaction on my Platinum Debit Card?","kw"),
 ("DOC-01866","Can I get an overdraft against my recurring deposit?","kw"),
 ("DOC-01922","What happens if I miss six Dream Deposit instalments in a row?","kw"),
 ("DOC-01282","Do I need an account with the bank to buy a Giftplus card?","kw"),
 ("DOC-01871","Can I take money out of my recurring deposit before maturity?","para"),
 ("DOC-02241","What is a balance transfer home loan?","kw"),
 ("DOC-02223","What does own contribution mean in a home loan?","kw"),
 ("DOC-01904","Can I add a nominee to my safe deposit locker?","para"),
 ("DOC-02118","How do I apply for a rights issue through ASBA?","kw"),
 ("DOC-01402","Where can I use the Regalia ForexPlus Card?","kw"),
 ("DOC-01506","Can I set up a standing instruction for postpaid bill payments too?","kw"),
 ("DOC-02295","How are home loans for under-construction property paid out?","para"),
 ("DOC-03069","Why do I see only six months of transactions after registering mPassbook on a new phone?","para"),
 ("DOC-02400","Am I eligible for an education loan to study in India?","kw"),
 ("DOC-01994","How do I change the bank details linked to my demat account?","para"),
 ("DOC-02716","What is a benefit policy in insurance?","kw"),
 ("DOC-02672","What is the top-up option that raises my cover every year?","para"),
 ("DOC-02856","Who can buy a multi-year home insurance policy?","para"),
 ("DOC-02024","Do dematerialised shares carry certificate numbers?","kw"),
 ("DOC-01778","Do I need a separate Apex Current Account at every office location?","kw"),
 ("DOC-02101","Will I get the same share certificates back after rematerialisation?","kw"),
]
ABSENT = [  # (question, kind)
 ("What is the annual fee on the HDFC Platinum Elite Supreme credit card?","fake_product"),
 ("What interest rate does the HDFC Diamond Millionaire savings account pay?","fake_product"),
 ("How do I buy Bitcoin through my bank account?","near_topic"),
 ("What is the weather in Mumbai today?","off_topic"),
 ("Give me a recipe for chicken biryani.","off_topic"),
 ("Who won the cricket world cup?","off_topic"),
 ("What is the capital of France?","off_topic"),
 ("Write a poem about the monsoon.","off_topic"),
 ("How do I file my GST return online?","near_topic"),
 ("What is the IFSC code of the ICICI Bank Pune branch?","near_topic"),
 ("How do I renew my passport?","near_topic"),
 ("Can I get a loan to buy a private jet?","near_topic"),
 ("Does the bank offer pet insurance for dogs?","near_topic"),
 ("How do I reset my Netflix password?","off_topic"),
 ("Explain quantum computing in simple terms.","off_topic"),
 ("How much does the bank's CEO earn?","near_topic"),
 ("What documents do I need for a Canada tourist visa?","near_topic"),
 ("How do I book a flight ticket online?","off_topic"),
 ("How do I change a flat tyre on my car?","off_topic"),
 ("What are the Sunday opening hours of the Fort branch in Mumbai?","near_topic"),
 ("What is the phone number of the Chennai branch manager?","near_topic"),
 ("Will the bank be closed on Diwali this year?","near_topic"),
 ("What is my current account balance?","personal_data"),
 ("When will my last transaction reach the beneficiary?","personal_data"),
]
D = pd.read_csv("rag/data/approved/canonical_documents.csv", dtype=str).fillna("")
ids = set(D.document_id)
for d, _, _ in ANSWERABLE: assert d in ids, f"unknown gold document {d}"
norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())
D["na"] = D.text.map(norm)
items = []
for d, q, style in ANSWERABLE:
    same = D[D.na == D[D.document_id == d].na.iloc[0]].document_id.tolist()      # identical answers are all acceptable
    items.append(dict(type="answerable", question=q, relevant_document_ids=sorted(set([d] + same)), style=style))
for q, kind in ABSENT: items.append(dict(type="absent", question=q, relevant_document_ids=[], style=kind))
random.Random(11).shuffle(items)
cnt = {"answerable": 0, "absent": 0}
for i, it in enumerate(items, 1):                                  # alternate dev/test within each type
    it["id"] = f"Q{i:03d}"; it["split"] = "dev" if cnt[it["type"]] % 2 == 0 else "test"; cnt[it["type"]] += 1
    it["review_status"] = "DRAFT_BY_CLAUDE_NEEDS_HUMAN_REVIEW"
json.dump(items, open("rag/evaluation/test_questions.json", "w"), indent=1, ensure_ascii=False)
print(len(items), {k: sum(1 for x in items if x["type"] == k) for k in ("answerable", "absent")},
      {s: sum(1 for x in items if x["split"] == s) for s in ("dev", "test")})
