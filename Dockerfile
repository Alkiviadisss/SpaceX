# 1. Χρησιμοποιούμε μια ελαφριά έκδοση της Python
FROM python:3.9-slim

# 2. Ορίζουμε τον φάκελο εργασίας μέσα στο container
WORKDIR /app

# 3. Αντιγράφουμε τα requirements και τα κάνουμε install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Αντιγράφουμε όλο τον υπόλοιπο κώδικα (app.py, utils/ κλπ)
COPY . .

# 5. Ανοίγουμε την πόρτα 8501 (η προεπιλογή του Streamlit)
EXPOSE 8501

# 6. Η εντολή που θα τρέξει όταν ξεκινήσει το container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]