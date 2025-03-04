import tkinter as tk
from tkinter import messagebox, scrolledtext
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from textblob import TextBlob
import heapq

# Ensure necessary NLTK resources are available
nltk_resources = ["punkt", "stopwords"]
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

class NewsSummarizationApp:
    def __init__(self, master):
        self.master = master
        master.title("News Article Summarization Tool")
        master.geometry("800x600")

        # Input Text Area
        self.input_label = tk.Label(master, text="Enter News Article:", font=('Arial', 12))
        self.input_label.pack(pady=(10, 5))

        self.input_text = scrolledtext.ScrolledText(master, height=10, width=90, wrap=tk.WORD)
        self.input_text.pack(pady=10)

        # Buttons Frame
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=10)

        # Summarize Button
        self.summarize_button = tk.Button(
            self.button_frame, 
            text="Summarize", 
            command=self.summarize_article, 
            bg='#4CAF50', 
            fg='white'
        )
        self.summarize_button.pack(side=tk.LEFT, padx=10)

        # Sentiment Analysis Button
        self.sentiment_button = tk.Button(
            self.button_frame, 
            text="Analyze Sentiment", 
            command=self.analyze_sentiment, 
            bg='#2196F3', 
            fg='white'
        )
        self.sentiment_button.pack(side=tk.LEFT, padx=10)

        # Output Text Area
        self.output_label = tk.Label(master, text="Output:", font=('Arial', 12))
        self.output_label.pack(pady=(10, 5))

        self.output_text = scrolledtext.ScrolledText(master, height=10, width=90, wrap=tk.WORD)
        self.output_text.pack(pady=10)

    def preprocess_text(self, text):
        """
        Preprocess the input text by removing stopwords and converting to lowercase.
        """
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())

        # Remove stopwords and non-alphabetic tokens
        return [word for word in words if word.isalnum() and word not in stop_words]

    def extract_keywords(self, processed_words):
        """
        Extract most frequent keywords.
        """
        return FreqDist(processed_words)

    def summarize_article(self):
        """
        Generate summary of the article.
        """
        try:
            # Get input text
            article = self.input_text.get("1.0", tk.END).strip()
            
            if not article:
                messagebox.showwarning("Warning", "Please enter an article to summarize.")
                return

            # Tokenize sentences
            sentences = sent_tokenize(article)
            
            # Preprocess text and extract keywords
            processed_words = self.preprocess_text(article)
            word_frequencies = self.extract_keywords(processed_words)

            # Calculate sentence scores based on word frequencies
            sentence_scores = {}
            for sentence in sentences:
                for word in word_tokenize(sentence.lower()):
                    if word in word_frequencies:
                        sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]

            # Get top N sentences (max 40% of total sentences, but at least 1)
            num_sentences = max(1, int(0.4 * len(sentences)))
            summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

            # Combine summary sentences
            summary = ' '.join(summary_sentences)
            
            # Display summary
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Summary:\n{summary}")
            self.output_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def analyze_sentiment(self):
        """
        Perform sentiment analysis on the article.
        """
        try:
            # Get input text
            article = self.input_text.get("1.0", tk.END).strip()
            
            if not article:
                messagebox.showwarning("Warning", "Please enter an article to analyze.")
                return

            # Use TextBlob for sentiment analysis
            blob = TextBlob(article)
            sentiment = blob.sentiment.polarity

            # Interpret sentiment
            sentiment_label = (
                "Positive" if sentiment > 0 else 
                "Negative" if sentiment < 0 else 
                "Neutral"
            )

            # Display sentiment analysis results
            result = (f"Sentiment Analysis:\n"
                      f"Polarity: {sentiment:.2f}\n"
                      f"Subjectivity: {blob.sentiment.subjectivity:.2f}\n"
                      f"Overall Sentiment: {sentiment_label}")

            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, result)
            self.output_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = NewsSummarizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
