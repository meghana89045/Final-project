# Final-project
AI-Powered Resume Ranker
Objective
The primary goal of this project is to leverage Natural Language Processing (NLP) techniques to effectively rank resumes for specific job profiles. This tool will assist HR professionals in identifying the best-fit candidates efficiently.

Tools
We will utilize the following technologies:

Python: For the overall programming and development.
SpaCy: For natural language processing tasks.
Scikit-learn: For implementing machine learning algorithms.
Flask: To create a web application interface.
Mini-Guide
Here’s a step-by-step guide to building the Resume Ranker:

Extract Text from PDF Resumes

Utilize libraries like PyPDF2 or pdfminer to extract text from resumes saved in PDF format.
Preprocess Text using SpaCy

Use SpaCy for tokenization, lemmatization, and removing stop words to clean and standardize the text data.
Vectorize using TF-IDF

Implement the Term Frequency-Inverse Document Frequency (TF-IDF) method to convert the cleaned text into a numerical format suitable for machine learning.
Create a Scoring Algorithm

Develop a scoring algorithm based on keywords derived from the job description to assess how well each candidate’s resume matches the requirements.
Rank Candidates

Use the scoring algorithm to rank candidates based on their fitted score.
Build a Web UI

Create a user-friendly web interface using Flask that allows HR personnel to upload resumes and see the rank of each candidate.
Add Download Option for HR Reports

Include functionality for HR users to download and save the ranked reports in a suitable format.
