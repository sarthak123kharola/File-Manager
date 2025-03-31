import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import mimetypes
from PIL import Image
import pytesseract
import requests
import io
import zipfile
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re
import datetime


class FileOrganizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title('File Organizer App')
        self.root.geometry('600x450')

        self.input_files = []

        tk.Label(root, text='Select Files or Folders to Organize:', font=('Arial', 12)).pack(pady=10)
        tk.Button(root, text='Browse Files', command=self.browse_files).pack(pady=5)
        tk.Button(root, text='Browse Folders', command=self.browse_folders).pack(pady=5)
        tk.Button(root, text='Organize Files', command=self.organize_files, font=('Arial', 12, 'bold')).pack(pady=20)

        self.model = self.load_pretrained_model()

    def load_pretrained_model(self):
        try:
            if os.path.exists('file_classifier_model.pkl'):
                model = joblib.load('file_classifier_model.pkl')
            else:
                data = ["Document text example", "Image file content", "Video file example", "Code file example"]
                labels = ["Text Files", "Images", "Videos", "Code Files"]

                model = make_pipeline(TfidfVectorizer(), MultinomialNB())
                model.fit(data, labels)

                joblib.dump(model, 'file_classifier_model.pkl')
            return model
        except Exception as e:
            messagebox.showerror('Error', f'Model loading failed: {e}')
            return None

    def browse_files(self):
        file_paths = filedialog.askopenfilenames()
        if file_paths:
            self.input_files.extend(file_paths)

    def browse_folders(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            for root_dir, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root_dir, file)
                    self.input_files.append(file_path)

    def organize_files(self):
        if not self.input_files:
            messagebox.showerror('Error', 'Please select files or folders to organize')
            return

        if not self.model:
            messagebox.showerror('Error', 'Model is not loaded properly')
            return

        try:
            for file_path in self.input_files:
                if not os.path.exists(file_path):
                    continue
                try:
                    category = self.categorize_file(file_path)
                    self.move_file(file_path, category)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

            messagebox.showinfo('Success', 'Files Organized Successfully!')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def categorize_file(self, file_path):
        file_name = os.path.basename(file_path).lower()
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type:
            if mime_type == 'application/pdf':
                return 'PDF Files'
            if mime_type.startswith('image'):
                try:
                    text_content = pytesseract.image_to_string(Image.open(file_path))
                    return self.model.predict([text_content])[0]
                except:
                    return 'Images'

            extension = os.path.splitext(file_path)[1].lower().replace('.', '')
            return extension.upper() + ' Files'

        if 'invoice' in file_name or 'bill' in file_name:
            return 'Invoices'
        elif 'report' in file_name or 'summary' in file_name:
            return 'Reports'
        elif re.search(r'\d{4}-\d{2}-\d{2}', file_name):
            return 'Dated Files'

        creation_time = os.path.getctime(file_path)
        creation_date = datetime.datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d')
        if datetime.datetime.now().strftime('%Y') in creation_date:
            return 'Recent Files'

        return 'Others'

    def move_file(self, file_path, category):
        try:
            output_folder = os.path.join(os.path.dirname(file_path), category)
            os.makedirs(output_folder, exist_ok=True)
            shutil.move(file_path, os.path.join(output_folder, os.path.basename(file_path)))
        except Exception as e:
            print(f"Failed to move {file_path}: {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = FileOrganizerApp(root)
    root.mainloop()