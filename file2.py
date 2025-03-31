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
        self.root.geometry('500x400')

        self.input_dir = tk.StringVar()

        tk.Label(root, text='Select Folder to Organize:', font=('Arial', 12)).pack(pady=10)
        tk.Entry(root, textvariable=self.input_dir, width=40).pack(pady=5)
        tk.Button(root, text='Browse', command=self.browse_folder).pack(pady=5)
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

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.input_dir.set(folder_path)

    def organize_files(self):
        input_folder = self.input_dir.get()

        if not input_folder:
            messagebox.showerror('Error', 'Please select a folder to organize')
            return

        if not self.model:
            messagebox.showerror('Error', 'Model is not loaded properly')
            return

        folder_tree = nx.DiGraph()
        folder_tree.add_node(input_folder)

        file_features = []
        file_paths = []

        try:
            for root_dir, _, files in os.walk(input_folder):
                for file in files:
                    file_path = os.path.join(root_dir, file)

                    if not os.path.exists(file_path):
                        continue

                    try:
                        category = self.categorize_file(file_path)
                        self.move_file(file_path, category)

                        file_features.append(self.extract_features(file_path))
                        file_paths.append(file_path)

                    except Exception as e:
                        print(f"Failed to process {file_path}: {e}")

            self.generate_folder_structure(file_features, file_paths)

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

        # Categorizing based on file name patterns
        if 'invoice' in file_name or 'bill' in file_name:
            return 'Invoices'
        elif 'report' in file_name or 'summary' in file_name:
            return 'Reports'
        elif re.search(r'\d{4}-\d{2}-\d{2}', file_name):
            return 'Dated Files'

        # Checking properties (e.g., creation date)
        creation_time = os.path.getctime(file_path)
        creation_date = datetime.datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d')
        if datetime.datetime.now().strftime('%Y') in creation_date:
            return 'Recent Files'

        return 'Others'

    def move_file(self, file_path, category):
        try:
            output_folder = os.path.join(self.input_dir.get(), category)
            os.makedirs(output_folder, exist_ok=True)

            if os.path.exists(file_path):
                shutil.move(file_path, os.path.join(output_folder, os.path.basename(file_path)))
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Failed to move {file_path}: {e}")

    def extract_features(self, file_path):
        return [len(file_path), os.path.getsize(file_path)]

    def generate_folder_structure(self, file_features, file_paths):
        if len(file_features) < 2:
            return

        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(file_features)

        kmeans = KMeans(n_clusters=5)
        labels = kmeans.fit_predict(features_2d)

        folder_tree = nx.DiGraph()

        for index, file_path in enumerate(file_paths):
            folder_name = f"Cluster_{labels[index]}"
            if not folder_tree.has_node(folder_name):
                folder_tree.add_node(folder_name)

            folder_tree.add_edge(folder_name, file_path)

        self.display_folder_tree(folder_tree)

    def display_folder_tree(self, folder_tree):
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(folder_tree)
        nx.draw(folder_tree, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=8, font_color='black')
        plt.title('Folder Structure')
        plt.show()


if __name__ == '__main__':
    root = tk.Tk()
    app = FileOrganizerApp(root)
    root.mainloop()
