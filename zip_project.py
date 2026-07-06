import os
import zipfile

def zip_project(output_filename="project_code.zip"):
    # Thu muc goc can nen
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cac thu muc va file can nen (chi nen code va config)
    include_dirs = ["src", "configs", "scripts"]
    include_files = ["requirements.txt", "README.md", "colab_guide.md"]
    
    print(f"Creating zip file: {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Nen cac thu muc code
        for d in include_dirs:
            dir_path = os.path.join(root_dir, d)
            if os.path.exists(dir_path):
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Tinh duong dan tuong doi de giai nen dung cau truc
                        arcname = os.path.relpath(file_path, root_dir)
                        zipf.write(file_path, arcname)
                        
        # Nen cac file cau hinh goc
        for f in include_files:
            file_path = os.path.join(root_dir, f)
            if os.path.exists(file_path):
                zipf.write(file_path, f)
                
    print(f"Zip file created successfully: {output_filename}!")

if __name__ == "__main__":
    zip_project()
