"""
Check Installed Package Versions
Displays current versions of all packages used in the project.
"""

import sys
import importlib.metadata
from datetime import datetime

print("="*80)
print("INSTALLED PACKAGE VERSIONS")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python: {sys.version.split()[0]}")
print("")

# Define required packages
packages = {
    'Deep Learning': ['torch', 'torchvision'],
    'Computer Vision': ['cv2', 'mediapipe', 'PIL'],
    'Data Science': ['numpy', 'pandas', 'sklearn'],
    'Visualization': ['matplotlib', 'seaborn'],
    'Utilities': ['tqdm', 'tabulate'],
    'Office Files': ['docx', 'pptx', 'openpyxl']
}

# Package name mapping (import name -> pip name)
pip_names = {
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'PIL': 'Pillow',
    'docx': 'python-docx',
    'pptx': 'python-pptx'
}

all_versions = {}

for category, package_list in packages.items():
    print(f"\n{category}:")
    print("-" * 80)
    
    for package in package_list:
        try:
            # Try to import
            if package == 'cv2':
                import cv2
                version = cv2.__version__
            elif package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif package == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                module = __import__(package)
                version = module.__version__
            
            pip_name = pip_names.get(package, package)
            print(f"  ✓ {pip_name:<25} {version}")
            all_versions[pip_name] = version
            
        except ImportError:
            pip_name = pip_names.get(package, package)
            print(f"  ✗ {pip_name:<25} NOT INSTALLED")
            all_versions[pip_name] = None
        except AttributeError:
            # No __version__ attribute, try importlib.metadata
            try:
                pip_name = pip_names.get(package, package)
                version = importlib.metadata.version(pip_name)
                print(f"  ✓ {pip_name:<25} {version}")
                all_versions[pip_name] = version
            except:
                pip_name = pip_names.get(package, package)
                print(f"  ? {pip_name:<25} INSTALLED (version unknown)")
                all_versions[pip_name] = "unknown"

# Generate requirements.txt content
print("\n" + "="*80)
print("REQUIREMENTS.TXT CONTENT")
print("="*80)
print("")

requirements_lines = [
    "# WHO Handwashing Step Classification - Requirements",
    f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "# Core",
    "python>=3.8,<3.12",
    "",
    "# Deep Learning"
]

for package in ['torch', 'torchvision']:
    if package in all_versions and all_versions[package]:
        requirements_lines.append(f"{package}=={all_versions[package]}")
    else:
        requirements_lines.append(f"{package}  # VERSION NOT DETECTED")

requirements_lines.extend([
    "",
    "# Computer Vision"
])

for package in ['opencv-python', 'mediapipe', 'Pillow']:
    if package in all_versions and all_versions[package]:
        requirements_lines.append(f"{package}=={all_versions[package]}")
    else:
        requirements_lines.append(f"{package}  # VERSION NOT DETECTED")

requirements_lines.extend([
    "",
    "# Data Science"
])

for package in ['numpy', 'pandas', 'scikit-learn']:
    if package in all_versions and all_versions[package]:
        requirements_lines.append(f"{package}=={all_versions[package]}")
    else:
        requirements_lines.append(f"{package}  # VERSION NOT DETECTED")

requirements_lines.extend([
    "",
    "# Visualization"
])

for package in ['matplotlib', 'seaborn']:
    if package in all_versions and all_versions[package]:
        requirements_lines.append(f"{package}=={all_versions[package]}")
    else:
        requirements_lines.append(f"{package}  # VERSION NOT DETECTED")

requirements_lines.extend([
    "",
    "# Utilities"
])

for package in ['tqdm', 'tabulate']:
    if package in all_versions and all_versions[package]:
        requirements_lines.append(f"{package}=={all_versions[package]}")
    else:
        requirements_lines.append(f"{package}  # VERSION NOT DETECTED")

requirements_lines.extend([
    "",
    "# Office Files"
])

for package in ['python-docx', 'python-pptx', 'openpyxl']:
    if package in all_versions and all_versions[package]:
        requirements_lines.append(f"{package}=={all_versions[package]}")
    else:
        requirements_lines.append(f"{package}  # VERSION NOT DETECTED")

# Display
for line in requirements_lines:
    print(line)

# Save to file
with open('requirements.txt', 'w') as f:
    f.write('\n'.join(requirements_lines))

print("\n" + "="*80)
print("SAVED!")
print("="*80)
print("\n✓ requirements.txt created")

# Create Kaggle version (no pinned versions)
kaggle_lines = [
    "# WHO Handwashing - Kaggle/Google Colab Requirements",
    "# No version pinning for compatibility",
    "",
    "# Deep Learning",
    "torch",
    "torchvision",
    "",
    "# Computer Vision",
    "opencv-python",
    "mediapipe",
    "Pillow",
    "",
    "# Data Science",
    "numpy",
    "pandas",
    "scikit-learn",
    "",
    "# Visualization",
    "matplotlib",
    "seaborn",
    "",
    "# Utilities",
    "tqdm",
    "tabulate",
    "",
    "# Office Files",
    "python-docx",
    "python-pptx",
    "openpyxl"
]

with open('requirements_kaggle.txt', 'w') as f:
    f.write('\n'.join(kaggle_lines))

print("✓ requirements_kaggle.txt created (for Kaggle/Colab)")

print("\nRecommendation:")
print("  • Use requirements.txt for local development")
print("  • Use requirements_kaggle.txt for Kaggle/Colab")