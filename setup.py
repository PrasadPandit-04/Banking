from setuptools import setup, find_packages

setup(
    name='Banking',
    version=0.1,
    packages=find_packages(),
)


"""
SETUP INSTRUCTIONS:
------------------
1. PURPOSE:
   - Makes 'src/' importable anywhere in the project (e.g., `from src.preprocessing import ...`).
   - Required for notebooks, scripts, and apps to work without path hacks.

2. INSTALL:
   - Run ONCE per project (in terminal):
     ```
     cd /path/to/my_project    # Navigate to this folder
     pip install -e .          # Install in editable mode (-e is critical!) --> Creates a link (symlink) to your project instead of copying files.
     ```
3. VERIFY:
    - Run in terminal:
    ```
    pip list | grep my_project  # Should show a path (e.g., `my_project 0.1 (from /path/to/project)`)
    ```

4. KEY NOTES / INSTRUCTIONS:
   - Use `-e` (editable) so code changes apply instantly.
   - If you accidentally run `pip install .` (without -e):
     1. Uninstall: `pip uninstall <package_name>`  # In this case my_project
     2. Reinstall correctly: `pip install -e .`
""" 