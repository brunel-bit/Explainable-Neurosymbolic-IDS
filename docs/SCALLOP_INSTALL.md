# Scallopy is distributed as a platform-specific wheel.
# Install the wheel matching the operating system, processor architecture,
# Python version and ABI before running the Scallop backend.
#
# Example:
# python -m pip install /path/to/scallopy-<version>-<python>-<abi>-<platform>.whl




cat > docs/SCALLOP_INSTALL.md <<'EOF'
# Installation de Scallopy

Le backend neurosymbolique utilise `scallopy`.

Scallopy n'est pas installé depuis PyPI dans la configuration actuelle du
projet. Il doit être compilé ou installé à partir d'un wheel compatible avec :

- le système d'exploitation ;
- l'architecture du processeur ;
- la version de Python ;
- l'ABI Python.

## Environnement validé

Le prototype a été validé avec :

- Python 3.12 ;
- scallopy 0.2.5 ;
- macOS Apple Silicon ARM64.

Le wheel utilisé pendant le développement était :

```text
scallopy-0.2.5-cp312-cp312-macosx_11_0_arm64.whl



Installation depuis un wheel comme ceci 
        python -m pip install /chemin/vers/scallopy-0.2.5-cp312-cp312-macosx_11_0_arm64.whl


        pour moi cetait 
            pip install ~/PythonProject/scallop/target/wheels/scallopy-0.2.5-cp312-cp312-macosx_11_0_arm64.whl