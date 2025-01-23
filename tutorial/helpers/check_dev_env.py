
import importlib.metadata
from packaging import requirements
import os

def check_install(requirements_file='../requirements.txt'):
    ''' check if required packages are installed based on a requirements file '''

    if not os.path.exists(requirements_file):
        print('Requirements file {} not found.'.format(requirements_file))
        return False

    missing_packages = []
    incorrect_versions = []

    with open(requirements_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    req = requirements.Requirement(line)
                    package_name = req.name
                    try:
                        version = importlib.metadata.version(package_name)
                         #check if there is a version specifier
                        if req.specifier:
                            if not req.specifier.contains(version):
                                incorrect_versions.append('{}  (installed: {}, required {})'.format(package_name, version, req.specifier))
                    except importlib.metadata.PackageNotFoundError:
                        missing_packages.append(package_name)
                except requirements.InvalidRequirement:
                    print('Invalid requirement: {}'.format(line))
                    return False
    if missing_packages:
        print('The following required packages are missing:')
        for package in missing_packages:
            print('- {}'.format(package))
        return False
    
    if incorrect_versions:
        print('The following packages have incorrect versions:')
        for package in incorrect_versions:
            print('- {}'.format(package))
        return False

    print('All required packages are installed.')
    return

def check_device():
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    return device