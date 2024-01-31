import subprocess
import pkg_resources
import json

def get_installed_distributions():
    return [d for d in pkg_resources.working_set]

def remove_package(package):
    subprocess.check_call(["pip", "uninstall", "-y", package])

def install_package(package, target=None, no_depth=False):

    if no_depth == True:
        if target != None:
            subprocess.check_call(["pip", "install", "--target", target, package, "--no-deps"])
        else:
            subprocess.check_call(["pip", "install", package, "--no-deps"])
    else:
        if target != None:
            subprocess.check_call(["pip", "install", "--target", target, package])
        else:
            subprocess.check_call(["pip", "install", package])

def test_code(device="cpu"):
    exist_status = subprocess.call(["python", "dependency_test_script.py", "--device", device])
    if exist_status != 0:
        raise Exception
    
def install_requirement(init_packages, target=None):
    for package in init_packages:
        install_package(package, target)

def remove_requirement(remove_packages):
    for package in remove_packages:
        remove_package(package)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', type=bool, default=False, help='install init packages')
    parser.add_argument('--target', type=str, default=None, help='pip target option')
    parser.add_argument('--initpkg', nargs='+', type=str, default=[], help='init packages')
    parser.add_argument('--check', type=bool, default=False, help='check dependency packages')
    parser.add_argument('--device', type=str, default="cpu", help='device select cpu or cuda')
    parser.add_argument('--remove', type=bool, default=False, help='remove unnecessary packages')
    args = parser.parse_args()

    if args.check == True:
        packages = get_installed_distributions()

        necessary_packages = []
        unnecessary_packages = []
        for package in packages:
            if package.project_name == "pip":
                continue
            elif package.project_name in "nvidia-cuda-cupti-cu11":
                continue
            
            try:
                print(f"Removing {package}")
                remove_package(package.project_name)

                test_code(args.device)
                install_package(package.project_name, args.target, no_depth=True)
                unnecessary_packages.append(package.project_name)
            
            except:
                print(f"Failed to test code without {package.project_name}")
                necessary_packages.append(package.project_name)
                install_package(package.project_name, args.target, no_depth=True)
                
        print(necessary_packages)
        print(unnecessary_packages)
        data = {
            "necessary_packages": necessary_packages,
            "unnecessary_packages": unnecessary_packages
        }

        with open('dependency_check.json', mode='w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    elif args.install == True:
        init_packages = ["torch", "torchvision", "opencv-python", "kornia", "timm", "setuptools"]
        init_packages.extend(args.initpkg)
        install_requirement(init_packages, args.target)
    
    elif args.remove == True:
        with open("dependency_check.json", mode="r", encoding="utf-8") as f:
            file = json.load(f)
        unnecessary_packages = file["unnecessary_packages"]
        remove_requirement(unnecessary_packages)

    else:
        print("Not valid options")
        raise Exception
    