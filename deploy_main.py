"""Brings together deployment and model

Usage: python3.9 deploy_main.py <path-to-creds-file>

For  <path-to-creds-file>, fill out the information in
deployment/creds_template.json 
"""

import sys
from deployment.run_bot import deploy 

def print_usage():
    usage = "Usage: python3.9 main.py <cred-file-path>"
    print(usage)

def main():
    if len(sys.argv) != 2:
        print_usage()
        return -1

    deploy()



if __name__ == '__main__':
    main()

