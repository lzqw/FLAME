import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.parse_args()

if __name__ == '__main__':
    main()
