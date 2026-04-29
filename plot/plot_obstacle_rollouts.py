import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='results/obstacle')
    parser.parse_args()

if __name__ == '__main__':
    main()
