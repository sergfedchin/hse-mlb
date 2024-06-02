import os


if __name__ == '__main__':
    for test in os.listdir('tests'):
        os.system(f'python main.py {os.path.join("tests", test)}')
