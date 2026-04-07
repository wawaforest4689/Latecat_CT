# setup.py
import os

from setuptools import find_packages, setup
from CodonTransformer.CodonData import prepare_training_data,prepare_data_from_fasta,\
    prepare_data_from_fasta_for_infer,split_dataset_by_structure

def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, "README.md")

    with open(readme_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__=='__main__':
    """
    setup(
        name="CodonTransformer",
        version="1.6.7",
        packages=find_packages(),
        install_requires=read_requirements(),
        author="Adibvafa Fallahpour",
        author_email="Adibvafa.fallahpour@mail.utoronto.ca",
        description=(
            "The ultimate tool for codon optimization, "
            "transforming protein sequences into optimized DNA sequences "
            "specific for your target organisms."
        ),
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        url="https://github.com/adibvafa/CodonTransformer",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.9",
    )
    """
    # prepare_training_data(os.path.join(os.getcwd(),'dataset'), os.path.join(os.getcwd(),'dataset/training_data.jsonl'))
    # print(os.path.join(os.getcwd(), 'dataset\Tests.xlsx'))
    # prepare_training_data(os.path.join(os.getcwd(), 'dataset/Test_1214.xlsx'),os.path.join(os.getcwd(), 'dataset/testing_1214.jsonl'),False)
    # prepare_data_from_fasta(os.path.join(os.getcwd(), 'dataset/SRR33141438.fasta'),savepath=os.path.join(os.getcwd(),'dataset/training_pichia.jsonl'))
    # prepare_data_from_fasta_for_infer(os.path.join(os.getcwd(), 'dataset/seqs.fasta'),savepath='dataset/testing_demo_1213.jsonl')
    split_dataset_by_structure(dataset_path='dataset/training_data.jsonl',cover=0.5)
    # a=[]
    # a.append([3 for i in range(10)])
    # print(len(a),a)




