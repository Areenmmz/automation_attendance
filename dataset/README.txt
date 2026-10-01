Put one folder per person here. The folder name is the name written to the CSV.

dataset/
    Name1/
        Name1_001.jpg
        Name1_002.jpg
        ...
    Name2/
        Name2_001.jpg
        ...

Either run:  python register_user.py --name "Meraj"
or copy 10-30 clear photos of the person into their folder by hand.
After adding/removing images, main.py rebuilds the embeddings automatically.
